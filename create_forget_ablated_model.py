"""
Build a new HF checkpoint whose MLP output projections remove SNMF forget directions.

Expected pipeline (matches the repo shell scripts):

  1. ``train_snmf.sh`` → writes ``outputs/snmf_train_results/layer_*/snmf_factors.pt``
     with ``mode=mlp_intermediate`` (required: ``F`` lives in the same space as
     ``mlp.down_proj`` input).
  2. ``run_analyze_snmf_results.sh`` → writes ``layer_*/feature_analysis_supervised.json``
     (``role_label`` per latent, from ``analyze_snmf_results.py``).
  3. This script → reads ``F`` + ``role_label``, applies ``W_V <- W_V @ P_perp``, saves a new model.

Use the **same** ``--model-path`` as training/analysis and the **same** ``--results-dir``
as ``--output-dir`` / ``RESULTS_DIR`` (default: ``outputs/snmf_train_results``).

Mathematical setup (matches the cited write-up):
  - Each SNMF column z_i ∈ ℝ^{d_mlp} is a direction in the post-activation (neuron) space.
  - The MLP output map is y = W_V @ x with x the intermediate activation (Gemma: down_proj).
  - To stop any activation component along z_i from reaching the residual stream via W_V, use
        W_V^{new} = W_V @ (I - P_i),
    where P_i = z_i z_i^T / ||z_i||^2 is orthogonal projection onto span{z_i}.

For multiple forget features {z_1,…,z_k}, use the projector onto their span:
    P_span = Z (Z^T Z + λ I)^{-1} Z^T,   Z = [z_1 | … | z_k],
    P_perp = I - P_span,
    W_V^{new} = W_V @ P_perp.

This edits weights only (no runtime hooks). Neurons are not zeroed; the subspace orthogonal
to the forget directions is unchanged for how it maps through W_V.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, Set

import torch

from evaluation.eveluate_model import run_standalone_eval
from model_utils import load_local_model
from utils import resolve_device, sorted_numeric_layer_dirs


def _load_role_map(layer_dir: Path) -> Dict[int, str]:
    path = layer_dir / "feature_analysis_supervised.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run analyze_snmf_results.py on --results-dir first "
            f"(e.g. run_analyze_snmf_results.sh) so each layer has role_label entries."
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v.get("role_label", "unknown")) for k, v in raw.items()}


def _forget_feature_matrix(
    layer_dir: Path,
    forget_roles: Set[str],
) -> torch.Tensor | None:
    """Returns Z of shape (d_mlp, k) with columns z_i from F, or None if nothing to remove."""
    ckpt_path = layer_dir / "snmf_factors.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mode = ckpt.get("mode", "mlp_intermediate")
    if mode != "mlp_intermediate":
        raise ValueError(
            f"{ckpt_path}: checkpoint mode is {mode!r}, but this script only applies to "
            f"mlp_intermediate (F must match down_proj input). Re-train with train_snmf.py "
            f"--mode mlp_intermediate (see train_snmf.sh)."
        )
    F = ckpt["F"].float().cpu()
    if F.ndim != 2:
        raise ValueError(f"Unexpected F shape in {ckpt_path}: {tuple(F.shape)}")

    roles = _load_role_map(layer_dir)
    k_all = F.shape[1]
    forget_cols = sorted(i for i, r in roles.items() if r in forget_roles and 0 <= i < k_all)
    if not forget_cols:
        return None
    Z = F[:, forget_cols].contiguous()
    return Z


def orthogonal_projector_complement(
    Z: torch.Tensor,
    ridge_lambda: float,
) -> torch.Tensor:
    """
    P_perp = I - Z (Z^T Z + λ I)^{-1} Z^T  ∈ ℝ^{d×d}, with Z ∈ ℝ^{d×k}.

    For k=1 this equals I - z z^T / (||z||^2 + λ) (≈ paper formula when λ=0).
    """
    d, k = Z.shape
    device, dtype = Z.device, Z.dtype
    I_d = torch.eye(d, device=device, dtype=dtype)
    if k == 0:
        return I_d
    g = Z.T @ Z + ridge_lambda * torch.eye(k, device=device, dtype=dtype)
    inv = torch.linalg.solve(g, torch.eye(k, device=device, dtype=dtype))
    p_span = Z @ inv @ Z.T
    return I_d - p_span


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply W_V <- W_V @ P_perp to remove SNMF forget directions from MLP output.",
        epilog=(
            "Typical use after train_snmf.sh and run_analyze_snmf_results.sh: same defaults "
            "as those scripts (model + outputs/snmf_train_results)."
        ),
    )
    p.add_argument(
        "--model-path",
        type=str,
        default="local_models/gemma-2-0.3B_reference_model",
        help="Same base model as train_snmf.py / analyze_snmf_results.py (default: train_snmf.sh).",
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default="outputs/snmf_train_results",
        help="SNMF output dir with layer_*/ (default: train_snmf.sh --output-dir).",
    )
    p.add_argument(
        "--save-path",
        type=str,
        default="local_models/gemma-2-0.3B_forget_ablated",
        help="Directory for save_pretrained (default: sibling of reference model).",
    )
    p.add_argument(
        "--forget-roles",
        type=str,
        nargs="+",
        default=["mult_forget", "div_forget", "forget_mixed"],
    )
    p.add_argument(
        "--ridge-lambda",
        type=float,
        default=1e-6,
        help="Tikhonov on Z^T Z when building the span projector (stability).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for loading the model during weight edit (auto: cuda if usable, else cpu; "
        "matches analyze_snmf_results / utils.resolve_device for old GPUs).",
    )
    p.add_argument(
        "--metadata-out",
        type=str,
        default="",
        help="Optional JSON path describing per-layer forget column counts.",
    )
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Do not run evaluation/eveluate_model.py before/after ablation.",
    )
    p.add_argument(
        "--eval-device",
        type=str,
        default="auto",
        help="Device for standalone eval (passed through to eveluate_model: auto|cuda|cpu).",
    )
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
        help="Batch size for CE leg of arithmetic eval.",
    )
    p.add_argument(
        "--eval-max-length",
        type=int,
        default=256,
    )
    p.add_argument(
        "--eval-cache-dir",
        type=str,
        default="./cache",
    )
    p.add_argument(
        "--eval-dataset-cache-dir",
        type=str,
        default="./cache",
    )
    p.add_argument(
        "--eval-eng-valid-file",
        type=str,
        default="data/valid_eng.jsonl",
        help="Relative to repo root if not absolute (same default as eveluate_model.py).",
    )
    p.add_argument(
        "--eval-json-out",
        type=str,
        default="",
        help="Write before/after metrics JSON here (default: <save-path>/ablation_eval_comparison.json).",
    )
    return p.parse_args()


def _summarize_eval(d: Dict[str, Any]) -> Dict[str, float]:
    """Keep scalar metrics for a compact table."""
    out: Dict[str, float] = {}
    for k, v in d.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
    return out


def _print_eval_comparison(before: Dict[str, Any], after: Dict[str, Any]) -> None:
    keys = sorted(set(before.keys()) | set(after.keys()))
    acc_keys = [k for k in keys if "acc" in k]
    other_keys = [k for k in keys if k not in acc_keys]
    print("\n=== Evaluation comparison (evaluation/eveluate_model.py) ===")
    for k in acc_keys + other_keys:
        b = before.get(k)
        a = after.get(k)
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            delta = a - b
            print(f"  {k}: before={b:.6f}  after={a:.6f}  delta={delta:+.6f}")
        else:
            print(f"  {k}: before={b!r}  after={a!r}")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    forget_roles = set(args.forget_roles)

    results_before: Dict[str, Any] | None = None
    results_after: Dict[str, Any] | None = None

    if not args.skip_eval:
        print("\n=== Baseline eval (original model, before ablation) ===")
        results_before = run_standalone_eval(
            args.model_path,
            device=args.eval_device,
            batch_size=args.eval_batch_size,
            max_length=args.eval_max_length,
            cache_dir=args.eval_cache_dir,
            dataset_cache_dir=args.eval_dataset_cache_dir,
            eng_valid_file=args.eval_eng_valid_file,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ablation_device = resolve_device(args.device)
    print(f"Ablation model load device (after resolve): {ablation_device}")
    local = load_local_model(args.model_path, device=ablation_device)
    model = local.model
    base = getattr(model, "model", model)

    meta: Dict[str, object] = {
        "model_path": args.model_path,
        "results_dir": str(results_dir),
        "forget_roles": sorted(forget_roles),
        "ridge_lambda": args.ridge_lambda,
        "layers": [],
    }

    for _layer_num, layer_dir in sorted_numeric_layer_dirs(results_dir):
        Z = _forget_feature_matrix(layer_dir, forget_roles)
        if Z is None:
            continue

        layer_idx = int(layer_dir.name.split("_")[-1])
        d_mlp = local.d_mlp
        if Z.shape[0] != d_mlp:
            raise ValueError(
                f"Layer {layer_idx}: F rows {Z.shape[0]} != model d_mlp {d_mlp}. "
                "Train SNMF with the same architecture / mlp_intermediate as this model."
            )

        down = base.layers[layer_idx].mlp.down_proj
        w = down.weight.data  # (d_model, d_mlp)
        dtype = w.dtype
        dev = w.device

        # Projector on CPU (float64): avoids CUDA on GPUs where PyTorch has no kernels (e.g. sm_61).
        z_cpu = Z.to(device="cpu", dtype=torch.float64)
        p_perp_cpu = orthogonal_projector_complement(z_cpu, ridge_lambda=args.ridge_lambda)
        p_perp = p_perp_cpu.to(device=dev, dtype=dtype)

        # W_V^{new} = W_V @ P_perp
        with torch.no_grad():
            w.copy_(torch.mm(w, p_perp))

        meta["layers"].append(
            {
                "layer": layer_idx,
                "n_forget_columns": int(Z.shape[1]),
                "d_mlp": int(d_mlp),
            }
        )
        print(
            f"Layer {layer_idx}: removed span of {Z.shape[1]} forget SNMF column(s) "
            f"via W_V @ P_perp."
        )

    if not meta["layers"]:
        raise RuntimeError(
            f"No forget features found under {results_dir} for roles={sorted(forget_roles)}."
        )

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    local.tokenizer.save_pretrained(save_path)
    print(f"Saved edited model and tokenizer to {save_path}")

    if args.metadata_out:
        out = Path(args.metadata_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote metadata to {out}")

    # Drop ablation model from memory before loading again for eval.
    del local, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not args.skip_eval:
        print("\n=== Post-ablation eval (saved checkpoint) ===")
        results_after = run_standalone_eval(
            str(save_path),
            device=args.eval_device,
            batch_size=args.eval_batch_size,
            max_length=args.eval_max_length,
            cache_dir=args.eval_cache_dir,
            dataset_cache_dir=args.eval_dataset_cache_dir,
            eng_valid_file=args.eval_eng_valid_file,
        )
        assert results_before is not None
        _print_eval_comparison(results_before, results_after)

        eval_out = (
            Path(args.eval_json_out)
            if args.eval_json_out
            else save_path / "ablation_eval_comparison.json"
        )
        eval_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline_model_path": args.model_path,
            "ablated_model_path": str(save_path),
            "before": _summarize_eval(results_before),
            "after": _summarize_eval(results_after),
        }
        with open(eval_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote eval comparison JSON to {eval_out}")


if __name__ == "__main__":
    main()
