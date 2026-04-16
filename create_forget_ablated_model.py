"""
Build a new HF checkpoint whose MLP removes SNMF forget directions (dual-sided where applicable).

Expected pipeline (matches the repo shell scripts):

  1. ``scripts/wmdp/train_snmf.sh`` → writes ``outputs/snmf_train_results/layer_*/snmf_factors.pt``
     with ``mode=mlp_intermediate`` (required: ``F`` lives in the same space as
     ``mlp.down_proj`` input).
  2. ``scripts/arith/run_analyze_snmf_results.sh`` → writes ``layer_*/feature_analysis_supervised.json``
     (``role_label`` per latent, from ``analyze_snmf_results.py``).
  3. This script → reads ``F`` + ``role_label``, applies projections on MLP weights, saves a new model.

Use the **same** ``--model-path`` as training/analysis and the **same** ``--results-dir``
as ``--output-dir`` / ``RESULTS_DIR`` (default: ``outputs/snmf_train_results``).

Mathematical setup (matches the cited write-up):
  - Each SNMF column z_i ∈ ℝ^{d_mlp} is a direction in the post-activation (neuron) space.
  - **Output side (down_proj):** y = W_V x with x ∈ ℝ^{d_mlp}. Remove forget span from x before W_V:
        W_V^{new} = W_V @ P_perp.
  - **Input / gate side (up_proj, gate_proj in Gemma-2):** these map ℝ^{d_model} → ℝ^{d_mlp}
    with weight W of shape (d_mlp, d_model). Output lives in the same space as z, so remove the
    span from the *output* of these layers:
        W^{new} = P_perp @ W.

For multiple forget features {z_1,…,z_k}, use the projector onto their span:
    P_span = Z (Z^T Z + λ I)^{-1} Z^T,   Z = [z_1 | … | z_k],
    P_perp = I - s P_span   (s = ``--span-projection-scale``; on-span scaling is (1-s)—s=1 removes
    the span component; s>1 over-subtracts and can flip that component; see ``--span-projection-scale`` help).

This edits weights only (no runtime hooks). Layers without ``gate_proj`` / ``up_proj`` only get
``down_proj`` edits.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, Set

import torch

from evaluation.eveluate_model import run_standalone_eval
from llm_utils.model_utils import load_local_model
from llm_utils.utils import resolve_device, sorted_numeric_layer_dirs


def _load_role_map(layer_dir: Path, supervised_json_filename: str) -> Dict[int, str]:
    path = layer_dir / supervised_json_filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run analyze_snmf_results.py on --results-dir first "
            f"(e.g. scripts/arith/run_analyze_snmf_results.sh) so each layer has role_label entries."
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v.get("role_label", "unknown")) for k, v in raw.items()}


def _forget_feature_matrix(
    layer_dir: Path,
    forget_roles: Set[str],
    supervised_json_filename: str,
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
            f"--mode mlp_intermediate (see scripts/wmdp/train_snmf.sh)."
        )
    F = ckpt["F"].float().cpu()
    if F.ndim != 2:
        raise ValueError(f"Unexpected F shape in {ckpt_path}: {tuple(F.shape)}")

    roles = _load_role_map(layer_dir, supervised_json_filename)
    k_all = F.shape[1]
    forget_cols = sorted(i for i, r in roles.items() if r in forget_roles and 0 <= i < k_all)
    if not forget_cols:
        return None
    Z = F[:, forget_cols].contiguous()
    return Z


def orthogonal_projector_complement(
    Z: torch.Tensor,
    ridge_lambda: float,
    *,
    span_projection_scale: float = 1.0,
) -> torch.Tensor:
    """
    P_perp = I - s · Z (Z^T Z + λ I)^{-1} Z^T  ∈ ℝ^{d×d}, with Z ∈ ℝ^{d×k},
    where s = span_projection_scale (coefficient on P_span).

    For s=1 this is the usual orthogonal complement projector. For k=1 and s=1 this equals
    I - z z^T / (||z||^2 + λ) (≈ paper formula when λ=0).
    """
    d, k = Z.shape
    device, dtype = Z.device, Z.dtype
    I_d = torch.eye(d, device=device, dtype=dtype)
    if k == 0:
        return I_d
    g = Z.T @ Z + ridge_lambda * torch.eye(k, device=device, dtype=dtype)
    inv = torch.linalg.solve(g, torch.eye(k, device=device, dtype=dtype))
    p_span = Z @ inv @ Z.T
    return I_d - float(span_projection_scale) * p_span


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Remove SNMF forget directions: W_down <- W_down @ P_perp; "
        "W_up,W_gate <- P_perp @ W when present.",
        epilog=(
            "Typical use after scripts/wmdp/train_snmf.sh and scripts/arith/run_analyze_snmf_results.sh: same defaults "
            "as those scripts (model + outputs/snmf_train_results)."
        ),
    )
    p.add_argument(
        "--model-path",
        type=str,
        default="local_models/gemma-2-0.3B_reference_model",
        help="Same base model as train_snmf.py / analyze_snmf_results.py (default: scripts/wmdp/train_snmf.sh).",
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default="outputs/snmf_train_results",
        help="SNMF output dir with layer_*/ (default: scripts/wmdp/train_snmf.sh --output-dir).",
    )
    p.add_argument(
        "--save-path",
        type=str,
        default="local_models/gemma-2-0.3B_forget_ablated",
        help="Directory for save_pretrained (default: sibling of reference model).",
    )
    p.add_argument(
        "--save-path-random",
        type=str,
        default="",
        help="Optional directory for random-direction ablated model. "
        "Default: <save-path>_random_baseline.",
    )
    p.add_argument(
        "--forget-roles",
        type=str,
        nargs="+",
        default=["mult_forget", "div_forget", "forget_mixed"],
    )
    p.add_argument(
        "--supervised-json-filename",
        type=str,
        default="feature_analysis_supervised.json",
        help=(
            "Per-layer supervised analysis JSON file name inside each layer_* folder "
            "(e.g. feature_analysis_supervised_wmdp_bio.json)."
        ),
    )
    p.add_argument(
        "--ridge-lambda",
        type=float,
        default=1e-6,
        help="Tikhonov on Z^T Z when building the span projector (stability).",
    )
    p.add_argument(
        "--span-projection-scale",
        type=float,
        default=1.0,
        help=(
            "Coefficient s on P_span in P_perp = I - s·P_span. On vectors in the forget span, "
            "this acts as scaling by (1-s): s=1 removes that component entirely (orthogonal "
            "complement); s<1 leaves a residual (softer removal); s>1 over-subtracts—(1-s) is "
            "negative, so the span component is flipped and scaled in magnitude (e.g. s=2 gives "
            "the opposite direction with equal norm). Default 1.0."
        ),
    )
    p.add_argument(
        "--random-baseline",
        action="store_true",
        help="Also run matched-count random-direction ablation baseline.",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=1234,
        help="Seed for reproducible random-direction baseline.",
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
        "--eval-mode",
        type=str,
        default="arithmetic",
        choices=["arithmetic", "wmdp_bio", "wmdp_cyber", "both_wmdp", "wmdp_bio_categorized"],
        help="Evaluation mode passed to run_standalone_eval / eveluate_model.py.",
    )
    p.add_argument(
        "--eval-large",
        action="store_true",
        help="Use larger/full evaluation limits for WMDP/MMLU tasks.",
    )
    p.add_argument(
        "--eval-no-mmlu",
        action="store_true",
        help="For single-domain WMDP eval modes, skip MMLU.",
    )
    p.add_argument(
        "--eval-wmdp-include-path",
        type=str,
        default="",
        help="Path to lm-eval task YAML directory (used with eval-mode=wmdp_bio_categorized).",
    )
    p.add_argument(
        "--eval-wmdp-task-name",
        type=str,
        default="wmdp_bio_robust",
        help="Task/group name for eval-mode=wmdp_bio_categorized.",
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
        default="/home/morg/students/rashkovits/Localized-UNDO/datasets/pretrain/valid_eng.jsonl",
    )
    p.add_argument(
        "--eval-json-out",
        type=str,
        default="",
        help="Write before/after metrics JSON here (default: <save-path>/ablation_eval_comparison.json).",
    )
    p.add_argument(
        "--down-proj-only",
        action="store_true",
        help="Only ablate down_proj (W_V @ P_perp). Skip gate_proj/up_proj for ablations that "
        "match the old single-sided behavior.",
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


def _random_direction_matrix(
    d_mlp: int,
    n_dirs: int,
    *,
    seed: int,
    layer_idx: int,
) -> torch.Tensor:
    """
    Build a random orthonormal-basis subset Z in R^(d_mlp x n_dirs).
    """
    if n_dirs <= 0:
        return torch.empty((d_mlp, 0), dtype=torch.float32)
    if n_dirs > d_mlp:
        raise ValueError(f"Requested n_dirs={n_dirs} but d_mlp={d_mlp}.")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + int(layer_idx))
    rnd = torch.randn((d_mlp, n_dirs), generator=gen, dtype=torch.float64)
    q, _ = torch.linalg.qr(rnd, mode="reduced")
    return q.to(dtype=torch.float32, device="cpu")


def _apply_ablation_to_model(
    model_path: str,
    results_dir: Path,
    forget_roles: Set[str],
    supervised_json_filename: str,
    ridge_lambda: float,
    device: str,
    random_baseline: bool,
    random_seed: int,
    *,
    span_projection_scale: float = 1.0,
    down_proj_only: bool = False,
) -> tuple[object, Dict[str, object]]:
    """
    Load model, apply either learned-direction or random-direction ablation.
    Returns (local_model_wrapper, metadata).
    """
    local = load_local_model(model_path, device=device)
    model = local.model
    base = getattr(model, "model", model)
    d_mlp = local.d_mlp

    meta: Dict[str, object] = {
        "model_path": model_path,
        "results_dir": str(results_dir),
        "forget_roles": sorted(forget_roles),
        "supervised_json_filename": supervised_json_filename,
        "ridge_lambda": ridge_lambda,
        "span_projection_scale": float(span_projection_scale),
        "ablation_type": "random_matched_count" if random_baseline else "learned_forget_directions",
        "random_seed": int(random_seed) if random_baseline else None,
        "down_proj_only": bool(down_proj_only),
        "layers": [],
    }

    for _layer_num, layer_dir in sorted_numeric_layer_dirs(results_dir):
        Z_learned = _forget_feature_matrix(layer_dir, forget_roles, supervised_json_filename)
        if Z_learned is None:
            continue

        layer_idx = int(layer_dir.name.split("_")[-1])
        if Z_learned.shape[0] != d_mlp:
            raise ValueError(
                f"Layer {layer_idx}: F rows {Z_learned.shape[0]} != model d_mlp {d_mlp}. "
                "Train SNMF with the same architecture / mlp_intermediate as this model."
            )
        n_forget = int(Z_learned.shape[1])
        Z = (
            _random_direction_matrix(d_mlp, n_forget, seed=random_seed, layer_idx=layer_idx)
            if random_baseline
            else Z_learned
        )

        mlp = base.layers[layer_idx].mlp
        w_down = mlp.down_proj.weight.data  # (d_model, d_mlp)
        dtype = w_down.dtype
        dev = w_down.device

        # Projector on CPU (float64): avoids CUDA on GPUs where PyTorch has no kernels (e.g. sm_61).
        z_cpu = Z.to(device="cpu", dtype=torch.float64)
        p_perp_cpu = orthogonal_projector_complement(
            z_cpu,
            ridge_lambda=ridge_lambda,
            span_projection_scale=span_projection_scale,
        )
        p_perp = p_perp_cpu.to(device=dev, dtype=dtype)

        with torch.no_grad():
            # W_V^{new} = W_V @ P_perp  (remove forget subspace from down_proj *input*)
            w_down.copy_(torch.mm(w_down, p_perp))
            if not down_proj_only:
                for name in ("gate_proj", "up_proj"):
                    lin = getattr(mlp, name, None)
                    if lin is None:
                        continue
                    w_in = lin.weight.data  # (d_mlp, d_model)
                    if w_in.shape[0] != p_perp.shape[0]:
                        raise ValueError(
                            f"Layer {layer_idx}: {name}.weight.shape[0]={w_in.shape[0]} != "
                            f"d_mlp={p_perp.shape[0]}; cannot apply P_perp @ W."
                        )
                    # y' = P_perp @ (W @ x + b)  =>  W' = P_perp @ W, b' = P_perp @ b
                    w_in.copy_(torch.mm(p_perp, w_in))
                    if lin.bias is not None:
                        b = lin.bias.data
                        lin.bias.data.copy_(torch.mv(p_perp, b))

        layer_meta = {
            "layer": layer_idx,
            "n_forget_columns": n_forget,
            "d_mlp": int(d_mlp),
            "dual_sided": not down_proj_only,
        }
        if random_baseline:
            layer_meta["random_seed"] = int(random_seed) + int(layer_idx)
        meta["layers"].append(layer_meta)
        side_msg = "W_down @ P_perp only" if down_proj_only else "P_perp @ W_gate/up + W_down @ P_perp"
        if random_baseline:
            print(
                f"Layer {layer_idx}: removed span of {n_forget} RANDOM direction(s) "
                f"(matched to forget count) with span_projection_scale={span_projection_scale} "
                f"via {side_msg}."
            )
        else:
            print(
                f"Layer {layer_idx}: removed span of {n_forget} forget SNMF column(s) "
                f"with span_projection_scale={span_projection_scale} via {side_msg}."
            )

    if not meta["layers"]:
        raise RuntimeError(
            f"No forget features found under {results_dir} for roles={sorted(forget_roles)}."
        )
    return local, meta


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
            eval_mode=args.eval_mode,
            large_eval=args.eval_large,
            no_mmlu=args.eval_no_mmlu,
            wmdp_include_path=args.eval_wmdp_include_path,
            wmdp_task_name=args.eval_wmdp_task_name,
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
    local, meta = _apply_ablation_to_model(
        model_path=args.model_path,
        results_dir=results_dir,
        forget_roles=forget_roles,
        supervised_json_filename=args.supervised_json_filename,
        ridge_lambda=args.ridge_lambda,
        device=ablation_device,
        random_baseline=False,
        random_seed=args.random_seed,
        span_projection_scale=args.span_projection_scale,
        down_proj_only=args.down_proj_only,
    )

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    local.model.save_pretrained(save_path)
    local.tokenizer.save_pretrained(save_path)
    print(f"Saved edited model and tokenizer to {save_path}")

    if args.metadata_out:
        out = Path(args.metadata_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote metadata to {out}")

    # Drop ablation model from memory before loading again for eval.
    del local
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not args.skip_eval:
        print("\n=== Post-ablation eval (saved checkpoint) ===")
        results_after = run_standalone_eval(
            str(save_path),
            eval_mode=args.eval_mode,
            large_eval=args.eval_large,
            no_mmlu=args.eval_no_mmlu,
            wmdp_include_path=args.eval_wmdp_include_path,
            wmdp_task_name=args.eval_wmdp_task_name,
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

    if args.random_baseline:
        print("\n=== Random baseline ablation (matched direction count) ===")
        local_rand, rand_meta = _apply_ablation_to_model(
            model_path=args.model_path,
            results_dir=results_dir,
            forget_roles=forget_roles,
            supervised_json_filename=args.supervised_json_filename,
            ridge_lambda=args.ridge_lambda,
            device=ablation_device,
            random_baseline=True,
            random_seed=args.random_seed,
            span_projection_scale=args.span_projection_scale,
            down_proj_only=args.down_proj_only,
        )
        save_path_random = (
            Path(args.save_path_random)
            if args.save_path_random
            else Path(f"{args.save_path}_random_baseline")
        )
        save_path_random.mkdir(parents=True, exist_ok=True)
        local_rand.model.save_pretrained(save_path_random)
        local_rand.tokenizer.save_pretrained(save_path_random)
        print(f"Saved random-baseline model and tokenizer to {save_path_random}")

        del local_rand
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not args.skip_eval:
            print("\n=== Post-random-baseline eval (saved checkpoint) ===")
            results_random = run_standalone_eval(
                str(save_path_random),
                eval_mode=args.eval_mode,
                large_eval=args.eval_large,
                no_mmlu=args.eval_no_mmlu,
                wmdp_include_path=args.eval_wmdp_include_path,
                wmdp_task_name=args.eval_wmdp_task_name,
                device=args.eval_device,
                batch_size=args.eval_batch_size,
                max_length=args.eval_max_length,
                cache_dir=args.eval_cache_dir,
                dataset_cache_dir=args.eval_dataset_cache_dir,
                eng_valid_file=args.eval_eng_valid_file,
            )
            assert results_before is not None and results_after is not None
            print("\n--- Learned-direction ablation vs random baseline ---")
            _print_eval_comparison(results_after, results_random)
            print("\n--- Original baseline vs random baseline ---")
            _print_eval_comparison(results_before, results_random)

            payload["random_baseline"] = {
                "random_seed": int(args.random_seed),
                "ablated_model_path": str(save_path_random),
                "metadata": rand_meta,
                "after": _summarize_eval(results_random),
            }
            with open(eval_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Updated eval comparison JSON with random baseline at {eval_out}")


if __name__ == "__main__":
    main()
