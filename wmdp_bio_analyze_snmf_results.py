"""
Analyze SNMF checkpoints trained on WMDP-bio supervision (e.g. data/bio_data.json:
bio_forget vs retain buckets). Unary per-latent log-ratios.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from llm_utils.model_utils import load_local_model
from llm_utils.utils import resolve_device, set_seed, sorted_numeric_layer_dirs
from wmdp_bio_supervised_analysis import (
    RETAIN_BASIS_CHOICES,
    RETAIN_BASIS_POOLED,
    ROLE_LABEL_MEANINGS,
    ROLE_LABEL_ORDER,
    analyze_features_supervised_wmdp_bio,
    plot_layer_wmdp_bio_trends,
)
from unsupervised_analysis import analyze_features_unsupervised

ANALYSIS_OVERVIEW = (
    "Each SNMF column is one latent. Unary supervised JSON stores per-latent log-ratios: "
    "bio_forget vs pooled retain (neutral ∪ bio_retain), vs neutral-only, vs bio_retain-only, "
    "and log(mean_bio_retain/mean_neutral) when both retain buckets exist. "
    "role_label uses one chosen basis (--supervised-retain-basis: pooled | neutral | bio_retain) "
    "with the same threshold on log(mean_forget/mean_retain_side). "
    "Counts are how many latents received each unary role label, per layer and in total."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SNMF results (WMDP-bio / bio_data.json).")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Folder with layer_X/snmf_factors.pt from train_snmf.py",
    )
    parser.add_argument(
        "--role-assignment-threshold",
        type=float,
        default=0.15,
        metavar="LOG_RATIO",
        help="Minimum |log(mean_forget/mean_retain_side)| margin for bio_forget_lean / retain_lean.",
    )
    parser.add_argument(
        "--supervised-retain-basis",
        type=str,
        default=RETAIN_BASIS_POOLED,
        choices=sorted(RETAIN_BASIS_CHOICES),
        help=(
            "Which retain pool sets role_label: pooled (neutral+bio_retain), neutral only, "
            "or bio_retain only. All three forget-vs-side log-ratios are still written per latent."
        ),
    )
    parser.add_argument("--skip-vocab", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-unsupervised", type=int, default=30)
    parser.add_argument(
        "--activation-context-top-n",
        type=int,
        default=10,
        help="Per latent: max/min activation contexts to log in supervised JSON.",
    )
    parser.add_argument(
        "--activation-context-window",
        type=int,
        default=15,
        help="Tokens before/after peak token in each context (same sample only).",
    )
    parser.add_argument(
        "--summary-filename",
        type=str,
        default="analysis_summary_wmdp_bio.json",
        help="Written under --results-dir (global role counts and meanings).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    set_seed(args.seed)
    device = resolve_device(args.device)

    print(f"Loading model from {args.model_path}...")
    local_model = load_local_model(args.model_path, device=device)

    per_layer_stats: list[dict] = []
    global_role_counts: Counter[str] = Counter()

    for layer_num, layer_folder in sorted_numeric_layer_dirs(results_dir):
        factors_path = layer_folder / "snmf_factors.pt"
        if not factors_path.exists():
            print(f"Skipping layer {layer_num} because snmf_factors.pt is missing.")
            continue

        supervised_path = layer_folder / "feature_analysis_supervised_wmdp_bio.json"
        print(f"\nProcessing {layer_folder.name}...")

        checkpoint = torch.load(factors_path, map_location="cpu", weights_only=False)
        F, G = checkpoint["F"], checkpoint["G"]
        token_ids, sample_ids = checkpoint["token_ids"], checkpoint["sample_ids"]
        labels, mode = checkpoint["labels"], checkpoint.get("mode", "mlp_intermediate")

        supervised_results = analyze_features_supervised_wmdp_bio(
            G,
            labels,
            sample_ids,
            token_ids,
            local_model.tokenizer,
            role_assignment_threshold=args.role_assignment_threshold,
            retain_basis=args.supervised_retain_basis,
            context_top_n=args.activation_context_top_n,
            context_window=args.activation_context_window,
        )

        with open(supervised_path, "w", encoding="utf-8") as f:
            json.dump(supervised_results, f, indent=2, ensure_ascii=False)

        n_features = len(supervised_results)
        layer_roles = Counter(
            supervised_results[k].get("role_label", "unknown") for k in supervised_results
        )
        global_role_counts.update(layer_roles)
        layer_entry: dict = {
            "layer": layer_num,
            "features_explored": n_features,
            "counts_by_role": dict(layer_roles),
        }
        per_layer_stats.append(layer_entry)
        print(
            f"  Layer {layer_num}: {n_features} latents | roles: "
            + ", ".join(f"{r}={c}" for r, c in sorted(layer_roles.items()))
        )

        if not args.skip_vocab:
            unsupervised_results = analyze_features_unsupervised(
                F=F,
                local_model=local_model,
                layer=layer_num,
                mode=mode,
                top_k_tokens=args.top_k_unsupervised,
            )
            out_unsup = layer_folder / "feature_analysis_unsupervised_wmdp_bio.json"
            with open(out_unsup, "w", encoding="utf-8") as f:
                json.dump(unsupervised_results, f, indent=2, ensure_ascii=False)

    print("\nGenerating WMDP-bio trend plots (one PNG per retain basis)...")
    for basis in sorted(RETAIN_BASIS_CHOICES):
        try:
            plot_layer_wmdp_bio_trends(str(results_dir), retain_basis=basis)
        except Exception as e:
            print(f"Could not generate plot for basis={basis}: {e}")

    total_features = sum(s["features_explored"] for s in per_layer_stats)
    summary_path = results_dir / args.summary_filename
    ordered_global: dict[str, int] = {r: global_role_counts.get(r, 0) for r in ROLE_LABEL_ORDER}
    for label, c in global_role_counts.items():
        if label not in ordered_global:
            ordered_global[label] = c

    summary_doc = {
        "overview": ANALYSIS_OVERVIEW,
        "pipeline": "wmdp_bio",
        "supervised_retain_basis": args.supervised_retain_basis,
        "retain_basis_note": (
            f"role_label used log(mean_forget/mean_retain) with retain side "
            f"{args.supervised_retain_basis!r}. Per-latent JSON also includes "
            "log_forget_vs_pooled_retain, log_forget_vs_neutral, log_forget_vs_bio_retain, "
            "and log_bio_retain_vs_neutral when counts allow."
        ),
        "role_assignment_threshold": args.role_assignment_threshold,
        "threshold_note": (
            "Minimum natural-log ratio margin for bio_forget_lean vs retain_lean vs weak_mixed; "
            "see wmdp_bio_supervised_analysis._assign_role_label_bio."
        ),
        "total_features_explored": total_features,
        "layers_processed": len(per_layer_stats),
        "global_counts_by_role": ordered_global,
        "per_layer": per_layer_stats,
        "role_meanings": ROLE_LABEL_MEANINGS,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_doc, f, indent=2, ensure_ascii=False)

    print("\n--- Global role counts (WMDP-bio, all layers) ---")
    for r in ROLE_LABEL_ORDER:
        c = global_role_counts.get(r, 0)
        if c:
            print(f"  {r}: {c}")
    for r, c in sorted(global_role_counts.items()):
        if r not in ROLE_LABEL_ORDER:
            print(f"  {r}: {c}")

    print(f"\nTotal latents profiled: {total_features}")
    print(f"Wrote summary: {summary_path}")
    print(f"\nAnalysis complete. Outputs use *_wmdp_bio.json suffixes under {args.results_dir}")


if __name__ == "__main__":
    main()
