"""
Supervised SNMF profiling for WMDP-bio style data (e.g. data/bio_data.json).

Unlike arithmetic runs (mult_concept / div_concept / neutral), bio_data uses only
``bio_forget`` (remove split) vs ``neutral`` (retain). Roles are binary:
forget-lean vs retain-lean vs weak / low-signal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from supervised_analysis import (
    _LOG_RATIO_EPS,
    _log_ratio,
    _marked_context_text,
    _sample_id_to_spans,
)
from utils import sorted_numeric_layer_dirs

# Labels from data/bio_data.json (see data_utils/create_bio_data.py)
BIO_FORGET_LABEL = "bio_forget"
RETAIN_LABEL = "neutral"

FORGET_LABELS = frozenset({BIO_FORGET_LABEL})
RETAIN_LABELS = frozenset({RETAIN_LABEL})


def _assign_role_label_bio(
    log_forget_vs_retain: float,
    mean_forget: float,
    mean_retain: float,
    n_forget: int,
    n_retain: int,
    min_log_ratio: float,
) -> str:
    if n_forget == 0 or n_retain == 0:
        return "insufficient_groups"
    total = mean_forget + mean_retain
    if total < 1e-9:
        return "low_signal"
    if log_forget_vs_retain >= min_log_ratio:
        return "bio_forget_lean"
    if log_forget_vs_retain <= -min_log_ratio:
        return "retain_lean"
    return "weak_mixed"


ROLE_LABEL_MEANINGS: Dict[str, str] = {
    "bio_forget_lean": (
        "Mean peak activation on bio_forget prompts is higher than on neutral (retain) "
        "by a clear log-ratio margin; latent aligns with the forget / remove split."
    ),
    "retain_lean": (
        "Mean peak activation on neutral (retain) prompts is higher than on bio_forget; "
        "latent aligns more with retained content."
    ),
    "weak_mixed": (
        "Log-ratio between forget and retain groups is small; no clear directional assignment."
    ),
    "low_signal": (
        "Negligible combined activation across forget and retain groups."
    ),
    "insufficient_groups": (
        "Missing prompts in bio_forget or neutral; cannot form a two-way comparison."
    ),
}

ROLE_LABEL_ORDER: Tuple[str, ...] = (
    "bio_forget_lean",
    "retain_lean",
    "weak_mixed",
    "low_signal",
    "insufficient_groups",
)


def plot_layer_wmdp_bio_trends(results_dir: str) -> None:
    """Aggregate supervised JSON and plot forget-vs-retain log-ratio by layer."""
    results_path = Path(results_dir)
    all_data: List[Dict[str, Any]] = []

    for layer_idx, layer_folder in sorted_numeric_layer_dirs(results_path):
        json_file = layer_folder / "feature_analysis_supervised_wmdp_bio.json"
        if not json_file.exists():
            continue
        with open(json_file, "r", encoding="utf-8") as f:
            layer_results = json.load(f)

        for latent_idx, profile in layer_results.items():
            lr = profile.get("log_ratios", {})
            all_data.append(
                {
                    "layer": layer_idx,
                    "latent_idx": int(latent_idx),
                    "role": profile.get("role_label", "unknown"),
                    "log_forget_vs_retain": lr.get("log_forget_vs_retain", np.nan),
                    "mean_act": profile.get("activation_stats", {}).get("mean", np.nan),
                }
            )

    df = pd.DataFrame(all_data)
    if df.empty:
        print("No feature_analysis_supervised_wmdp_bio.json data found for plotting.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.set_style("whitegrid")
    sns.lineplot(
        data=df,
        x="layer",
        y="log_forget_vs_retain",
        hue="role",
        ax=ax,
        marker="o",
        err_style="band",
        errorbar="sd",
        legend="brief",
    )
    ax.set_title(
        "log(mean forget / mean retain) by layer and role_label (WMDP-bio)",
        fontsize=14,
    )
    ax.set_ylabel("log ratio")
    ax.axhline(0, ls="--", color="black", alpha=0.4)
    plt.tight_layout()
    plot_path = results_path / "layer_wmdp_bio_trends.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved WMDP-bio trend plot to {plot_path}")
    plt.close(fig)

    print("\n--- Summary by role_label (WMDP-bio) ---")
    summary = (
        df.groupby("role")
        .agg(
            {
                "log_forget_vs_retain": ["mean", "std"],
                "mean_act": ["mean", "std"],
                "latent_idx": "count",
            }
        )
        .round(3)
    )
    print(summary)


def analyze_features_supervised_wmdp_bio(
    feature_acts: torch.Tensor,
    labels: List[str],
    sample_ids: List[int],
    token_ids: List[int],
    tokenizer,
    context_top_n: int = 10,
    context_window: int = 15,
    forget_labels: frozenset = FORGET_LABELS,
    retain_labels: frozenset = RETAIN_LABELS,
    role_assignment_threshold: float = 0.15,
) -> Dict[int, Dict[str, Any]]:
    """
    Per-latent prompt-mean peak activations for bio_forget vs neutral only.

    Primary log-ratio: log(mean_forget / mean_retain). Same peak-token aggregation as
    the arithmetic pipeline; class-count bias removed via per-group means.
    """
    print("Profiling latents (WMDP-bio supervised: bio_forget vs neutral)...")

    n_tokens, n_latents = feature_acts.shape
    sample_ids_arr = np.asarray(sample_ids)
    all_token_ids = np.asarray(token_ids, dtype=np.int64)
    labels_arr = np.asarray(labels)
    spans = _sample_id_to_spans(sample_ids_arr)

    feature_acts_np = feature_acts.detach().cpu().numpy().astype(np.float64, copy=False)
    frob_sq = float(np.sum(feature_acts_np**2)) + _LOG_RATIO_EPS

    try:
        _, _, vt = np.linalg.svd(feature_acts_np, full_matrices=False)
        svd_row0 = vt[0, :].astype(np.float64)
    except np.linalg.LinAlgError:
        svd_row0 = np.full(n_latents, np.nan, dtype=np.float64)

    sample_ids_list = list(spans.keys())
    n_prompts = len(sample_ids_list)

    sample_labels_arr = labels_arr[np.asarray(sample_ids_list, dtype=np.int64)]
    supervised_mask = np.isin(sample_labels_arr, list(forget_labels | retain_labels))
    sample_labels_sup = sample_labels_arr.copy()
    sample_labels_sup[~supervised_mask] = ""

    is_retain = np.isin(sample_labels_sup, list(retain_labels))
    is_forget = np.isin(sample_labels_sup, list(forget_labels))
    n_retain = int(np.sum(is_retain))
    n_forget = int(np.sum(is_forget))

    prompt_max_vals = np.empty((n_prompts, n_latents), dtype=np.float64)
    prompt_max_indices = np.empty((n_prompts, n_latents), dtype=np.int64)
    prompt_min_vals = np.empty((n_prompts, n_latents), dtype=np.float64)
    prompt_min_indices = np.empty((n_prompts, n_latents), dtype=np.int64)

    ar = np.arange(n_latents, dtype=np.int64)
    for i, sid in enumerate(sample_ids_list):
        samp_start, samp_end = spans[sid]
        seg = feature_acts_np[samp_start:samp_end, :]

        local_argmax = np.argmax(seg, axis=0)
        prompt_max_indices[i, :] = samp_start + local_argmax
        prompt_max_vals[i, :] = seg[local_argmax, ar]

        local_argmin = np.argmin(seg, axis=0)
        prompt_min_indices[i, :] = samp_start + local_argmin
        prompt_min_vals[i, :] = seg[local_argmin, ar]

    feature_profiles: Dict[int, Dict[str, Any]] = {}

    for latent_idx in range(n_latents):
        col = feature_acts_np[:, latent_idx]
        col_max = prompt_max_vals[:, latent_idx]
        sum_retain = float(np.sum(col_max[is_retain]))
        sum_forget = float(np.sum(col_max[is_forget]))
        mean_retain = sum_retain / n_retain if n_retain > 0 else 0.0
        mean_forget = sum_forget / n_forget if n_forget > 0 else 0.0

        log_forget_vs_retain = _log_ratio(mean_forget, mean_retain)

        role_label = _assign_role_label_bio(
            log_forget_vs_retain,
            mean_forget,
            mean_retain,
            n_forget,
            n_retain,
            role_assignment_threshold,
        )

        col_sq = float(np.sum(col**2))
        profile: Dict[str, Any] = {
            "role_label": role_label,
            "group_sums": {
                "neutral": round(sum_retain, 6),
                "bio_forget": round(sum_forget, 6),
            },
            "group_counts": {
                "neutral": n_retain,
                "bio_forget": n_forget,
            },
            "group_means": {
                "neutral": round(mean_retain, 6),
                "bio_forget": round(mean_forget, 6),
            },
            "log_ratios": {
                "log_forget_vs_retain": round(log_forget_vs_retain, 6),
            },
            "activation_stats": {
                "mean": round(float(np.mean(col_max)), 6),
                "max": round(float(np.max(col_max)), 6),
                "std": round(float(np.std(col_max)), 6),
                "sum_abs": round(float(np.sum(np.abs(col_max))), 6),
            },
            "column_frobenius_fraction": round(col_sq / frob_sq, 8),
            "svd_top_right_loading": round(float(svd_row0[latent_idx]), 8)
            if np.isfinite(svd_row0[latent_idx])
            else None,
        }

        kctx = min(context_top_n, n_prompts)
        if kctx > 0:
            col_min = prompt_min_vals[:, latent_idx]
            pos_sample_idx = np.argpartition(col_max, -kctx)[-kctx:]
            pos_sample_idx = pos_sample_idx[np.argsort(col_max[pos_sample_idx])[::-1]]
            pos_global_idx = prompt_max_indices[pos_sample_idx, latent_idx]

            neg_sample_idx = np.argpartition(col_min, kctx - 1)[:kctx]
            neg_sample_idx = neg_sample_idx[np.argsort(col_min[neg_sample_idx])]
            neg_global_idx = prompt_min_indices[neg_sample_idx, latent_idx]
            profile["top_positive_activation_contexts"] = [
                _marked_context_text(
                    tokenizer,
                    all_token_ids,
                    sample_ids_arr,
                    spans,
                    int(gi),
                    context_window,
                )
                for gi in pos_global_idx
            ]
            profile["top_negative_activation_contexts"] = [
                _marked_context_text(
                    tokenizer,
                    all_token_ids,
                    sample_ids_arr,
                    spans,
                    int(gi),
                    context_window,
                )
                for gi in neg_global_idx
            ]

        feature_profiles[latent_idx] = profile

    role_map: Dict[str, List[int]] = {}
    for idx, p in feature_profiles.items():
        role_map.setdefault(p["role_label"], []).append(idx)

    print("\nLatent summary by role_label (WMDP-bio; top 5 per role by log_forget_vs_retain):")
    for role in sorted(role_map.keys()):
        indices = role_map[role]
        indices.sort(
            key=lambda x: feature_profiles[x]["log_ratios"]["log_forget_vs_retain"],
            reverse=True,
        )
        top_str = ", ".join(
            f"{i}(log_fr:{feature_profiles[i]['log_ratios']['log_forget_vs_retain']:.2f})"
            for i in indices[:5]
        )
        print(f"  {role:22} | n={len(indices):4} | {top_str}")

    return feature_profiles
