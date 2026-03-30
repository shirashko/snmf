import os
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

MULT_LABEL = "mult_concept"
DIV_LABEL = "div_concept"
RETAIN_LABEL = "neutral"
FORGET_LABEL = {MULT_LABEL, DIV_LABEL}

RETAIN_LABELS = {RETAIN_LABEL}
MULT_LABELS = {MULT_LABEL}
DIV_LABELS = {DIV_LABEL}

_LOG_RATIO_EPS = 1e-12


def _log_ratio(num: float, den: float) -> float:
    return float(np.log((num + _LOG_RATIO_EPS) / (den + _LOG_RATIO_EPS)))


def _sample_id_to_spans(sample_ids_arr: np.ndarray) -> Dict[int, Tuple[int, int]]:
    """Map each sample_id to contiguous [start, end) slice in flat token arrays."""
    n = len(sample_ids_arr)
    if n == 0:
        return {}
    boundaries = np.r_[0, np.flatnonzero(sample_ids_arr[1:] != sample_ids_arr[:-1]) + 1, n]
    spans: Dict[int, Tuple[int, int]] = {}
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        spans[int(sample_ids_arr[a])] = (int(a), int(b))
    return spans


def _token_piece(tokenizer, tid: int) -> str:
    toks = tokenizer.convert_ids_to_tokens([int(tid)])
    return toks[0] if toks else str(tid)


def _sp_piece_to_space(text: str) -> str:
    """SentencePiece / Gemma use U+2581 (▁) at subword starts; swap for ASCII space for readable JSON."""
    return text.replace("\u2581", " ")


def _marked_context_text(
    tokenizer,
    all_token_ids: np.ndarray,
    sample_ids_arr: np.ndarray,
    spans: Dict[int, Tuple[int, int]],
    global_idx: int,
    context_window: int,
) -> str:
    """
    Local window (``context_window`` tokens each side, same sample only); pieces from
    ``convert_ids_to_tokens`` joined like the tutorial; peak token wrapped in ``**...**``.
    """
    sid = int(sample_ids_arr[global_idx])
    samp_start, samp_end = spans[sid]
    win_lo = max(samp_start, int(global_idx) - context_window)
    win_hi = min(samp_end, int(global_idx) + context_window + 1)
    parts: List[str] = []
    for j in range(win_lo, win_hi):
        if int(sample_ids_arr[j]) != sid:
            continue
        piece = _token_piece(tokenizer, int(all_token_ids[j]))
        parts.append("**" + piece + "**" if j == global_idx else piece)
    return _sp_piece_to_space("".join(parts))


def _assign_role_label(
    log_mult_vs_neutral_div: float,
    log_div_vs_neutral_mult: float,
    log_forget_vs_neutral: float,
    sum_mult: float,
    sum_div: float,
    sum_neutral: float,
    log_tau: float = 0.15,
) -> str:
    """Heuristic role from group-sum log-ratios (not top-k dominance)."""
    total = sum_mult + sum_div + sum_neutral
    if total < 1e-9:
        return "low_signal"

    if log_forget_vs_neutral >= log_tau:
        if log_mult_vs_neutral_div >= log_tau and log_mult_vs_neutral_div >= log_div_vs_neutral_mult:
            return "mult_forget"
        if log_div_vs_neutral_mult >= log_tau and log_div_vs_neutral_mult > log_mult_vs_neutral_div:
            return "div_forget"
        return "forget_mixed"

    if log_mult_vs_neutral_div >= log_tau and log_mult_vs_neutral_div >= log_div_vs_neutral_mult:
        return "mult_lean"
    if log_div_vs_neutral_mult >= log_tau:
        return "div_lean"

    if log_forget_vs_neutral <= -log_tau:
        return "neutral_lean"

    return "weak_mixed"


def plot_layer_concept_trends(results_dir: str):
    """
    Aggregates supervised analysis from all layers and plots trends (group-sum log-ratios).
    """
    results_path = Path(results_dir)
    all_data = []

    for layer_folder in sorted(results_path.glob("layer_*"), key=lambda x: int(x.name.split('_')[1])):
        layer_idx = int(layer_folder.name.split('_')[1])
        json_file = layer_folder / "feature_analysis_supervised.json"

        if not json_file.exists():
            continue

        with open(json_file, 'r') as f:
            layer_results = json.load(f)

        for latent_idx, profile in layer_results.items():
            lr = profile.get("log_ratios", {})
            all_data.append({
                'layer': layer_idx,
                'latent_idx': int(latent_idx),
                'role': profile.get('role_label', 'unknown'),
                'log_mult_nd': lr.get('log_mult_vs_neutral_div', np.nan),
                'log_div_nm': lr.get('log_div_vs_neutral_mult', np.nan),
                'log_fd_n': lr.get('log_forget_vs_neutral', np.nan),
                'mean_act': profile.get('activation_stats', {}).get('mean', np.nan),
            })

    df = pd.DataFrame(all_data)
    if df.empty:
        print("No feature_analysis_supervised.json data found for plotting.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    sns.set_style("whitegrid")

    sns.lineplot(
        data=df, x='layer', y='log_mult_nd', hue='role',
        ax=axes[0], marker='o', err_style="band", errorbar='sd', legend='brief'
    )
    axes[0].set_title("log(mult / (neutral + div)) by layer and role_label", fontsize=14)
    axes[0].set_ylabel("log ratio")
    axes[0].axhline(0, ls='--', color='black', alpha=0.4)

    sns.lineplot(
        data=df, x='layer', y='log_div_nm', hue='role',
        ax=axes[1], marker='o', err_style="band", errorbar='sd', legend='brief'
    )
    axes[1].set_title("log(div / (neutral + mult)) by layer and role_label", fontsize=14)
    axes[1].set_ylabel("log ratio")
    axes[1].axhline(0, ls='--', color='black', alpha=0.4)

    sns.lineplot(
        data=df, x='layer', y='log_fd_n', hue='role',
        ax=axes[2], marker='s', err_style="band", errorbar='sd', legend='brief'
    )
    axes[2].set_title("log((mult + div) / neutral) by layer and role_label", fontsize=14)
    axes[2].set_ylabel("log ratio")
    axes[2].axhline(0, ls='--', color='black', alpha=0.4)

    plt.tight_layout()
    plot_path = results_path / "layer_concept_trends.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved trend plot to {plot_path}")
    plt.close(fig)

    print("\n--- Summary by role_label ---")
    summary = df.groupby('role').agg({
        'log_mult_nd': ['mean', 'std'],
        'log_div_nm': ['mean', 'std'],
        'log_fd_n': ['mean', 'std'],
        'mean_act': ['mean', 'std'],
        'latent_idx': 'count',
    }).round(3)
    print(summary)


def analyze_features_supervised(
        feature_acts: torch.Tensor,
        labels: List[str],
        sample_ids: List[int],
        token_ids: List[int],
        tokenizer,
        context_top_n: int = 10,
        context_window: int = 15,
        forget_labels: set = FORGET_LABEL,
        retain_labels: set = RETAIN_LABELS,
        mult_labels: set = MULT_LABELS,
        div_labels: set = DIV_LABELS,
        role_assignment_threshold: float = 0.15,
) -> Dict[int, Dict[str, Any]]:
    """
    For each latent, compute group-sum log-ratios (neutral / mult / div).

    Each prompt contributes only its single maximum-activation token (per latent) to
    the group sums, removing prompt-length bias in the log-ratio computations.

    - log(mult / (neutral + div))
    - log(div / (neutral + mult))
    - log((mult + div) / neutral)

    Also records activation stats, column energy share, and first right-singular-vector loading
    (one SVD of G per call). For the top ``context_top_n`` max- and min-activation tokens,
    logs lists of local window strings only (``context_window`` tokens each side, same sample),
    peak marked with ``**...**``.
    """
    print("Profiling latents (supervised, group-sum log-ratios)...")

    n_tokens, n_latents = feature_acts.shape
    sample_ids_arr = np.asarray(sample_ids)
    all_token_ids = np.asarray(token_ids, dtype=np.int64)
    labels_arr = np.asarray(labels)
    token_labels = labels_arr[sample_ids_arr]
    spans = _sample_id_to_spans(sample_ids_arr)

    feature_acts_np = feature_acts.detach().cpu().numpy().astype(np.float64, copy=False)
    frob_sq = float(np.sum(feature_acts_np ** 2)) + _LOG_RATIO_EPS

    # Right singular vectors: how each latent loads on the leading variance axis of G
    try:
        _, _, vt = np.linalg.svd(feature_acts_np, full_matrices=False)
        svd_row0 = vt[0, :].astype(np.float64)
    except np.linalg.LinAlgError:
        svd_row0 = np.full(n_latents, np.nan, dtype=np.float64)

    sample_ids_list = list(spans.keys())
    n_prompts = len(sample_ids_list)

    # Prompt-level labels for supervised group membership.
    sample_labels_arr = labels_arr[np.asarray(sample_ids_list, dtype=np.int64)]
    supervised_mask_samples = np.isin(sample_labels_arr, list(forget_labels | retain_labels))
    sample_labels_sup = sample_labels_arr.copy()
    sample_labels_sup[~supervised_mask_samples] = ""

    is_neutral = np.isin(sample_labels_sup, list(retain_labels))
    is_mult = np.isin(sample_labels_sup, list(mult_labels))
    is_div = np.isin(sample_labels_sup, list(div_labels))

    # Reduce each prompt to one peak token per latent:
    # max for positive context/log-sums, and min for negative context display.
    prompt_max_vals = np.empty((n_prompts, n_latents), dtype=np.float64)
    prompt_max_indices = np.empty((n_prompts, n_latents), dtype=np.int64)
    prompt_min_vals = np.empty((n_prompts, n_latents), dtype=np.float64)
    prompt_min_indices = np.empty((n_prompts, n_latents), dtype=np.int64)

    ar = np.arange(n_latents, dtype=np.int64)
    for i, sid in enumerate(sample_ids_list):
        samp_start, samp_end = spans[sid]
        seg = feature_acts_np[samp_start:samp_end, :]  # (seq_len_in_prompt, n_latents)

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
        sum_neutral = float(np.sum(col_max[is_neutral]))
        sum_mult = float(np.sum(col_max[is_mult]))
        sum_div = float(np.sum(col_max[is_div]))
        sum_forget = sum_mult + sum_div

        others_for_mult = sum_neutral + sum_div
        others_for_div = sum_neutral + sum_mult

        log_mult_vs_neutral_div = _log_ratio(sum_mult, others_for_mult)
        log_div_vs_neutral_mult = _log_ratio(sum_div, others_for_div)
        log_forget_vs_neutral = _log_ratio(sum_forget, sum_neutral)

        role_label = _assign_role_label(
            log_mult_vs_neutral_div,
            log_div_vs_neutral_mult,
            log_forget_vs_neutral,
            sum_mult,
            sum_div,
            sum_neutral,
            log_tau=role_assignment_threshold,
        )

        col_sq = float(np.sum(col ** 2))
        profile: Dict[str, Any] = {
            "role_label": role_label,
            "group_sums": {
                "neutral": round(sum_neutral, 6),
                "mult": round(sum_mult, 6),
                "div": round(sum_div, 6),
                "forget": round(sum_forget, 6),
            },
            "log_ratios": {
                "log_mult_vs_neutral_div": round(log_mult_vs_neutral_div, 6),
                "log_div_vs_neutral_mult": round(log_div_vs_neutral_mult, 6),
                "log_forget_vs_neutral": round(log_forget_vs_neutral, 6),
            },
            "activation_stats": {
                "mean": round(float(np.mean(col)), 6),
                "max": round(float(np.max(col)), 6),
                "std": round(float(np.std(col)), 6),
                "sum_abs": round(float(np.sum(np.abs(col))), 6),
            },
            "column_frobenius_fraction": round(col_sq / frob_sq, 8),
            "svd_top_right_loading": round(float(svd_row0[latent_idx]), 8)
            if np.isfinite(svd_row0[latent_idx])
            else None,
        }

        kctx = min(context_top_n, n_prompts)

        if kctx > 0:
            # Pick top-k prompts by max activation, and top-k prompts by min activation.
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

    print("\nLatent summary by role_label (top 5 per role by log_forget_vs_neutral):")
    for role in sorted(role_map.keys()):
        indices = role_map[role]
        indices.sort(
            key=lambda x: feature_profiles[x]["log_ratios"]["log_forget_vs_neutral"],
            reverse=True,
        )
        top_str = ", ".join(
            f"{i}(log_fd_n:{feature_profiles[i]['log_ratios']['log_forget_vs_neutral']:.2f})"
            for i in indices[:5]
        )
        print(f"  {role:18} | n={len(indices):4} | {top_str}")

    return feature_profiles