#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=snmf_forget_wmdp_bio
#SBATCH --output=logs/snmf_forget_pipe_%j.out
#SBATCH --error=logs/snmf_forget_pipe_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G

# End-to-end: SNMF training -> supervised analysis -> forget ablation checkpoint (+ optional eval).
#
# Defaults: WMDP-bio profile for Gemma-2-2B, writing/reading
# outputs/snmf_train_results_wmdp_bio_gemma2_2b.
# Override anything via env before sbatch, e.g.:
#   SNMF_OUTPUT_DIR=outputs/my_run ABLATION_OUTPUT_DIR=outputs/my_run_ablate_meta sbatch scripts/wmdp/run_snmf_forget_pipeline.sh
#
# Skip steps (reuse existing artifacts):
#   SKIP_TRAIN=1    — skip train_snmf.py (expects layer_* under SNMF_OUTPUT_DIR)
#   SKIP_ANALYZE=1  — skip wmdp_bio_analyze_snmf_results.py

set -euo pipefail

# --- Environment Setup ---
source /home/morg/students/rashkovits/miniconda3/etc/profile.d/conda.sh
conda activate snmf_env

# --- Space & Cache Management ---
export HF_HOME="${HF_HOME:-/home/morg/students/rashkovits/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$HF_HOME/torch}"
export TMPDIR="${TMPDIR:-$HF_HOME}"

# --- Project Setup ---
REPO_ROOT="/home/morg/students/rashkovits/snmf"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

mkdir -p logs "$HF_HOME"

# ========== Shared I/O ==========
# Where train_snmf writes and analyze + forget ablation read SNMF artifacts.
# Default points to your WMDP-bio SNMF run.
SNMF_OUTPUT_DIR="${SNMF_OUTPUT_DIR:-outputs/snmf_train_results_wmdp_bio_gemma2_2b}"
# Forget-ablation metadata + eval JSON (separate from SNMF layer_* tree).
ABLATION_OUTPUT_DIR="${ABLATION_OUTPUT_DIR:-outputs/forget_ablation_wmdp_bio_gemma2_2b}"

# Base model used by WMDP-bio SNMF run.
MODEL_PATH="${MODEL_PATH:-/home/morg/students/rashkovits/Localized-UNDO/models/wmdp/gemma-2-2b}"
DATA_PATH="${DATA_PATH:-data/bio_data.json}"

# --- train_snmf.py (see scripts/wmdp/train_snmf.sh) ---
LAYERS="${LAYERS:-0-25}"
RANK="${RANK:-300}"
SNMF_MODE="${SNMF_MODE:-mlp_intermediate}"
SNMF_INIT="${SNMF_INIT:-svd}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"
SPARSITY="${SPARSITY:-0.01}"
MAX_ITER="${MAX_ITER:-5000}"
TRAIN_SEED="${TRAIN_SEED:-42}"

# --- wmdp_bio_analyze_snmf_results.py ---
SUMMARY_FILE="${SUMMARY_FILE:-analysis_summary_wmdp_bio.json}"
ANALYZE_DEVICE="${ANALYZE_DEVICE:-auto}"
ANALYZE_SEED="${ANALYZE_SEED:-42}"

# --- create_forget_ablated_model (see run_create_forget_ablated_model.sh) ---
# New checkpoint dir so MODEL_PATH is never overwritten by save_pretrained.
SAVE_PATH="${SAVE_PATH:-local_models/gemma-2-2b_wmdp_bio_forget_ablated}"
SAVE_PATH_RANDOM="${SAVE_PATH_RANDOM:-${SAVE_PATH}_random_baseline}"
# Per-layer supervised role JSON expected by create_forget_ablated_model.py.
SUPERVISED_JSON_FILENAME="${SUPERVISED_JSON_FILENAME:-feature_analysis_supervised_wmdp_bio.json}"
mkdir -p "$ABLATION_OUTPUT_DIR"
METADATA_OUT="${METADATA_OUT:-${ABLATION_OUTPUT_DIR}/forget_ablation_metadata.json}"
EVAL_JSON_OUT="${EVAL_JSON_OUT:-${ABLATION_OUTPUT_DIR}/ablation_eval_comparison.json}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1e-6}"
RANDOM_SEED="${RANDOM_SEED:-1234}"
ABLATION_DEVICE="${ABLATION_DEVICE:-auto}"
EVAL_DEVICE="${EVAL_DEVICE:-auto}"
EVAL_MODE="${EVAL_MODE:-wmdp_bio}"
EVAL_LARGE="${EVAL_LARGE:-0}"
EVAL_NO_MMLU="${EVAL_NO_MMLU:-0}"
EVAL_WMDP_INCLUDE_PATH="${EVAL_WMDP_INCLUDE_PATH:-}"
EVAL_WMDP_TASK_NAME="${EVAL_WMDP_TASK_NAME:-wmdp_bio_robust}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-256}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-./cache}"
EVAL_DATASET_CACHE_DIR="${EVAL_DATASET_CACHE_DIR:-./cache}"
EVAL_ENG_VALID_FILE="${EVAL_ENG_VALID_FILE:-/home/morg/students/rashkovits/Localized-UNDO/datasets/pretrain/valid_eng.jsonl}"
SKIP_EVAL="${SKIP_EVAL:-0}"
RANDOM_BASELINE="${RANDOM_BASELINE:-1}"
DOWN_PROJ_ONLY="${DOWN_PROJ_ONLY:-0}"
FORGET_ROLES="${FORGET_ROLES:-bio_forget_lean}"

SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_ANALYZE="${SKIP_ANALYZE:-0}"

# --- Parallelism Optimization ---
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "================================================================"
echo " SNMF → analyze → forget ablation pipeline"
echo " Node: ${SLURMD_NODENAME:-local}"
echo " Base model (train + ablate): $MODEL_PATH"
echo " SNMF output (train + read): $SNMF_OUTPUT_DIR"
echo " Ablation run artifacts:      $ABLATION_OUTPUT_DIR"
echo " Learned ablation save:        $SAVE_PATH"
echo " Random baseline save:         $SAVE_PATH_RANDOM  (if RANDOM_BASELINE=1)"
echo " Supervised JSON filename:     $SUPERVISED_JSON_FILENAME"
echo " Eval mode:                    $EVAL_MODE"
echo " SKIP_TRAIN=$SKIP_TRAIN  SKIP_ANALYZE=$SKIP_ANALYZE"
echo "================================================================"

mkdir -p "$SNMF_OUTPUT_DIR"

# ========== 1) Train SNMF ==========
if [[ "$SKIP_TRAIN" == "1" ]]; then
  echo "[1/3] SKIP_TRAIN=1 — skipping train_snmf.py"
else
  echo "[1/3] Training SNMF → $SNMF_OUTPUT_DIR"
  python train_snmf.py \
    --model-path "$MODEL_PATH" \
    --data-path "$DATA_PATH" \
    --output-dir "$SNMF_OUTPUT_DIR" \
    --layers "$LAYERS" \
    --rank "$RANK" \
    --mode "$SNMF_MODE" \
    --init "$SNMF_INIT" \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --device "$TRAIN_DEVICE" \
    --sparsity "$SPARSITY" \
    --max-iter "$MAX_ITER" \
    --seed "$TRAIN_SEED"
  echo "[1/3] Training finished."
fi

# ========== 2) Analyze (supervised roles) ==========
if [[ "$SKIP_ANALYZE" == "1" ]]; then
  echo "[2/3] SKIP_ANALYZE=1 — skipping wmdp_bio_analyze_snmf_results.py"
else
  echo "[2/3] Analyzing SNMF results in $SNMF_OUTPUT_DIR"
  python wmdp_bio_analyze_snmf_results.py \
    --model-path "$MODEL_PATH" \
    --results-dir "$SNMF_OUTPUT_DIR" \
    --summary-filename "$SUMMARY_FILE" \
    --role-assignment-threshold 0.15 \
    --supervised-retain-basis pooled \
    --device "$ANALYZE_DEVICE" \
    --seed "$ANALYZE_SEED" \
    --activation-context-top-n 10 \
    --activation-context-window 15
  echo "[2/3] Analysis finished ($SNMF_OUTPUT_DIR/$SUMMARY_FILE)."
fi

# ========== 3) Forget ablation ==========
echo "[3/3] Forget ablation (read $SNMF_OUTPUT_DIR)"
EVAL_ARGS=()
if [[ "$SKIP_EVAL" == "1" ]]; then
  EVAL_ARGS+=(--skip-eval)
fi
if [[ "$EVAL_LARGE" == "1" ]]; then
  EVAL_ARGS+=(--eval-large)
fi
if [[ "$EVAL_NO_MMLU" == "1" ]]; then
  EVAL_ARGS+=(--eval-no-mmlu)
fi
if [[ -n "$EVAL_WMDP_INCLUDE_PATH" ]]; then
  EVAL_ARGS+=(--eval-wmdp-include-path "$EVAL_WMDP_INCLUDE_PATH")
fi
if [[ -n "$EVAL_WMDP_TASK_NAME" ]]; then
  EVAL_ARGS+=(--eval-wmdp-task-name "$EVAL_WMDP_TASK_NAME")
fi
RANDOM_ARGS=()
if [[ "$RANDOM_BASELINE" == "1" ]]; then
  RANDOM_ARGS+=(--random-baseline)
  RANDOM_ARGS+=(--save-path-random "$SAVE_PATH_RANDOM")
  RANDOM_ARGS+=(--random-seed "$RANDOM_SEED")
fi
DOWN_ONLY_ARGS=()
if [[ "$DOWN_PROJ_ONLY" == "1" ]]; then
  DOWN_ONLY_ARGS+=(--down-proj-only)
fi

python create_forget_ablated_model.py \
  --model-path "$MODEL_PATH" \
  --results-dir "$SNMF_OUTPUT_DIR" \
  --supervised-json-filename "$SUPERVISED_JSON_FILENAME" \
  --save-path "$SAVE_PATH" \
  --forget-roles $FORGET_ROLES \
  --ridge-lambda "$RIDGE_LAMBDA" \
  --device "$ABLATION_DEVICE" \
  --metadata-out "$METADATA_OUT" \
  --eval-json-out "$EVAL_JSON_OUT" \
  --eval-device "$EVAL_DEVICE" \
  --eval-mode "$EVAL_MODE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --eval-max-length "$EVAL_MAX_LENGTH" \
  --eval-cache-dir "$EVAL_CACHE_DIR" \
  --eval-dataset-cache-dir "$EVAL_DATASET_CACHE_DIR" \
  --eval-eng-valid-file "$EVAL_ENG_VALID_FILE" \
  "${RANDOM_ARGS[@]}" \
  "${DOWN_ONLY_ARGS[@]}" \
  "${EVAL_ARGS[@]}"

echo "[3/3] Done. Checkpoint: $SAVE_PATH"
echo "================================================================"
echo " Pipeline complete."
echo " SNMF dir:     $SNMF_OUTPUT_DIR"
echo " Ablation log: $ABLATION_OUTPUT_DIR"
echo " Eval JSON:    $EVAL_JSON_OUT"
echo "================================================================"
