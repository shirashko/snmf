#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=forget_ablate_wmdp_bio
#SBATCH --output=logs/forget_ablate_wmdp_bio_%j.out
#SBATCH --error=logs/forget_ablate_wmdp_bio_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G

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

# ========== Defaults: WMDP-bio + Gemma-2-2b ==========
# Must point to the same model used for SNMF training/analysis.
MODEL_PATH="${MODEL_PATH:-/home/morg/students/rashkovits/Localized-UNDO/models/wmdp/gemma-2-2b}"
# Directory containing layer_*/snmf_factors.pt and supervised role JSON.
RESULTS_DIR="${RESULTS_DIR:-outputs/snmf_train_results_wmdp_bio_gemma2_2b}"
SUPERVISED_JSON_FILENAME="${SUPERVISED_JSON_FILENAME:-feature_analysis_supervised_wmdp_bio.json}"

# Ablated checkpoint output paths.
SAVE_PATH="${SAVE_PATH:-local_models/wmdp/gemma-2-2b_wmdp_bio_forget_ablated}"
SAVE_PATH_RANDOM="${SAVE_PATH_RANDOM:-local_models/wmdp/gemma-2-2b_wmdp_bio_forget_ablated_random_baseline}"

# Ablation controls.
FORGET_ROLES="${FORGET_ROLES:-bio_forget_lean}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1e-6}"
SPAN_PROJECTION_SCALE="${SPAN_PROJECTION_SCALE:-1.0}"
ABLATION_DEVICE="${ABLATION_DEVICE:-auto}"
DOWN_PROJ_ONLY="${DOWN_PROJ_ONLY:-0}"

# Random matched-count baseline controls.
RANDOM_BASELINE="${RANDOM_BASELINE:-1}"
RANDOM_SEED="${RANDOM_SEED:-1234}"

# Eval controls (run by default before/after ablation).
# Default EVAL_MODE=wmdp_bio runs standard lm-eval task "wmdp_bio" (+ MMLU unless EVAL_NO_MMLU=1).
# Do not set EVAL_WMDP_* below unless you use EVAL_MODE=wmdp_bio_categorized (robust/shortcut YAML groups).
SKIP_EVAL="${SKIP_EVAL:-0}"
EVAL_DEVICE="${EVAL_DEVICE:-auto}"
EVAL_MODE="${EVAL_MODE:-wmdp_bio}"
EVAL_LARGE="${EVAL_LARGE:-0}"
EVAL_NO_MMLU="${EVAL_NO_MMLU:-0}"
EVAL_WMDP_INCLUDE_PATH="${EVAL_WMDP_INCLUDE_PATH:-}"
EVAL_WMDP_TASK_NAME="${EVAL_WMDP_TASK_NAME:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

# --- Parallelism Optimization ---
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "================================================================"
echo " Running create_forget_ablated_model.py (WMDP-bio)"
echo " Node: ${SLURMD_NODENAME:-local}"
echo " Model path:                   $MODEL_PATH"
echo " Results dir:                  $RESULTS_DIR"
echo " Supervised JSON filename:     $SUPERVISED_JSON_FILENAME"
echo " Save path (learned):          $SAVE_PATH"
echo " Save path (random baseline):  $SAVE_PATH_RANDOM  (if RANDOM_BASELINE=1)"
echo " Forget roles:                 $FORGET_ROLES"
echo " Eval mode:                    $EVAL_MODE"
if [[ "$EVAL_MODE" == "wmdp_bio" ]]; then
  echo " Eval tasks:                   lm-eval wmdp_bio (+ MMLU unless EVAL_NO_MMLU=1)"
elif [[ "$EVAL_MODE" == "wmdp_bio_categorized" ]]; then
  echo " Eval (categorized):           include=${EVAL_WMDP_INCLUDE_PATH:-<unset>} task=${EVAL_WMDP_TASK_NAME:-wmdp_bio_robust}"
fi
echo "================================================================"

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
if [[ "$EVAL_MODE" == "wmdp_bio_categorized" ]]; then
  if [[ -n "$EVAL_WMDP_INCLUDE_PATH" ]]; then
    EVAL_ARGS+=(--eval-wmdp-include-path "$EVAL_WMDP_INCLUDE_PATH")
  fi
  EVAL_ARGS+=(--eval-wmdp-task-name "${EVAL_WMDP_TASK_NAME:-wmdp_bio_robust}")
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
  --results-dir "$RESULTS_DIR" \
  --supervised-json-filename "$SUPERVISED_JSON_FILENAME" \
  --save-path "$SAVE_PATH" \
  --forget-roles $FORGET_ROLES \
  --ridge-lambda "$RIDGE_LAMBDA" \
  --span-projection-scale "$SPAN_PROJECTION_SCALE" \
  --device "$ABLATION_DEVICE" \
  --eval-device "$EVAL_DEVICE" \
  --eval-mode "$EVAL_MODE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  "${RANDOM_ARGS[@]}" \
  "${DOWN_ONLY_ARGS[@]}" \
  "${EVAL_ARGS[@]}"

echo "================================================================"
echo " Done."
echo " Learned checkpoint: $SAVE_PATH"
echo " Eval JSON:          $SAVE_PATH/ablation_eval_comparison.json"
echo "================================================================"
