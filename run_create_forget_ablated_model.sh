#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=forget_ablate
#SBATCH --output=logs/forget_ablate_%j.out
#SBATCH --error=logs/forget_ablate_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G

# --- Environment Setup ---
source /home/morg/students/rashkovits/miniconda3/etc/profile.d/conda.sh
conda activate snmf_env

# --- Space & Cache Management ---
export HF_HOME="/home/morg/students/rashkovits/hf_cache"
export TORCH_HOME="/home/morg/students/rashkovits/hf_cache/torch"
export TMPDIR="/home/morg/students/rashkovits/hf_cache"

# --- Project Setup ---
cd /home/morg/students/rashkovits/snmf
export PYTHONPATH=$PYTHONPATH:$(pwd)

mkdir -p logs $HF_HOME

# --- I/O (override with sbatch --export=ALL,RESULTS_DIR=... or env before sbatch) ---
MODEL_PATH="${MODEL_PATH:-local_models/gemma-2-0.3B_reference_model}"
RESULTS_DIR="${RESULTS_DIR:-outputs/snmf_train_results}"
SAVE_PATH="${SAVE_PATH:-local_models/gemma-2-0.3B_forget_ablated}"
SAVE_PATH_RANDOM="${SAVE_PATH_RANDOM:-${SAVE_PATH}_random_baseline}"
METADATA_OUT="${METADATA_OUT:-${RESULTS_DIR}/forget_ablation_metadata.json}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1e-6}"
RANDOM_SEED="${RANDOM_SEED:-1234}"

# Weight-edit pass: auto uses utils.resolve_device (CPU on sm_61 + sm_70+ PyTorch wheels).
# Force cuda only on supported GPUs: ABLATION_DEVICE=cuda
ABLATION_DEVICE="${ABLATION_DEVICE:-auto}"
# Standalone eval (evaluation/eveluate_model.py): auto -> cuda if usable, else cpu.
EVAL_DEVICE="${EVAL_DEVICE:-auto}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-256}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-./cache}"
EVAL_DATASET_CACHE_DIR="${EVAL_DATASET_CACHE_DIR:-./cache}"
EVAL_ENG_VALID_FILE="${EVAL_ENG_VALID_FILE:-data/valid_eng.jsonl}"
# Set SKIP_EVAL=1 to skip before/after accuracy (faster).
SKIP_EVAL="${SKIP_EVAL:-0}"
# Set RANDOM_BASELINE=0 to disable random matched-count baseline.
RANDOM_BASELINE="${RANDOM_BASELINE:-1}"

# Space-separated role names (same defaults as create_forget_ablated_model.py).
FORGET_ROLES="${FORGET_ROLES:-mult_forget div_forget forget_mixed}"

# --- Parallelism Optimization ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Execute ---
echo "--------------------------------------------------------"
echo "Forget ablation + optional eval on Node: $SLURMD_NODENAME"
echo "Base model:        $MODEL_PATH"
echo "SNMF results dir:  $RESULTS_DIR"
echo "Save ablated to:   $SAVE_PATH"
echo "Save random to:    $SAVE_PATH_RANDOM"
echo "Metadata JSON:     $METADATA_OUT"
echo "Forget roles:      $FORGET_ROLES"
echo "Ablation device:   $ABLATION_DEVICE | Eval device: $EVAL_DEVICE"
echo "Random baseline:   $RANDOM_BASELINE | Random seed: $RANDOM_SEED"
echo "Skip eval:         $SKIP_EVAL"
echo "--------------------------------------------------------"

EVAL_ARGS=()
if [[ "$SKIP_EVAL" == "1" ]]; then
  EVAL_ARGS+=(--skip-eval)
fi

RANDOM_ARGS=()
if [[ "$RANDOM_BASELINE" == "1" ]]; then
  RANDOM_ARGS+=(--random-baseline)
  RANDOM_ARGS+=(--save-path-random "$SAVE_PATH_RANDOM")
  RANDOM_ARGS+=(--random-seed "$RANDOM_SEED")
fi

python create_forget_ablated_model.py \
  --model-path "$MODEL_PATH" \
  --results-dir "$RESULTS_DIR" \
  --save-path "$SAVE_PATH" \
  --forget-roles $FORGET_ROLES \
  --ridge-lambda "$RIDGE_LAMBDA" \
  --device "$ABLATION_DEVICE" \
  --metadata-out "$METADATA_OUT" \
  --eval-device "$EVAL_DEVICE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --eval-max-length "$EVAL_MAX_LENGTH" \
  --eval-cache-dir "$EVAL_CACHE_DIR" \
  --eval-dataset-cache-dir "$EVAL_DATASET_CACHE_DIR" \
  --eval-eng-valid-file "$EVAL_ENG_VALID_FILE" \
  "${RANDOM_ARGS[@]}" \
  "${EVAL_ARGS[@]}"

echo "--------------------------------------------------------"
echo "Forget ablation job finished"
echo "Checkpoint: $SAVE_PATH"
echo "Random checkpoint (if enabled): $SAVE_PATH_RANDOM"
echo "Eval comparison (if run): $SAVE_PATH/ablation_eval_comparison.json"
echo "--------------------------------------------------------"
