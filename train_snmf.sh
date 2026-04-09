#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=train_snmf_wmdp_bio
#SBATCH --output=logs/train_wmdp_bio_%j.out
#SBATCH --error=logs/train_wmdp_bio_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
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

# Defaults target the WMDP-bio Gemma-2-2b setup.
MODEL_PATH="${MODEL_PATH:-/home/morg/students/rashkovits/Localized-UNDO/models/wmdp/gemma-2-2b}"
DATA_PATH="${DATA_PATH:-data/bio_data.json}"
# Keep outputs separate from earlier arithmetic/0.3B runs.
OUTPUT_DIR="${OUTPUT_DIR:-outputs/snmf_train_results_wmdp_bio_gemma2_2b}"
LAYERS="${LAYERS:-0-25}"        # Gemma-2-2b has 26 layers => indices 0..25
RANK="${RANK:-300}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SNMF_MODE="${SNMF_MODE:-mlp_intermediate}"
SNMF_INIT="${SNMF_INIT:-svd}"
DEVICE="${DEVICE:-auto}"
SPARSITY="${SPARSITY:-0.01}"
MAX_ITER="${MAX_ITER:-3000}"
SEED="${SEED:-42}"
mkdir -p logs "$OUTPUT_DIR" $HF_HOME

# --- Parallelism Optimization ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Execute Training ---
echo "--------------------------------------------------------"
echo "Starting SNMF Training on Node: $SLURMD_NODENAME"
echo "Model path: $MODEL_PATH"
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Layers: $LAYERS"
echo "--------------------------------------------------------"

python train_snmf.py \
    --model-path "$MODEL_PATH" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --layers "$LAYERS" \
    --rank "$RANK" \
    --mode "$SNMF_MODE" \
    --init "$SNMF_INIT" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --sparsity "$SPARSITY" \
    --max-iter "$MAX_ITER" \
    --seed "$SEED"

echo "--------------------------------------------------------"
echo "SNMF Training Finished"