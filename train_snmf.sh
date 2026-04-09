#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=train_snmf
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
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

# Default output dir is separate from outputs/snmf_train_results so existing runs are not overwritten.
OUTPUT_DIR="${OUTPUT_DIR:-outputs/snmf_train_results_pipeline}"
mkdir -p logs "$OUTPUT_DIR" $HF_HOME

# --- Parallelism Optimization ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Execute Training ---
echo "--------------------------------------------------------"
echo "Starting SNMF Training on Node: $SLURMD_NODENAME"
echo "Output directory: $OUTPUT_DIR"
echo "--------------------------------------------------------"

python train_snmf.py \
    --model-path "local_models/gemma-2-0.3B_reference_model" \
    --data-path "data/data.json" \
    --output-dir "$OUTPUT_DIR" \
    --layers "0-13" \
    --rank 100 \
    --mode "mlp_intermediate" \
    --init "svd" \
    --batch-size 8 \
    --device "auto" \
    --sparsity 0.01 \
    --max-iter 5000 \
    --seed 42

echo "--------------------------------------------------------"
echo "SNMF Training Finished"