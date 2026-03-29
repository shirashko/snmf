#!/bin/bash
#SBATCH --job-name=snmf-train
#SBATCH --output=/home/morg/students/rashkovits/snmf/logs/train-snmf-%j.out
#SBATCH --error=/home/morg/students/rashkovits/snmf/logs/train-snmf-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=studentkillable

set -euo pipefail

ROOT="/home/morg/students/rashkovits/snmf"
cd "$ROOT"

# Keep pip/HF temps off small $HOME quota (optional but recommended)
export TMPDIR="${TMPDIR:-/home/morg/students/rashkovits/tmp}"
mkdir -p "$TMPDIR"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/home/morg/students/rashkovits/hf_cache/pip-cache}"
mkdir -p "$PIP_CACHE_DIR"
export HF_HOME="${HF_HOME:-/home/morg/students/rashkovits/hf_cache/huggingface}"

# Conda
source "/home/morg/students/rashkovits/miniconda3/etc/profile.d/conda.sh"
conda activate "/home/morg/students/rashkovits/envs/snmf"

export PYTHONPATH=.

python experiments/train/train.py \
    --sparsity 0.01 \
    --ranks 50 \
    --max-iterations-per-layer 2000 \
    --patience 1500 \
    --model-name local_models/gemma-2-0.3B_reference_model \
    --factorization-mode mlp \
    --layers 0 \
    --data-path data/final_dataset_20_concepts.json \
    --model-device cuda \
    --data-device cpu \
    --fitting-device cuda \
    --base-path . \
    --save-path experiments/artifacts/ \
    --seed 42
