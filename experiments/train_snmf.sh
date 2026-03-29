#!/usr/bin/env bash
# Use cuda when your GPU is supported by this PyTorch build (sm_70+). Older GPUs (e.g. Pascal sm_61)
# fall back to CPU automatically in train.py. On Apple Silicon only, use mps for model/fitting.

PYTHONPATH=. python experiments/train/train.py \
    --sparsity 0.01 \
    --ranks 50 \
    --max-iterations-per-layer 2000 \
    --patience 1500 \
    --model-name local_models/gemma-2-0.3B_reference_model \
    --factorization-mode mlp \
    --layers 0 \
    --data-path data/final_dataset_20_concepts.json \
    --model-device auto \
    --data-device cpu \
    --fitting-device auto \
    --base-path . \
    --save-path experiments/artifacts/ \
    --seed 42
