#!/usr/bin/env bash

PYTHONPATH=. python experiments/train/train.py \
    --sparsity 0.01 \
    --ranks 50 \
    --max-iterations-per-layer 2000 \
    --patience 1500 \
    --model-name local_models/gemma-2-0.3B_reference_model \
    --factorization-mode mlp \
    --layers 0 \
    --data-path data/final_dataset_20_concepts.json \
    --model-device mps \
    --data-device cpu \
    --fitting-device mps \
    --base-path . \
    --save-path experiments/artifacts/ \
    --seed 42
