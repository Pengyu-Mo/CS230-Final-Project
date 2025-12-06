#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# For reproducible cuDNN behavior (optional)
# export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -u ../train/resnet_vad.py