#!/usr/bin/env bash
set -euo pipefail

: "${IMAGENET_ROOT:?Set IMAGENET_ROOT to the ImageNet directory}"
: "${LFNN_RUN_ROOT:?Set LFNN_RUN_ROOT to the output parent directory}"

run_dir="${LFNN_RUN_ROOT}/imagenet_resnet101_lfnn_delta75_no_alignment"
mkdir -p "$run_dir"

torchrun --standalone --nproc-per-node=2 scripts/train_imagenet_lfnn.py \
  --config configs/imagenet_resnet101_delta75_no_alignment.yaml \
  --data-root "$IMAGENET_ROOT" \
  --output-dir "$run_dir" \
  2>&1 | tee "$run_dir/console.log"

