#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT}"
GROUND_TRUTH="${GROUND_TRUTH:?Set GROUND_TRUTH}"
META_MAT="${META_MAT:?Set META_MAT}"
BP_CKPT="${BP_CKPT:?Set BP_CKPT}"
LFNN_CKPT="${LFNN_CKPT:?Set LFNN_CKPT}"
OUT="${OUT:-checkpoint_eval}"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -e "$OUT" ]]; then echo "OUT already exists; choose a fresh directory" >&2; exit 1; fi
mkdir -p "$OUT"
python "$SOURCE_DIR/verify_imagenet_val_official.py" --val-root "$DATA_ROOT/val" --ground-truth "$GROUND_TRUTH" --meta-mat "$META_MAT" --output-dir "$OUT/validation_manifest"
for METHOD in bp lfnn; do
  if [[ "$METHOD" == "bp" ]]; then CKPT="$BP_CKPT"; else CKPT="$LFNN_CKPT"; fi
  EVAL_ARGS=()
  if [[ "${PROTOCOL:-matched}" == "historical" ]]; then
    if [[ "$METHOD" == "bp" ]]; then
      FACTORY="${BP_FACTORY:?Set exact historical BP factory module:function}"
      SELECTOR="${BP_SELECTOR:?Set BP_SELECTOR e.g. tensor}"
      RECIPE="${BP_RECIPE:?Set path to historical BP recipe JSON}"
      KWARGS="${BP_FACTORY_KWARGS:-}"
    else
      FACTORY="${LFNN_FACTORY:?Set exact historical LFNN factory module:function}"
      SELECTOR="${LFNN_SELECTOR:?Set LFNN_SELECTOR e.g. index:0}"
      RECIPE="${LFNN_RECIPE:?Set path to historical LFNN recipe JSON}"
      KWARGS="${LFNN_FACTORY_KWARGS:-}"
    fi
    EVAL_ARGS+=(--protocol historical --model-factory "$FACTORY" --output-selector "$SELECTOR" --recipe-json "$RECIPE")
    if [[ -n "$KWARGS" ]]; then EVAL_ARGS+=(--factory-kwargs-json "$KWARGS"); fi
  elif [[ "${PROTOCOL:-matched}" != "matched" ]]; then
    echo "PROTOCOL must be matched or historical" >&2; exit 1
  fi
  python "$SOURCE_DIR/evaluate_checkpoint_224.py" "${EVAL_ARGS[@]}" --method "$METHOD" --checkpoint "$CKPT" --data-root "$DATA_ROOT" --output-dir "$OUT/$METHOD" 2>&1 | tee "$OUT/${METHOD}_console.log"
  python "$SOURCE_DIR/validate_checkpoint_evaluation.py" --evaluation-dir "$OUT/$METHOD" --validation-manifest-dir "$OUT/validation_manifest" --checkpoint "$CKPT"
  python "$SOURCE_DIR/recompute_metrics.py" "$OUT/$METHOD/predictions.csv" > "$OUT/${METHOD}_recomputed_metrics.txt"
done
