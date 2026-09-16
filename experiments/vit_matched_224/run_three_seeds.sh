#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to ImageNet root with train/ and val/}"
GROUND_TRUTH="${GROUND_TRUTH:?Set GROUND_TRUTH to ILSVRC2012_validation_ground_truth.txt}"
META_MAT="${META_MAT:?Set META_MAT to ILSVRC2012_devkit_t12/data/meta.mat}"
NPROC="${NPROC:-4}"
OUT="${OUT:-runs_matched_vit}"
SEEDS="${SEEDS:-101 202 303}"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -e "$OUT" ]]; then echo "OUT already exists; choose a fresh directory" >&2; exit 1; fi
mkdir -p "$OUT"
PROVENANCE_ARGS=()
if [[ -n "${REPO_DIR:-}" ]]; then PROVENANCE_ARGS+=(--repo-dir "$REPO_DIR"); fi
python "$SOURCE_DIR/capture_provenance.py" --source-dir "$SOURCE_DIR" --output-dir "$OUT/provenance" "${PROVENANCE_ARGS[@]}"
python "$SOURCE_DIR/verify_imagenet_val_official.py" --val-root "$DATA_ROOT/val" --ground-truth "$GROUND_TRUTH" --meta-mat "$META_MAT" --output-dir "$OUT/validation_manifest"
mkdir -p "$OUT/shared_init"
for SEED in $SEEDS; do
  INIT="$OUT/shared_init/seed_${SEED}.pt"
  python "$SOURCE_DIR/prepare_shared_init.py" --seed "$SEED" --output "$INIT" 2>&1 | tee "$OUT/shared_init/seed_${SEED}_prepare.log"
  python "$SOURCE_DIR/verify_pair_setup.py" --shared-init "$INIT" --seed "$SEED" --output "$OUT/shared_init/seed_${SEED}_pair_check.json"
  mkdir -p "$OUT/initial_comparison/seed_${SEED}"
  python "$SOURCE_DIR/compare_initial_predictions.py" --shared-init "$INIT" --data-root "$DATA_ROOT" --output-dir "$OUT/initial_comparison/seed_${SEED}" 2>&1 | tee "$OUT/initial_comparison/seed_${SEED}/console.log"
  for METHOD in bp lfnn; do
    RUN_DIR="$OUT/seed_${SEED}/${METHOD}"; mkdir -p "$RUN_DIR"
    TRAIN_COMMAND=(torchrun --standalone --nproc_per_node="$NPROC" "$SOURCE_DIR/train_matched_vit.py" --method "$METHOD" --seed "$SEED" --data-root "$DATA_ROOT" --shared-init "$INIT" --output-dir "$RUN_DIR" --epochs 90 --batch-size 256 --lr 0.001 --momentum 0.9 --weight-decay 0.0005 --eta-min 0 --workers 8)
    printf '%q ' "${TRAIN_COMMAND[@]}" > "$RUN_DIR/command.txt"
    printf '\n' >> "$RUN_DIR/command.txt"
    "${TRAIN_COMMAND[@]}" 2>&1 | tee "$RUN_DIR/console.log"
    python "$SOURCE_DIR/export_reviewer_metrics.py" "$RUN_DIR/metrics.jsonl" --output "$RUN_DIR/reviewer_metrics.csv"
    python "$SOURCE_DIR/evaluate_checkpoint_224.py" --method "$METHOD" --checkpoint "$RUN_DIR/last.pt" --data-root "$DATA_ROOT" --output-dir "$RUN_DIR/final_checkpoint_re_evaluation" 2>&1 | tee "$RUN_DIR/final_checkpoint_re_evaluation.log"
    python "$SOURCE_DIR/recompute_metrics.py" "$RUN_DIR/final_checkpoint_re_evaluation/predictions.csv" > "$RUN_DIR/final_checkpoint_recomputed_metrics.txt"
  done
done
python "$SOURCE_DIR/validate_evidence_bundle.py" --root "$OUT" --seeds $SEEDS --require-checkpoints
python "$SOURCE_DIR/summarize_paired_runs.py" --root "$OUT" --seeds $SEEDS 2>&1 | tee "$OUT/paired_summary_console.txt"
echo "Evidence bundle completed successfully: $OUT"
