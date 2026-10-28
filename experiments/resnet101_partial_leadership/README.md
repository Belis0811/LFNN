# ResNet-101 partial leadership

Standalone ImageNet-1K training with four ResNet-101 workers, error-based
leader selection, and follower alignment. The implementation was recovered
from the earlier LFNN experiment-script package.

## Model and objective

The workers are `stem + layer1`, `layer2`, `layer3`, and `layer4`.
Each worker has a local classifier. With `leadership_fraction: 0.75`, the
three workers with the lowest global-minibatch mean cross-entropies become
leaders and the remaining worker becomes the follower. Assignments are
recomputed each optimization step.

The follower minimizes MSE against the detached class probabilities of the
best leader. Worker inputs are detached at block boundaries. A separate final
classifier receives detached last-worker features, so its supervised loss
updates that classifier only. The default objective weights are 1.0 for the
output classifier, 0.5 for leader supervision, and 0.5 for follower alignment.

## Install

From the repository root:

```bash
cd experiments/resnet101_partial_leadership
python -m pip install -r requirements.txt
python -m pip install -e .
python -m unittest discover -s tests -v
```

The tests check leader counts, error-based selection, and follower gradients.
PyTorch and torchvision are required for training and tensor tests.

## Train

ImageNet must have matching `train/<class_name>` and `val/<class_name>`
directories. Set paths and run the two-GPU launcher:

```bash
export IMAGENET_ROOT=/absolute/path/to/imagenet
export LFNN_RUN_ROOT=/absolute/path/to/runs
bash commands/run_resnet101_delta75.sh
```

For a single-device run, use the Python trainer directly:

```bash
python scripts/train_imagenet_lfnn.py \
  --config configs/imagenet_resnet101_delta75.yaml \
  --data-root "$IMAGENET_ROOT" \
  --output-dir "$LFNN_RUN_ROOT/resnet101_delta75_single_device"
```

The configuration specifies 90 epochs, global batch size 256, SGD with initial
learning rate 0.1, momentum 0.9, Nesterov momentum, weight decay 1e-4, and cosine
decay with `T_max=90`. Set `experiment.seed` and a distinct output directory
in the configuration for each repetition.

Control configurations and launchers are included for:

- `imagenet_resnet101_delta75_no_alignment.yaml`: same selection, follower weight zero.
- `imagenet_resnet101_delta100.yaml`: all four workers are leaders.
- `imagenet_resnet101_bp.yaml`: end-to-end BP under the matched data/optimizer recipe.

## Outputs and evaluation

Executed runs write `resolved_config.yaml`, `run_metadata.json`, per-epoch
`metrics.jsonl`, per-step `selection_trace.jsonl`, and final/best checkpoints.
The trace records selected workers, follower loss, and worker gradient norms.
The trainer records the executed source/configuration hashes. Reported runs
should start from a committed, clean checkout; the launcher enforces the
source check unless explicitly overridden.

```bash
python scripts/validate_lfnn_run.py \
  --run-dir "$LFNN_RUN_ROOT/imagenet_resnet101_lfnn_delta75" \
  --output-json "$LFNN_RUN_ROOT/imagenet_resnet101_lfnn_delta75/validation_report.json"
```

`scripts/eval_imagenet_checkpoint.py` provides checkpoint evaluation and
optional per-image prediction export. Use `--help` for model and preprocessing
arguments. Keep final-epoch and best-epoch evaluations separate.

## Release scope

This directory supplies source code and explicit configuration choices; it
contains no recorded accuracy values or historical training outputs. The
association between a historical result and its executed source must be
established from that run's records. The reference defaults here are
minibatch-based selection, probability-space MSE, and a 256-unit local-head
hidden layer. They must not be silently substituted for a run that used
per-sample selection, logit-space alignment, or another head architecture.

Training is sequential across workers within each process. Its elapsed time
is not a measurement of a block-parallel implementation.

