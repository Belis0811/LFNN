# Matched ViT-B/16 controls at 224 x 224

This package corrects execution and evidence checks. It contains no experimental scores or training results. Existing logs have not been edited. Fixing source code does not establish which source produced an older run.

## Scientific scope

The training launcher implements a **new matched 224×224 control**, not an exact reproduction of the historical released recipe. Both arms use the same seed-specific SWAG-initialized shared ViT parameters, including the same newly initialized 224-position embedding and final classifier. BP propagates the final loss through the full model. The local arm partitions twelve transformer layers into four consecutive three-layer workers, adds three auxiliary heads, and detaches activations at worker boundaries.

This local arm is the **all-leader boundary case (δ=100%)**. There are no followers or alignment-loss updates in this package. It addresses the BP comparison in Comment 2; it does not itself supply the partial-leadership experiment requested in Comment 1.

Each cross-entropy uses a minibatch mean. The existing v4 local objective is preserved: the four cross-entropies are then averaged, giving each loss a weight of 0.25; BP's final loss has weight 1.0. These weights are logged explicitly. Equal optimizer LR therefore does not mean equal gradient scaling per local loss. Do not describe the methods as having identical objectives or identical total parameter counts: LFNN has auxiliary heads.

## Corrections

- Fixed pandas Top-5 recomputation (`any(axis=1)`).
- Validation bypasses DDP forward collectives while retaining distributed count/loss reduction, avoiding uneven-validation-batch collective mismatches.
- All training ranks still run forward through DDP and participate in validation reductions.
- Enforced train/validation class-map equality; official validation checks require unique filenames covering 1–50,000, official WNID classes, and the official per-image ground truth.
- Initial comparisons now require both Top-1 and Top-5 agreement; shared-init hashes link setup checks, paired comparison, and training configuration.
- Recorded full preprocessing, worker count, AMP setting, local loss scaling, partitioning, executed Python source hashes, and class-map digest.
- Added validation of exported CSV values field by field against JSONL records, complete ordered epochs, both accuracy calculations, LR schedule, paired configurations, parameter manifests, initial metrics, final predictions, and checkpoint digests.
- Added strict historical checkpoint evaluation using an explicitly supplied original constructor, final-head selector, and preprocessing JSON. Historical checkpoints are not silently loaded into the new architecture.
- Launchers refuse an existing output directory; evaluation refuses existing result files. No resume facility is provided.

## Environment and data

Install mutually compatible PyTorch and torchvision packages for your CUDA environment, then install `requirements.txt`. Python 3.9 or later is required. The resolved software versions are captured when the launcher runs.

Provide ImageNet under `train/` and `val/`, with identical official WNID class directories. Preserve original `ILSVRC2012_val_XXXXXXXX.JPEG` filenames. Obtain the official validation ground-truth file and devkit `meta.mat` independently; never derive labels from filename blocks.

```bash
export DATA_ROOT=/absolute/path/to/imagenet
export GROUND_TRUTH=/absolute/path/to/ILSVRC2012_validation_ground_truth.txt
export META_MAT=/absolute/path/to/ILSVRC2012_devkit_t12/data/meta.mat
export NPROC=4
export OUT=/absolute/path/to/a_new_run_directory
python -m unittest discover -s . -p 'test_*.py' -v
python verify_local_gradients.py --device cuda
bash run_three_seeds.sh
```

The optional gradient check uses random inputs only to test gradient boundaries, not to estimate accuracy. It requires PyTorch and sufficient memory.

The launcher uses seeds 101, 202, 303; 90 epochs; global batch 256; SGD LR 0.001, momentum 0.9, weight decay 0.0005; cosine schedule T_max=90 and eta_min=0; eight data-loader workers per rank; no AMP. It requires at least one loader worker to isolate augmentation RNG streams from model construction. The global batch must be divisible by NPROC. The default NPROC is four and is recorded, not inferred from historical logs.

Training augmentation is RandomResizedCrop(224), scale [0.08,1], ratio [0.75,4/3], and horizontal flip probability 0.5. Validation uses shorter-edge Resize(256), CenterCrop(224). Both use bilinear interpolation with antialiasing and ImageNet normalization: mean [0.485,0.456,0.406], std [0.229,0.224,0.225]. No label smoothing or mixup is implemented.

Final-epoch accuracy is the primary endpoint. Best-epoch results are recorded separately. No target accuracy is encoded. The same pretrained shared state is prepared once per seed; only the incompatible positional embedding remains newly initialized. This is fine-tuning from SWAG, not training from scratch.

## Evaluate new matched checkpoints

```bash
export BP_CKPT=/absolute/path/to/new_bp_last.pt
export LFNN_CKPT=/absolute/path/to/new_lfnn_last.pt
export OUT=/absolute/path/to/a_new_evaluation_directory
export PROTOCOL=matched
bash run_checkpoint_evaluation.sh
```

## Evaluate historical checkpoints

Use the actual constructors and preprocessing from those runs. An adapter factory must return an uninitialized model with the exact checkpoint architecture/key layout and must not download pretrained weights. Put the original source plus adapter on PYTHONPATH. The evaluator supports a raw state dict or a dict containing `model` or `state_dict`; an ambiguous container requires `--state-dict-key` in a direct evaluator invocation. A uniform DDP `module.` prefix is stripped; all parameters then load strictly. No key-remapping or missing-parameter fallback is performed.

```bash
export PYTHONPATH=/absolute/path/to/original_source
export PROTOCOL=historical
export BP_FACTORY=checkpoint_adapters:build_original_bp
export LFNN_FACTORY=checkpoint_adapters:build_original_lfnn
export BP_SELECTOR=tensor
export LFNN_SELECTOR=index:0
export BP_RECIPE=/absolute/path/to/verified_bp_preprocessing.json
export LFNN_RECIPE=/absolute/path/to/verified_lfnn_preprocessing.json
export BP_CKPT=/absolute/path/to/original_bp_checkpoint.pt
export LFNN_CKPT=/absolute/path/to/original_lfnn_checkpoint.pt
export OUT=/absolute/path/to/a_new_historical_evaluation_directory
bash run_checkpoint_evaluation.sh
```

The factory names above are placeholders for your original source, not provided model definitions. Optional `BP_FACTORY_KWARGS` and `LFNN_FACTORY_KWARGS` point to constructor-argument JSON objects. The final-head selector must match the original model's output. Use `index:0` only after checking that the first returned tensor is the final classifier.

A recipe JSON contains exactly these fields (this example is the **new matched** validation recipe; it is not a statement about the historical recipe):

```json
{"resize":256,"resize_mode":"shorter_edge","crop":224,"interpolation":"bilinear","antialias":true,"mean":[0.485,0.456,0.406],"std":[0.229,0.224,0.225]}
```

For a historical fixed-shape resize, use `"resize":[256,256],"resize_mode":"fixed_shape"`; record the actual historical normalization, interpolation, and antialias setting. Do not assume matching crop size alone establishes matching preprocessing. Preserve the original constructor/repository revision and resolved configuration alongside the evaluation.

## Evidence and checkpoint confidentiality

All checkpoints remain local; no script uploads or publishes them. Local validators can verify hashes and reload-generated predictions without sharing weights. Before sharing any archive, choose which files to disclose; the pipeline does not produce or upload a public archive automatically.

Use `validate_evidence_bundle.py --root RUN --seeds 101 202 303 --require-checkpoints` on the complete local run. Omitting `--require-checkpoints` permits withheld checkpoint files and records that their bytes were not checked. Passing artifact-consistency checks does not authenticate an experiment and does not let a reviewer evaluate unavailable weights.

A numerical mismatch causes failure; the validator never changes metrics. Investigate input transforms, output-head selection, dataset mapping, source identity, checkpoint selection and numerical batch-size effects. Do not adjust recorded results to make checks pass. The code uses exact prediction-count agreement and a small tolerance for loss recomputation; the source does not promise bitwise reproducibility across hardware.

## Verification performed before release

Python compilation, both Bash syntax checks, and 13 focused utility tests passed. Tests use tiny artificial fixtures explicitly confined to temporary test directories. PyTorch, CUDA, ImageNet, the official devkit and the confidential checkpoints were not available here. GPU training, model forward/backward, distributed execution, and historical checkpoint loading have **not** been executed here. The provided gradient test and full launcher remain runtime checks to perform on your machine.
