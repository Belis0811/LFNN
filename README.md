#### Requirements
* **Python** 3.8+
* **Tensorflow** 2.18.0
* **PyTorch** 2.0.0+
* **torchvision** 0.15.0+
*  **NumPy**,**scikit-image**,**matplotlib**,etc
  
Please go to the `requirements.txt` file to check all dependencies.

Or run the following code to install:
```
pip install -r requirements.txt
```

#### Experiment scope and reproducibility

The MNIST, CIFAR-10, and ImageNet-subset partial-leadership implementations
are the detailed notebooks and executable exports under `model/LFNN`.
They implement leader selection and follower-to-leader MSE. The scaled
multi-output trainers under `model/LFNN-BPfree` use the delta=100% all-leader
case, without follower selection or follower alignment. A log's storage
directory does not identify its training implementation.

The explicit source/log associations in
[`experiment_configs.json`](experiment_configs.json) are:

| Experiment | Detailed notebook | Executable export | Released log |
| --- | --- | --- | --- |
| MNIST partial leadership | [MNIST_LFNN_detailed.ipynb](model/LFNN/MNIST_LFNN_detailed.ipynb) | [MNIST_LFNN.py](model/LFNN/MNIST_LFNN.py) | [mnist_lfnn_l_training_log.txt](model/LFNN-BPfree/mnist_lfnn_l_training_log.txt) |
| CIFAR-10 partial leadership | [Cifar10_LFNN_detailed.ipynb](model/LFNN/Cifar10_LFNN_detailed.ipynb) | [Cifar10_LFNN.py](model/LFNN/Cifar10_LFNN.py) | [cifar10_lfnn_l_training_log.txt](model/LFNN-BPfree/cifar10_lfnn_l_training_log.txt) |
| ImageNet subset partial leadership | [ImageNet_LFNN_detailed.ipynb](model/LFNN/ImageNet_LFNN_detailed.ipynb) | [ImageNet_LFNN_subset.py](model/LFNN/ImageNet_LFNN_subset.py) | [imagenet__lfnn_l_training_log.txt](model/LFNN-BPfree/imagenet__lfnn_l_training_log.txt) |

These associations document the intended experiment mapping; agreement between
a released source and log alone does not establish the exact source revision
executed. Configuration fields printed during a run and settings documented
in the released source must be distinguished. A field absent from a historical
log must not be described as directly recorded in that log. Source revisions,
resolved configurations, execution commands, and output hashes should accompany
new runs.

The partial-leadership exports currently enable `use_global_loss=True` by
default, adding a global output cross-entropy to the leader and follower
objectives. Their default objective should therefore not be described as
having only local losses. This is separate from the block-local ViT control
described below.

Run the released partial-leadership exports with:

```bash
python model/LFNN/MNIST_LFNN.py
python model/LFNN/Cifar10_LFNN.py
python model/LFNN/ImageNet_LFNN_subset.py
```

#### Released ImageNet training configurations

The historical LFNN ViT log records 90 epochs, batch size 256, two CUDA
devices, and SWAG initialization. The released
[`train_vit.py`](model/LFNN-BPfree/ImageNet/train_vit.py) and configuration
manifest specify the remaining training settings, including two data-loader
workers and `CosineAnnealingLR(T_max=200)`. Settings obtained from the
released source are source-documented settings, not additional fields in the
historical log.

```bash
cd model/LFNN-BPfree/ImageNet
python train_vit.py
```

The BP trainer uses the following released presets. All three presets train
for 90 epochs with batch size 256, eight data-loader workers, SGD with
momentum 0.9, and an epoch-stepped cosine learning-rate schedule:

| BP model | Initialization | Initial LR | Weight decay | Cosine T_max | eta_min |
| --- | --- | --- | --- | --- | --- |
| ViT-B/16 | SWAG E2E at 224; incompatible positional embedding excluded | 0.001 | 0.0005 | 90 | 0 |
| ResNet-101 | From scratch | 0.1 | 0.00005 | 89 | 0.0001 |
| ResNet-152 | From scratch | 0.05 | 0.00005 | 89 | 0.00001 |

The historical ViT BP log explicitly identifies `CosineAnnealingLR`; its
configuration header does not print `T_max`. The table reports the preset
in the released source and manifest. The historical BP and LFNN ViT recipes
have different scheduler periods (90 and 200, respectively), so they should
not be described as identical training configurations. Specifying SGD and
its initial learning rate alone is not a complete description of either
scheduled run.

Command-line arguments or `LFNN_*` environment variables may override BP
presets. Inspect the resolved configuration for the actual invocation:

```bash
python model/BP/train_imagenet_bp_baseline.py --model vit_b_16 --data-root /path/to/imagenet --output-dir runs/vit_bp
python model/BP/train_imagenet_bp_baseline.py --model resnet101 --data-root /path/to/imagenet --output-dir runs/resnet101_bp
python model/BP/train_imagenet_bp_baseline.py --model resnet152 --data-root /path/to/imagenet --output-dir runs/resnet152_bp
```

The released four-output LFNN ResNet trainers use 90 epochs and cosine
`T_max=200`; see the manifest for their model-specific batch sizes and
optimizer settings:

```bash
cd model/LFNN-BPfree/ImageNet
python ResNet101_4outputs.py
python ResNet152_4outputs.py
```

#### Matched ViT-B/16 control at 224 x 224

[`experiments/vit_matched_224`](experiments/vit_matched_224/README.md)
provides a separate controlled BP/LFNN training and checkpoint-evaluation
pipeline. Within each seed, both methods start from identical shared ViT
weights, including the newly initialized positional embedding. Both use
224 x 224 inputs, 90 epochs, global batch size 256, and SGD with initial
learning rate 0.001, momentum 0.9, weight decay 0.0005, and cosine
`T_max=90, eta_min=0`.

The LFNN arm has four consecutive three-transformer-layer workers, three
auxiliary heads, and activation detachment between workers. Its four
minibatch-mean cross-entropies are averaged (weight 0.25 each); BP uses
the final cross-entropy with weight 1. This is the delta=100% all-leader
control and does not test partial leadership or follower alignment.

The package includes official validation-label checks, paired initialization
checks, source/configuration capture, raw metric export, checkpoint reload
evaluation, and artifact-consistency validation. It contains code and utility
tests, with no experimental accuracy claims. It does not establish that this
source produced any previously released logs. See the package README for
data requirements, execution commands, loss scaling, and the checks actually
performed before release.

#### 1. LFNN
Our LFNN codes can be run properly in Google Colab with the links below:

[Cifar10_LFNN](https://colab.research.google.com/github/Belis0811/LFNN/blob/main/model/LFNN/Cifar10_LFNN.ipynb)

[MNIST_LFNN](https://colab.research.google.com/github/Belis0811/LFNN/blob/main/model/LFNN/MNIST_LFNN.ipynb)

A more detailed version can be found at links below:

[Cifar10_LFNN_Detailed](https://colab.research.google.com/github/Belis0811/LFNN/blob/main/model/LFNN/Cifar10_LFNN_detailed.ipynb)

[MNIST_LFNN_Detailed](https://colab.research.google.com/github/Belis0811/LFNN/blob/main/model/LFNN/MNIST_LFNN_detailed.ipynb)

[ImageNet_LFNN Detailed](https://colab.research.google.com/github/Belis0811/LFNN/blob/main/model/LFNN/ImageNet_LFNN_detailed.ipynb)


#### 2. LFNN-BPfree
Our LFNN-BPfree codes can be run properly in Google Colab with the links below:

[Cifar10_LFNN_BPfree](https://colab.research.google.com/github/Belis0811/LFNN/blob/main/model/LFNN-BPfree/Cifar10_LFNN_BPfree.ipynb)

[MNIST_LFNN_BPfree](https://colab.research.google.com/github/Belis0811/LFNN/blob/main/model/LFNN-BPfree/MNIST_LFNN_BPfree.ipynb)

**To run the Tiny Image code and ImageNet code, please do the preparation step first, then run our codes.**

#### Preparation
Put the Tiny ImageNet dataset into the root folder, then name the dataset folder tiny-imagenet-200. The dataset could be found at https://www.kaggle.com/c/tiny-imagenet/data. Put the ImageNet dataset into the root folder, then name the dataset folder imagenet. Note that imagenet need some more preprocessing, please refer to https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh for detail. The ImageNet dataset is located at https://image-net.org/, and you could download it yourself.


#### Tiny-Imagenet
direct to 'tiny-imagenet' folder
```
cd LFNN/model/LFNN-BPfree/tiny-imagenet
```

train VGG with 4 outputs using tiny ImageNet
```
python main_vgg.py
```

train 8 outputs, 12 outputs and 16 outputs by running
```
python Main_8out.py
```

```
python Main_12out.py
```
and
```
python Main_16out.py
```
#### ImageNet
direct to 'imagenet' folder
```
cd LFNN/model/LFNN-BPfree/ImageNet
```

train ResNet101 with 2 outputs
```
python ResNet101_2outputs.py
```

train ResNet152 with 2 outputs
```
python ResNet152_2outputs.py
```

train ResNet101 with 4 outputs
```
python ResNet101_4outputs.py
```

train ResNet152 with 4 outputs
```
python ResNet152_4outputs.py
```

#### Note that we only provided part of the data for real-world applications to show reproducibility.

#### Brain Age
direct to 'BrainAge' folder
```
cd LFNN/model/LFNN-BPfree/BrainAge
```
Train our LFNN model with 'train_BP_free.py'
```
python train_BP_free.py
```

#### Cell
direct to 'Cell' folder
```
cd LFNN/model/LFNN-BPfree/Cell
```
Train our LFNN-UNet model with 'live_dead_segment.ipynb'

Sample data are stored under the **data** folder in the **Cell** directory

The notebook uses an 80/10/10 seeded split, reports macro metrics on a labeled
held-out test set, and interprets processed mask IDs as `0=background`,
`1=live`, and `2=dead`. It does not merge or remap an injured class. Any
upstream class conversion used to create the `.npy` masks must be disclosed
alongside comparisons with E-U-Net.

#### ResNet-101 partial leadership

The standalone [ResNet-101 partial-leadership implementation](experiments/resnet101_partial_leadership/README.md)
contains a four-worker model, error-based leader selection, detached-teacher
follower alignment, and explicit configurations for delta=0.75. It also includes
no-alignment, all-leader, and matched BP configurations, launch commands,
checkpoint evaluation, and run-recording utilities. Configuration paths are
listed separately under `implementations` in `experiment_configs.json`.
