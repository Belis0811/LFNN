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

The dynamic top-delta leader selection and follower-to-best-leader MSE are
implemented by the MNIST, CIFAR-10, and ImageNet-subset `*_detailed.ipynb`
notebooks under `model/LFNN`. The matching `.py` files are executable exports.
The scaled ViT, ResNet, brain-age, Tiny-ImageNet, and cell trainers under
`model/LFNN-BPfree` implement the delta=100% all-leader case with multiple
supervised outputs; they do not perform dynamic leader selection or follower
alignment.

The table-producing hyperparameters transcribed from the released logs are
recorded in [`experiment_configs.json`](experiment_configs.json) and in the
training sources. For partial-leadership runs, each manifest entry links the
detailed notebook, its executable export, and the released log. It also records
the batch size, epoch count, optimizer, and leadership setting.

The log-mapped ViT, ResNet, and brain-age training paths preserve the
optimization graphs, optimizer membership, epoch budgets, batch sizes, and
learning-rate schedules used by the corresponding logged runs. Portability
fixes for CPU/single-GPU execution and checkpoint access do not insert new
gradient-detach boundaries or otherwise change the logged training objectives.

The detailed notebooks and these executable exports use the full configurations
corresponding to the released MNIST, CIFAR-10, and ImageNet-subset logs:

```
python model/LFNN/MNIST_LFNN.py
python model/LFNN/Cifar10_LFNN.py
python model/LFNN/ImageNet_LFNN_subset.py
```

The full ImageNet ViT run uses the logged configuration of 90 epochs, batch
size 256, two CUDA devices, and SWAG initialization. Data-loader and scheduler
settings that were not printed in the log retain the values from the released
training source. `main.ipynb` contains the same source as `train_vit.py`:

```
cd model/LFNN-BPfree/ImageNet
python train_vit.py
```

The four-output ImageNet ResNet runs use 200 epochs, matching the released
200-epoch logs:

```
cd model/LFNN-BPfree/ImageNet
python ResNet101_4outputs.py
python ResNet152_4outputs.py
```

The BP trainer selects the corresponding experiment preset from `--model`.
Command-line arguments or `LFNN_*` environment variables may still override a
preset. Each cosine schedule uses the `T_max` recorded by the corresponding
released BP learning-rate trace:

```
python model/BP/train_imagenet_bp_baseline.py --model vit_b_16 --data-root /path/to/imagenet --output-dir runs/vit_bp
python model/BP/train_imagenet_bp_baseline.py --model resnet101 --data-root /path/to/imagenet --output-dir runs/resnet101_bp
python model/BP/train_imagenet_bp_baseline.py --model resnet152 --data-root /path/to/imagenet --output-dir runs/resnet152_bp
```

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


