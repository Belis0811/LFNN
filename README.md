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
implemented by the MNIST, CIFAR-10, and ImageNet-subset scripts under
`model/LFNN`. The scaled ViT, ResNet, brain-age, Tiny-ImageNet, and cell
trainers under `model/LFNN-BPfree` implement the LFNN-l special case as
block-local deep supervision; they do not perform dynamic leader selection.

The table-producing hyperparameters transcribed from the released logs are recorded in
[`experiment_configs.json`](experiment_configs.json) and in the executable
training scripts. Each manifest entry links one code path to one released log
and records the batch size, epoch count, optimizer, and leadership setting.

The current scaled trainers also contain correctness repairs to device handling,
block boundaries, model forward passes, and evaluation. Those repairs change
the implementation relative to the historical log-producing source. Therefore,
the historical logs must not be presented as outputs of the corrected code;
fresh runs and logs are required before reporting results from this revision.

Leader-follower runs corresponding to the released MNIST, CIFAR-10, and
ImageNet-subset logs:

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

The BP trainer selects the corresponding experiment preset from `--model`. Command-line
arguments or `LFNN_*` environment variables may still override a preset. Set
`LFNN_SEED` for each replicate and record the generated log and checkpoints:

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
**Because our data is too big, GitHub cannot hold such big files. We provided the Google Drive link that contains sample data npy files. You can also find the link under `LFNN/model/LFNN-BPfree/BrainAge/data/data_dowload.txt`: [brain-age sample data](https://drive.google.com/drive/folders/1NQ4V68W72q-OPbDHB_--oh1_Gomkzr7l?usp=sharing).**

Train our LFNN model with 'train_BP_free.py'
```
python train_BP_free.py
```
Sample data are stored under the **data** folder in the **BrainAge** directory

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


