#### Requirements
* **Python** 3.8+
* **Tensorflow** 1.15+
* **PyTorch** 2.0.0+
* **torchvision** 0.15.0+
*  **NumPy**,**scikit-image**,**matplotlib**,etc
  
Please go to the `requirement.txt` file to check all dependencies.

Or run the following code to install:
```
pip install -r requirements.txt
```

#### 1. LFNN
Our LFNN codes can be run properly in Google Colab with the links below:

[Cifar10_LFNN](https://colab.research.google.com/github/Belis0811/LFNN/blob/main/model/LFNN/Cifar10_LFNN.ipynb)

[MNIST_LFNN](https://colab.research.google.com/github/Belis0811/LFNN/blob/main/model/LFNN/MNIST_LFNN.ipynb)

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
cd LFNN/model/LFNN-BPFree/tiny-imagenet
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
cd LFNN/model/LFNN-BPFree/imagenet
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
cd LFNN/model/LFNN-BPFree/BrainAge
```
**Because our data is too big, github cannot hold such big files. We provided the Google Drive link that contains sample data npy files. You could find the link under [LFNN/model/LFNN-BPFree/BrainAge/data/data_download.txt](https://drive.google.com/drive/folders/1NQ4V68W72q-OPbDHB_--oh1_Gomkzr7l?usp=sharing)**

Train our LFNN model with 'train_BP_free.py'
```
python train_BP_free.py
```
Sample data are stored under the **data** folder in the **BrainAge** directory

#### Cell
direct to 'Cell' folder
```
cd LFNN/model/LFNN-BPFree/Cell
```
Train our LFNN-UNet model with 'live_dead_segment.ipynb'

Sample data are stored under the **data** folder in the **Cell** directory


