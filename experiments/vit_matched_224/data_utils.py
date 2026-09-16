import random
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Sampler
from torchvision import datasets, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class ImageFolderWithPath(datasets.ImageFolder):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        path, stored = self.samples[index]
        if target != stored: raise RuntimeError('dataset label mismatch')
        return image, target, path

def train_transform():
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.08,1.0), ratio=(3/4,4/3), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

def val_transform():
    return transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

def make_train_dataset(root): return ImageFolderWithPath(str(Path(root)/'train'), transform=train_transform())
def make_val_dataset(root): return ImageFolderWithPath(str(Path(root)/'val'), transform=val_transform())

def verify_standard_val_distribution(ds):
    counts = Counter(y for _, y in ds.samples)
    if len(ds) != 50000: raise RuntimeError(f'expected 50000 validation images, found {len(ds)}')
    if len(counts) != 1000: raise RuntimeError(f'expected 1000 classes, found {len(counts)}')
    bad = {k:v for k,v in counts.items() if v != 50}
    if bad: raise RuntimeError(f'{len(bad)} classes do not contain exactly 50 validation images')
    return counts

def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed); random.seed(seed)

class DistributedEvalSampler(Sampler):
    def __init__(self, dataset, rank=0, world_size=1): self.dataset=dataset; self.rank=rank; self.world_size=world_size
    def __iter__(self): return iter(range(self.rank, len(self.dataset), self.world_size))
    def __len__(self):
        n=len(self.dataset)
        return 0 if self.rank >= n else (n-1-self.rank)//self.world_size + 1


def preprocessing_config():
    return {'train': {'random_resized_crop':224,'scale':[0.08,1.0],
                     'ratio':[0.75,4/3],'horizontal_flip_probability':0.5,
                     'interpolation':'bilinear','antialias':True,'mean':MEAN,'std':STD},
            'validation': {'resize':256,'resize_mode':'shorter_edge','crop':224,
                           'interpolation':'bilinear','antialias':True,'mean':MEAN,'std':STD}}

def make_val_dataset_from_recipe(root, recipe):
    required={'resize','resize_mode','crop','interpolation','antialias','mean','std'}
    if set(recipe)!=required: raise ValueError(f'Recipe must contain exactly {sorted(required)}')
    if recipe['crop']!=224: raise ValueError('This evaluator requires crop=224')
    if recipe['resize_mode']=='shorter_edge' and isinstance(recipe['resize'],int): size=recipe['resize']
    elif recipe['resize_mode']=='fixed_shape' and isinstance(recipe['resize'],list) and len(recipe['resize'])==2: size=tuple(recipe['resize'])
    else: raise ValueError('Invalid resize mode/size')
    if len(recipe['mean'])!=3 or len(recipe['std'])!=3 or any(x<=0 for x in recipe['std']): raise ValueError('Invalid normalization')
    interpolation=transforms.InterpolationMode(recipe['interpolation'])
    transform=transforms.Compose([transforms.Resize(size,interpolation=interpolation,antialias=recipe['antialias']),
        transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(recipe['mean'],recipe['std'])])
    return ImageFolderWithPath(str(Path(root)/'val'),transform=transform)
