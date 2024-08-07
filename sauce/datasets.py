from abc import ABCMeta, abstractmethod
from tqdm import tqdm
from os.path import join, basename, dirname
import pandas as pd
from PIL import Image
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms as T

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torchvision.transforms.functional import InterpolationMode

from wilds import get_dataset
from kornia.augmentation import (RandomAffine, RandomHorizontalFlip, RandomVerticalFlip, Resize, RandomCrop,
                                 Normalize)

# NORMALIZE = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# taken from resnet-50 on imagenet
NORM_CONSTANTS = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class TrainingAugments(torch.nn.Module):
    def __init__(self, p_affine: float, p_flip_h: float,
                 scale: float, shift: float, degrees: float) -> None:
        super().__init__()
        random_affine = RandomAffine(degrees=degrees, scale=(1-scale, 1+scale), translate=(shift, shift), p=p_affine)
        # flip_vertical = RandomVerticalFlip(p=p_flip_v)
        flip_horizontal = RandomHorizontalFlip(p=p_flip_h)
        normalize = Normalize(mean=NORM_CONSTANTS[0], std=NORM_CONSTANTS[1])
        # crop = RandomCrop((h_res, w_res))
        self.transforms = torch.nn.Sequential(#flip_vertical,
                                              # crop,
                                              flip_horizontal,
                                              random_affine,
                                              normalize)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        return self.transforms(x)


class ValidationAugments(torch.nn.Module):
    def __init__(self,  *args, **kwargs) -> None:
        super().__init__()
        normalize = Normalize(mean=NORM_CONSTANTS[0], std=NORM_CONSTANTS[1])
        self.transforms = torch.nn.Sequential(normalize)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        return self.transforms(x)

# these are done in the dataset class
# I had errors when I left some of these out from the dataset class
def to_tensor_transform():
    train_transform = [T.ToTensor()]
    resize = T.Resize((224, 224))
    train_transform.append(resize)
    return T.Compose(train_transform)


class AbstractDataset(Dataset, metaclass=ABCMeta):

    def __init__(self, **kwargs):
        super().__init__()
        self.intitialize(**kwargs)
        self.kwargs = kwargs
        self._set_transforms()

    @abstractmethod
    def _initialize(self, **kwargs):
        """any"""

    @abstractmethod
    def _set_transforms(self):
        """any"""


class ImageDataset(AbstractDataset):

    def _initialize(self, **kwargs):
        self.csv = pd.read_csv(kwargs['csv'], index_col=0)
        self.targets = np.array() # pose
        self.images = [T.ToTensor()(Image.open(join(kwargs['root'], img))) for img in self.csv['image'].list()]
        self.distractors = self.csv[kwargs['distractors']].list()

    def __len__(self):
        return len(self.images)
        
    def _set_transforms(self):
        print("Yes I was set once, now you set me twice")
        transforms = [  # from Yale_B
            T.Normalize(mean=(0.5), std=(0.5)),
            T.Resize((224, 224))
        ]
        self.transforms = T.Compose(transforms)
        # TODO

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.transforms(self.imgs[index])
        target = torch.FloatTensor(self.targets[index])
        z = torch.FloatTensor(self.distractors[index])

        return image, target, z


class WildsDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size, num_workers, metadata, holdout_ratio,
                 transform_args: dict, data_name: str, seed: int, class_weights: Optional[list] = None,
                 **kwargs):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.holdout_ratio = holdout_ratio
        self.ds_kwargs=kwargs
        self.seed = seed
        self.train_transform = TrainingAugments(**transform_args)
        self.val_transform = ValidationAugments(**transform_args)

    def collate_fn(self, batch):
        batch = default_collate(batch)
        # print(batch[2][:, 0].shape)
        return batch[0], batch[1].float().unsqueeze(-1), batch[2][:, 0:1].float()  # cleanup metadata
    
    def get_circe_data(self):
        train_idx = self.train_data.indices
        targets = self.train_data.dataset._y_array[train_idx]
        distractors = self.train_data.dataset._metadata_array[train_idx, 0]
        assert all(targets == self.train_data.dataset._metadata_array[train_idx, 1])
        train, test = train_test_split(range(len(targets)), test_size=1 - self.holdout_ratio, random_state=self.seed)
        print("CIRCE USES TRAIN REGRESSION FOR TRAINING OF MODEL AS WELL FOR THIS DATASET")
        return torch.FloatTensor(targets[train].numpy().reshape(-1, 1)), torch.FloatTensor(distractors[train].numpy().reshape(-1, 1))

    def setup(self, stage: Optional[str] = None):
        all_data = get_dataset(self.dataset, **self.ds_kwargs)
        self.train_data = all_data.get_subset('train', transform=to_tensor_transform())
        self.eval_data = all_data.get_subset('val', transform=to_tensor_transform())
        self.test_data = all_data.get_subset('test', transform=to_tensor_transform())

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x = batch[0]
        if self.trainer.training:
            x = self.train_transform(x)
        else:
            x = self.val_transform(x)
        
        return x, batch[1], batch[2]

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
