'''
Yale B Extended torch dataset
'''
from collections import defaultdict
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from os.path import join, basename, splitext, dirname
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import pytorch_lightning as pl

from sauce.utils.cdcor import bandwidth_selection

from kornia.augmentation import Resize, Normalize
from kornia.color import GrayscaleToRgb

from PIL import Image


class GeneralTransforms(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize((224, 224)),
            GrayscaleToRgb(),
            Normalize(mean=(0.5), std=(0.5))
        )

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = x / 255.0  # Convert to float and scale
        return self.transforms(x)


class YaleBExtended(Dataset):
    def __init__(self, path:str, noise:float, ood:bool, nl_type:str, holdout_ratio:float, use_holdout_to_train=True, seed: int = 42,
                 *args, **kwargs):
        ids = os.listdir(path)
        self.images = []
        self.holdout_ratio = holdout_ratio
        self.use_holdout_to_train = use_holdout_to_train
        self.seed=seed
        for id in ids:
            self.images.extend(glob(join(path, id, '*E*.pgm')))

        self._set_id_info(ids)
        self.all_images = np.array(self.images).copy()

        labels = [splitext(basename(x))[0].split('_')[1] for x in self.images]
        self.targets = np.array([int(x[1:3]) for x in labels]) # pose
        self.distractors = np.array([int(x[4:8]) for x in labels], dtype=float) # azimuth
        self.elevation = np.array([int(x[9:]) for x in labels], dtype=float)

        self.nl_type=nl_type
        if not ood:
            self.correlate()

        self.set_noise(path, noise)
        self.targets = self.targets / 9
        self.distractors = self.distractors / 130

        if self.holdout_ratio > 0:
            self._regress_YZ()

    def _regress_YZ(self):
        '''
        Create a held-out set and learn a linear regressor from Y to Z on it.
        '''
        print("Computing Y->Z residuals.")

        train, test = train_test_split(range(len(self.targets)), test_size=1 - self.holdout_ratio, random_state=self.seed)

        Y = self.targets[train].reshape(-1, 1)
        Z = self.distractors[train].reshape(-1, 1)
        self.linear_reg = linear_model.LinearRegression()
        self.linear_reg.fit(Y, Z)

        self.images = np.array(self.images)

        if self.use_holdout_to_train:
            print('\n\nHOLDOUT DATA WILL BE USED FOR TRAINING\n\n')
        else:
            self.targets = self.targets[test]
            self.distractors = self.distractors[test]
            self.images = self.images[test]

        self.targets_heldout = Y
        self.distractors_heldout = Z

        print('Train size: {}, Heldout size: {}'.format(self.targets.shape[0], Y.shape[0]))

    def _set_id_info(self, ids):
        self.id_info = {}
        for id in ids:
            self.id_info[id] = defaultdict(list)
        for i in range(self.__len__()):
            label = splitext(basename(self.images[i]))[0]
            id = label.split('_')[0]
            pose = int(label.split('_')[1][1:3])
            azimuth = int(label.split('_')[1][4:8])
            elevation = int(label.split('_')[1][9:])
            if pose not in self.id_info[id]['pose']:
                self.id_info[id]['pose'].append(pose)
            if azimuth not in self.id_info[id]['azimuth']:
                self.id_info[id]['azimuth'].append(azimuth)
            if elevation not in self.id_info[id]['elevation']:
                self.id_info[id]['elevation'].append(elevation)
        for id in ids:
            self.id_info[id]['pose'] = sorted(self.id_info[id]['pose'])
            self.id_info[id]['azimuth'] = sorted(self.id_info[id]['azimuth'])
            self.id_info[id]['elevation'] = sorted(self.id_info[id]['pose'])

    def set_noise(self, path, noise):
        poses = list(range(10))
        sorted_images = sorted(self.all_images)
        shift = np.random.normal(loc=0, scale=noise, size=self.__len__())
        for i in tqdm(range(self.__len__())):
            pose = self.targets[i]
            azimuth = int(self.distractors[i])
            az_sign = '+' if azimuth >= 0 else '-'
            elevation = int(self.elevation[i])
            el_sign = '+' if elevation >= 0 else '-'
            id = basename(dirname(self.images[i]))
            pose_idx = poses.index(pose)
            noisy_pose = int(pose_idx + shift[i])

            if noisy_pose < len(poses) and noisy_pose >= 0 and poses[noisy_pose] in self.id_info[id]['pose']:
                new_pose = poses[noisy_pose]
            else:
                new_pose = pose

            fname = '{}/{}_P{:02d}A{}{:03d}E{}{:02d}.pgm'.format(id, id, new_pose, az_sign, abs(azimuth), el_sign, abs(elevation))
            try:
                self.images[i] = sorted_images[sorted_images.index(join(path, fname))]
            except:
                print('noisy image not found')
                import pdb; pdb.set_trace()

    def correlate(self):
        indices = np.array([])
        unique_poses = np.unique(self.targets)

        while len(indices) < self.__len__():
            eps = np.abs(np.random.normal(loc=0, scale=1, size=self.__len__()))
            for y in unique_poses:
                y_idx = np.where(self.targets == y)[0]
                if self.nl_type == 'mod':
                    z1 = np.where(self.targets >= 0.5*np.abs(self.distractors/10) - eps/2)[0]
                    z2 = np.where(self.targets <= 0.5*np.abs(self.distractors/10) + eps/2)[0]
                    z_idx = np.array(list(set(z1.tolist()).intersection(set(z2.tolist()))))

                    valid_idx = np.array(list(set(z_idx.tolist()).intersection(set(y_idx.tolist()))))
                    indices = np.concatenate([indices, valid_idx])
                elif self.nl_type == 'y-cone':
                    az = 0.5*(self.distractors/130 + 1) * 9
                    for _ in range(10):
                        eps = 0.5 * 2 * (np.random.randint(0, 2, size=1) - 0.5) * y ** 2 / 9
                        z_idx = np.where(np.abs(az - 0.5 * y - eps) < 1)[0]
                        valid_idx = np.array(list(set(z_idx.tolist()).intersection(set(y_idx.tolist()))))
                        indices = np.concatenate([indices, valid_idx])
        p = np.random.permutation(len(indices))
        indices = indices[p][:self.__len__()].astype(np.int32)
        self.images = np.array(self.images)[indices]
        self.targets = self.targets[indices]
        self.distractors = self.distractors[indices]
        self.elevation = self.elevation[indices]

    def get_target_bandwidth(self):
        return bandwidth_selection(torch.FloatTensor(self.targets))

    def __len__(self):
        return len(self.images)

    def _read_image(self, index):
        img = Image.open(self.images[index])
        img = np.array(img)
        img = torch.FloatTensor(img)
        return img

    def __getitem__(self, index):
        image = self._read_image(index)
        x = image
        y = torch.FloatTensor(self.targets[index : index + 1])
        z = torch.FloatTensor(self.distractors[index : index + 1])
        return x, y, z


class YaleDataModule(pl.LightningDataModule):

    def __init__(self, dataset, batch_size, num_workers, ood:bool, metadata, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ds_kwargs=kwargs
        self.transformations = GeneralTransforms()
        self.ood = ood

    def get_circe_data(self):
        return torch.FloatTensor(self.train_data.targets_heldout), torch.FloatTensor(self.train_data.distractors_heldout)

    def get_target_bandwidth(self):
        return self.train_data.get_target_bandwidth()

    def setup(self, stage: Optional[str] = None):
        self.train_data = YaleBExtended(f'{self.dataset}/train', ood=self.ood, **self.ds_kwargs)
        self.eval_data = YaleBExtended(f'{self.dataset}/val', ood=self.ood, **self.ds_kwargs)
        self.test_data = YaleBExtended(f'{self.dataset}/test', ood=True, **self.ds_kwargs)

    def on_after_batch_transfer(self, batch, batch_idx):
        x = batch[0]
        x = self.transformations(x)        
        return x, batch[1], batch[2]

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

if __name__ == '__main__':
    '''
    Split Extended Yale-B data into train-val-test sets.
    '''
    import os
    import shutil
    import numpy as np
    from os.path import join
    from sklearn.model_selection import train_test_split

    path = './ExtendedYaleB'
    ids = np.array(os.listdir(path))

    print("Splitting Extended Yale-B.")
    train, val = train_test_split(range(len(ids)), test_size=0.2, random_state=42)
    val, test = train_test_split(val, test_size=0.5, random_state=42)

    os.makedirs(join(path, 'train'), exist_ok=True)
    for folder in ids[train]:
        shutil.move(join(path, folder), join(path, 'train'))

    os.makedirs(join(path, 'val'), exist_ok=True)
    for folder in ids[val]:
        shutil.move(join(path, folder), join(path, 'val'))

    os.makedirs(join(path, 'test'), exist_ok=True)
    for folder in ids[test]:
        shutil.move(join(path, folder), join(path, 'test'))
