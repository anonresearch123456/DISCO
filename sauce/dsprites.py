'''
dsprites torch dataset
'''
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize
from torchvision.transforms.functional import affine

import pytorch_lightning as pl

from pathlib import Path

from sauce.utils.cdcor import bandwidth_selection


factor_map = {
    'color' : 0,
    'shape' : 1,
    'scale' : 2,
    'orientation' : 3,
    'position_x' : 4,
    'position_y' :  5
}


class Dsprites(Dataset):
    def __init__(self, path:str, noise:float, ood:bool=False,
                 holdout_ratio:float=0.01, use_holdout_to_train=False,
                 seed:int=42):
        data = np.load(path)
        self.images = np.asarray(data['imgs'], dtype=np.float32)
        self.latent_values = data['latents_values']
        self.latent_classes = data['latents_classes']
        self.holdout_ratio = holdout_ratio
        self.use_holdout_to_train = use_holdout_to_train
        self.nl_type = "tricky"
        self.seed = seed
        print('Holdout ration: {}'.format(holdout_ratio))

        target = "position_y"
        distractor = "position_x"

        if not ood:
            self.correlate(distractor, target)
        else:
            self.sample_ood(distractor, target)
        if self.holdout_ratio > 0:
            self._regress_YZ()
        else:
            print('Regression is NOT done!')
        
        self.set_noise(target, noise)
        self.normalize = Normalize(mean=(0.5), std=(0.5))

    def get_target_bandwidth(self):
        return bandwidth_selection(torch.FloatTensor(self.targets))

    def _regress_YZ(self):
        '''
        Create a held-out set and learn a linear regressor from Y to Z on it.
        '''
        print("Computing Y->Z residuals.")

        train, test = train_test_split(range(len(self.targets)), test_size=1 - self.holdout_ratio, random_state=self.seed)

        Y = self.targets[train].numpy().reshape(-1, 1)
        Z = self.distractors[train].numpy().reshape(-1, 1)
        self.linear_reg = linear_model.LinearRegression()
        self.linear_reg.fit(Y, Z)

        if self.use_holdout_to_train:
            print('\n\nHOLDOUT DATA WILL BE USED FOR TRAINING\n\n')
        else:
            self.targets = self.targets[test]
            self.distractors = self.distractors[test]
            self.images = self.images[test]

        self.targets_heldout = Y
        self.distractors_heldout = Z

        print('Train size: {}, Heldout size: {}'.format(self.targets.shape[0], Y.shape[0]))

    def __len__(self):
        return self.images.shape[0]

    def set_noise(self, target:str, noise:int):
        N = len(self.images)
        shift = 2*np.random.normal(loc=0, scale=noise, size=N)
        self.translate = np.zeros((N, 2))
        if target == 'position_x':
            self.translate[:, 0] = shift
        elif target == 'position_y':
            self.translate[:, 1] = shift

    def sample_ood(self, distractor:str, target:str):
        print("Sampling OOD data.")
        if distractor == 'shape' and target == 'position_x':
            self.targets = torch.FloatTensor(self.latent_classes[:, factor_map['position_x']]) / 31
            self.distractors = torch.FloatTensor(self.latent_classes[:, factor_map['shape']]) / 2
        elif distractor == 'position_x' and target == 'position_y':
            self.targets = torch.FloatTensor(self.latent_classes[:, factor_map['position_y']]) / 31
            self.distractors = torch.FloatTensor(self.latent_classes[:, factor_map['position_x']]) / 31
        elif distractor == 'color' and target == 'position_x':
            self.targets = torch.FloatTensor(self.latent_classes[:, factor_map['position_x']]) / 31
            colors = np.linspace(0, 1, 6)[1:]
            colors = np.random.choice(colors, len(self.images))
            self.images *= np.expand_dims(np.expand_dims(colors, -1), -1)
            self.distractors = torch.FloatTensor(self.images.reshape(len(self.images), -1).max(axis=1))

    def correlate_tricky(self, distractor: str, target: str):
        print("Sampling Z-Y correlated data.")
        N = self.latent_classes.shape[0]

        # y_ = y + noise
        # xi = N(0, 1)
        # z = y + xi
        # z_ = y + 0.1 xi^2

        if distractor == 'position_x' and target == 'position_y':
            y_pos = self.latent_classes[:, factor_map['position_y']] + 1
            x_pos = self.latent_classes[:, factor_map['position_x']] + 1
            print('y_pos min {} max {}'.format(y_pos.min(), y_pos.max()))
            print('x_pos min {} max {}'.format(x_pos.min(), x_pos.max()))
            indices = np.array([])
            noise_draws = np.array([])

            while len(indices) < N:
                for y in range(1, 33):
                    y_idx = np.where(y_pos == y)[0]
                    noise = 2 * (np.random.randint(0, 2, N) - 0.5) * np.sqrt(np.random.randint(0, 4, N))
                    x_idx = np.where(np.abs(x_pos - noise ** 2 - y) < 1)[0]
                    valid_idx = np.array(list(set(x_idx.tolist()).intersection(set(y_idx.tolist()))), dtype=int)
                    indices = np.concatenate([indices, valid_idx])
                    noise_draws = np.concatenate([noise_draws, noise[valid_idx]])

            p = np.random.permutation(len(indices))
            indices = indices[p][:N].astype(int)
            noise_draws = noise_draws[p][:N]
            self.images = self.images[indices]
            self.latent_classes = self.latent_classes[indices]
            self.latent_values = self.latent_values[indices]
            self.targets = torch.FloatTensor(self.latent_classes[:, factor_map['position_y']]) / 31
            self.distractors = torch.FloatTensor(noise_draws)

    def correlate(self, distractor: str, target: str):
        if self.nl_type == 'tricky':
            self.correlate_tricky(distractor, target)
        else:
            print("Sampling Z-Y correlated data.")
            N = self.latent_classes.shape[0]

            if distractor == 'position_x' and target == 'position_y':
                y_pos = self.latent_classes[:, factor_map['position_y']] + 1
                x_pos = self.latent_classes[:, factor_map['position_x']] + 1
                print('y_pos min {} max {}'.format(y_pos.min(), y_pos.max()))
                print('x_pos min {} max {}'.format(x_pos.min(), x_pos.max()))
                eps = np.abs(np.random.normal(loc=0, scale=1, size=N))
                indices = np.array([])

                for y in range(1, 33):
                    y_idx = np.where(y_pos == y)[0]
                    if self.nl_type == 'cone':
                        for tries in range(10):
                            eps = np.random.normal(loc=0, scale=7 * y / 32, size=1)
                            if eps > 0:
                                x1 = np.where(x_pos >= y)[0]
                                x2 = np.where(x_pos <= y + eps)[0]
                            else:
                                x1 = np.where(x_pos <= y)[0]
                                x2 = np.where(x_pos >= y + eps)[0]
                            x_idx = np.array(list(set(x1.tolist()).intersection(set(x2.tolist()))), dtype=int)
                            valid_idx = np.array(list(set(x_idx.tolist()).intersection(set(y_idx.tolist()))), dtype=int)
                            indices = np.concatenate([indices, valid_idx])
                    elif self.nl_type == 'y-cone':
                        for tries in range(10):
                            eps = 0.5 * 2 * (np.random.randint(0, 2, size=1) - 0.5) * y ** 2 / 32
                            x_idx = np.where(np.abs(x_pos - 0.5 * y - eps) < 1)[0]
                            valid_idx = np.array(list(set(x_idx.tolist()).intersection(set(y_idx.tolist()))), dtype=int)
                            indices = np.concatenate([indices, valid_idx])
                    else:
                        if self.nl_type == 'quadratic':
                            x1 = np.where(x_pos >= (y_pos)**2 / 32 - eps)[0]
                            x2 = np.where(x_pos <= (y_pos)**2 / 32 + eps)[0]
                        elif self.nl_type == 'quadratic-centered':
                            x1 = np.where(x_pos/32 >= 4*(y_pos/32 - 0.5)**2 - eps/64)[0]
                            x2 = np.where(x_pos/32 <= 4*(y_pos/32 - 0.5)**2 + eps/64)[0]
                        x_idx = np.array(list(set(x1.tolist()).intersection(set(x2.tolist()))))
                        valid_idx = np.array(list(set(x_idx.tolist()).intersection(set(y_idx.tolist()))))
                        indices = np.concatenate([indices, valid_idx])

                np.random.shuffle(indices)
                indices = np.random.choice(indices, N, replace=True)
                indices = np.asarray(indices, dtype=int)
                self.images = self.images[indices]
                self.latent_classes = self.latent_classes[indices]
                self.latent_values = self.latent_values[indices]
                self.targets = torch.FloatTensor(self.latent_classes[:, factor_map['position_y']]) / 31
                self.distractors = torch.FloatTensor(self.latent_classes[:, factor_map['position_x']]) / 31
            
    def __getitem__(self, index):
        image = np.expand_dims(self.images[index], -1).transpose(2, 0, 1)
        image = self.normalize(torch.FloatTensor(image))
        image = affine(image, angle=0, translate=tuple(self.translate[index]),
                         scale=1, shear=0, fill=-1.)

        return image, self.targets[index : index + 1], self.distractors[index: index + 1]


class DspritesDataModule(pl.LightningDataModule):
    def __init__(self, dataset:str, noise:float, num_workers: int = 8, ood:bool=False, holdout_ratio:float=0.01,
                 use_holdout_to_train=False, batch_size:int=32, seed:int = 42, **kwargs):
        super().__init__()
        self.path = dataset
        self.noise = noise
        self.ood = ood
        self.holdout_ratio = holdout_ratio
        self.use_holdout_to_train = use_holdout_to_train
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

    def get_target_bandwidth(self):
        return self.train_ds.get_target_bandwidth()
    
    def get_circe_data(self):
        return torch.FloatTensor(self.train_ds.targets_heldout), torch.FloatTensor(self.train_ds.distractors_heldout)

    def setup(self, stage=None):
        self.train_ds = Dsprites(Path(self.path) / "dsprites_train.npz", self.noise, self.ood, self.holdout_ratio, self.use_holdout_to_train, self.seed)
        self.val_ds = Dsprites(Path(self.path) / "dsprites_val.npz", self.noise, self.ood, 0.0, True)
        self.test_ds = Dsprites(Path(self.path) / "dsprites_test.npz", self.noise, True, 0.0, True)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)


if __name__ == '__main__':
    '''
    Split dsprites data into train-val-test sets.
    '''
    import numpy as np
    import sys
    from os.path import join, dirname
    from sklearn.model_selection import train_test_split


    path = sys.argv[1]
    data = np.load(path, encoding='latin1', allow_pickle=True)
    images = data['imgs']
    classes = data['latents_classes']
    values = data['latents_values']

    print("Splitting dsprites.")
    N = classes.shape[0]
    train, test = train_test_split(list(range(N)), test_size=0.2, random_state=42)
    val, test = train_test_split(test, test_size=0.5, random_state=42)
    train, val ,test = np.array(train), np.array(val), np.array(test)

    train_images = images[train]
    train_values = values[train]
    train_classes = classes[train]
    np.savez_compressed(join(dirname(path), 'dsprites_train'),
                        imgs=train_images,
                        latents_classes=train_classes,
                        latents_values=train_values)

    val_images = images[val]
    val_values = values[val]
    val_classes = classes[val]
    np.savez_compressed(join(dirname(path), 'dsprites_val'),
                        imgs=val_images,
                        latents_classes=val_classes,
                        latents_values=val_values)

    test_images = images[test]
    test_values = values[test]
    test_classes = classes[test]
    np.savez_compressed(join(dirname(path), 'dsprites_test'),
                        imgs=test_images,
                        latents_classes=test_classes,
                        latents_values=test_values)
