from typing import Optional, Tuple

from torch.utils.data import DataLoader

import numpy as np
import torch
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from sauce.utils.cdcor import bandwidth_selection


def create_batch_rotated_ellipsoids(size, radii_ratio: torch.Tensor, scale_factor: torch.Tensor,
                                    centers:torch.Tensor, rotation_angles: torch.Tensor, device='cuda'):
    # Create a 3D grid of coordinates
    x = torch.linspace(0, size-1, size, device=device)
    y = torch.linspace(0, size-1, size, device=device)
    z = torch.linspace(0, size-1, size, device=device)
    x, y, z = torch.meshgrid(x, y, z, indexing='ij')

    # expand to match the batch size
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    z = z.unsqueeze(0)
    
    # Convert input parameters to tensors and add batch dimension
    radii = radii_ratio * scale_factor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    centers = centers.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    alpha, beta, gamma = rotation_angles[:, 0], rotation_angles[:, 1], rotation_angles[:, 2]

    # Rotation matrices for each angle in the batch
    cos_alpha, sin_alpha = torch.cos(alpha), torch.sin(alpha)
    cos_beta, sin_beta = torch.cos(beta), torch.sin(beta)
    cos_gamma, sin_gamma = torch.cos(gamma), torch.sin(gamma)

    Rx = torch.stack([torch.ones_like(cos_alpha, device=device), torch.zeros_like(cos_alpha,device=device), torch.zeros_like(cos_alpha,device=device),
                      torch.zeros_like(cos_alpha,device=device), cos_alpha, -sin_alpha,
                      torch.zeros_like(cos_alpha,device=device), sin_alpha, cos_alpha], dim=-1).view(-1, 3, 3)
    Ry = torch.stack([cos_beta, torch.zeros_like(cos_beta,device=device), sin_beta,
                      torch.zeros_like(cos_beta,device=device), torch.ones_like(cos_beta,device=device), torch.zeros_like(cos_beta,device=device),
                      -sin_beta, torch.zeros_like(cos_beta,device=device), cos_beta], dim=-1).view(-1, 3, 3)
    Rz = torch.stack([cos_gamma, -sin_gamma, torch.zeros_like(cos_gamma,device=device),
                      sin_gamma, cos_gamma, torch.zeros_like(cos_gamma,device=device),
                      torch.zeros_like(cos_gamma,device=device), torch.zeros_like(cos_gamma,device=device), torch.ones_like(cos_gamma,device=device)], dim=-1).view(-1, 3, 3)
    R = torch.bmm(torch.bmm(Rz, Ry), Rx)

    # Adjust coordinates and apply rotation
    xyz = torch.stack([x - centers[:, 0], y - centers[:, 1], z - centers[:, 2]], dim=-1)
    rotated_xyz = torch.einsum('bij, bxyzj -> bxyzi', R, xyz)

    # Ellipsoid equation for each ellipsoid in the batch
    ellipsoid = ((rotated_xyz[..., 0]/radii[..., 0])**2 + 
                 (rotated_xyz[..., 1]/radii[..., 1])**2 + 
                 (rotated_xyz[..., 2]/radii[..., 2])**2) <= 1

    return ellipsoid.float()


def generate_3d_data(dataset_size: int = 1024, image_size: int = 32, scale_bias:bool = True,
                     center_bias:bool = True, brightness_bias:bool = True,
                      device='cuda'):
    ## Simulate Data
    N = dataset_size

    # sample y from normal distribution centered at 0 with std dev pi/3
    y = torch.normal(0, torch.pi/4, size=(N,), device=device)

    # sample N scale factor influenced by y
    if scale_bias:
        scale_factor = 4 + 1.0 * torch.sin(y) + torch.rand((N,), device=device) * 1.5
    else:
        # random scale factor
        scale_factor = torch.rand((N,), device=device) * 4 + 3

    # sample (N,3) centers of ellipsoids with sample bias on y. Should be dispersed around the center of the 32x32x32 volume
    # the center of the volume is (16, 16, 16)
    if center_bias:
        centers = (16 + 1.5 * torch.sin(y).unsqueeze(-1) * torch.ones((N, 3), device=device)
                + torch.randn((N, 3), device=device) * 1.1)
    else:
        centers = (torch.tensor([16, 16, 16], device=device).unsqueeze(0).repeat(N, 1)
                + torch.randn((N, 3), device=device) * 1.5)

    # random rotation angles for X and Z (0.3, 0,3 originally, with pi/8 std dev)
    rotation_angle_x = torch.normal(0.35, torch.pi / 16.0, size=(N,), device=device)
    rotation_angle_z = torch.normal(-0.35, torch.pi / 16.0, size=(N,), device=device)
    rotation_angle_y = torch.normal(0., torch.pi / 8.0, size=(N,), device=device)

    # rotation_angle_x = torch.tensor([torch.pi / 3.0], device=device).repeat(N)
    # rotation_angle_z = torch.tensor([torch.pi / 3.0], device=device).repeat(N)

    # Use y as rotation_angle_y, already influenced by y
    rotation_angles = torch.stack([y, y, y], dim=-1)

    # brightness of the ellipsoids
    if brightness_bias:    
        brightness = torch.exp(torch.sin(y)*1.25) + torch.exp(torch.randn((N,), device=device) * 0.5) + 1.0
    else:
        # random brightness
        brightness = torch.rand((N,), device=device) * 2.0 + 2.0

    brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # fixed radii ratios
    radii_ratios = torch.tensor([1, 2.5, 0.5], device=device)

    ellipsoids = create_batch_rotated_ellipsoids(image_size, radii_ratios, scale_factor, centers, rotation_angles, device=device)

    # due to memory we will do it in a loop
    for i in range(1, N):
        ellipsoids[i] = ellipsoids[i] * brightness[i] + torch.randn_like(ellipsoids[i], device=device) * 0.1

    # stack centers, scale factors and brightness on axis one, s.t. the shape is (N, 3+1+1)
    bias = torch.concat([centers, scale_factor.reshape(-1, 1), brightness.reshape(-1, 1)], dim=1)

    # reshape ellipsoids to (N, 1, 32, 32, 32)
    ellipsoids = ellipsoids.unsqueeze(1)

    return (ellipsoids.cpu().numpy(),
            y.cpu().numpy(), bias.cpu().numpy(), N, centers.cpu().numpy(),
            scale_factor.squeeze().cpu().numpy(), brightness.squeeze().cpu().numpy())


class MultiBiasSimulationDataset3D(torch.utils.data.Dataset):
    def __init__(self, dataset_size: int, image_size: int,
                 use_holdout_to_train: bool, holdout_ratio: float, seed: int = 42,
                 scale_bias:bool = True, center_bias:bool = True, brightness_bias:bool = True,
                 *args, **kwargs):
        self.use_holdout_to_train = use_holdout_to_train
        self.holdout_ratio = holdout_ratio
        (self.x, self.y, self.bias, self.N, self.centers,
         self.scale_factors, self.brightness) = generate_3d_data(dataset_size, image_size, scale_bias, center_bias, brightness_bias)
        print(f"Dataset size: {len(self.x)}")
        if holdout_ratio:
            self.loo_split(seed)
        else:
            print("no LOO split")

    def loo_split(self, seed):
        '''
        Create a held-out set and learn a linear regressor from Y to Z on it.
        '''
        train, test = train_test_split(range(len(self.y)), test_size=1 - self.holdout_ratio, random_state=seed)

        Y = self.y[train]
        Z = self.bias[train]

        assert Y.ndim <= 2 and Z.ndim <= 2

        Y = Y if Y.ndim == 2 else Y.reshape(-1, 1)
        Z = Z if Z.ndim == 2 else Z.reshape(-1, 1)

        self.x = np.array(self.x)

        if self.use_holdout_to_train:
            print('\n\nHOLDOUT DATA WILL BE USED FOR TRAINING\n\n')
        else:
            self.y = self.y[test]
            self.bias = self.bias[test]
            self.x = self.x[test]

        self.y_heldout = Y
        self.z_heldout = Z

        print('Train size: {}, Heldout size: {}'.format(self.y.shape[0], Y.shape[0]))

    def get_target_bandwidth(self):
        return bandwidth_selection(torch.FloatTensor(self.y))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float)
        y = torch.tensor(self.y[idx], dtype=torch.float)
        bias = torch.tensor(self.bias[idx], dtype=torch.float)
        return x, y.unsqueeze(-1), bias


class MultiBiasSimulationDataModule3D(pl.LightningDataModule):
    def __init__(self, batch_size: int, dataset_size: int,
                 use_holdout_to_train: bool, holdout_ratio: float,
                 image_size: int, sigma: float,
                 *args, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.image_size = image_size
        self.sigma = sigma
        self.use_holdout_to_train = use_holdout_to_train
        self.holdout_ratio = holdout_ratio
        self.seed = kwargs['seed']

    def get_circe_data(self):
        return torch.FloatTensor(self.train_dataset.y_heldout), torch.FloatTensor(self.train_dataset.z_heldout)
    
    def get_target_bandwidth(self):
        return self.train_dataset.get_target_bandwidth()

    def setup(self, stage=None):
        # if stage == "fit" or stage is None:
        scale_bias = True
        center_bias = True
        brightness_bias = True
        self.train_dataset = MultiBiasSimulationDataset3D(dataset_size=self.dataset_size, image_size=self.image_size, sigma=self.sigma,
                                                       use_holdout_to_train=self.use_holdout_to_train, holdout_ratio=self.holdout_ratio, seed=self.seed,
                                                       scale_bias=scale_bias, center_bias=center_bias, brightness_bias=brightness_bias)
        self.val_dataset = MultiBiasSimulationDataset3D(dataset_size=self.dataset_size//4, image_size=self.image_size, sigma=self.sigma,
                                                     use_holdout_to_train=self.use_holdout_to_train, holdout_ratio=0,
                                                     scale_bias=scale_bias, center_bias=center_bias, brightness_bias=brightness_bias)
        # if stage == "test" or stage is None:
        self.test_dataset = MultiBiasSimulationDataset3D(dataset_size=self.dataset_size//4, image_size=self.image_size, sigma=self.sigma,
                                                      use_holdout_to_train=self.use_holdout_to_train, holdout_ratio=0,
                                                      scale_bias=False, center_bias=False, brightness_bias=False)
        torch.cuda.empty_cache()

    # shuffle all since they are created being ordered by target/bias attribute; this will make the computation of HSIC, Dcor etc. invalid
    # for test and valid: we use cdcor to track the development of cond. dependency;
    # doing shuffling here also ensures that the metric is computed correctly
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
