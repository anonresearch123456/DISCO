from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence
import torch
import torch.nn as nn
from torchvision import models
from sauce.blocks import ConvBnReLU, ResBlock, ConvBlock, ThreeDConvBlock, ThreeDResBlock, ThreeDConvBnReLU
import torchvision

BACKBONE_MAPPING = {
    'input': ConvBnReLU,
    'convnet': ConvBlock,
    'resnet': ResBlock,
    'pool': nn.MaxPool2d,
    'gap': nn.AdaptiveAvgPool2d,
}

ThreeDBACKBONE_MAPPING = {
    'convnet': ThreeDConvBlock,
    'resnet': ThreeDResBlock,
    'input': ThreeDConvBnReLU,
    'pool': nn.MaxPool3d,
    'gap': nn.AdaptiveAvgPool3d,
}


class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Load a pretrained ResNet-50 model
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.resnet50 = models.resnet50(weights=weights)
        # Remove the avgpool and fc layer
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.n_filters_out = 2048

    def forward(self, x):
        x = self.resnet50(x)
        x = self.gap(x)
        return [x.squeeze()]
    
    def get_n_filters_out(self):
        return self.n_filters_out


class CustomBackbone(nn.Module):
    def __init__(
        self,
        in_channels,
        n_basefilters,
        n_blocks,
        block_type,
        layer_size,
        threed=False,
    ):
        if threed:
            assert block_type in ThreeDBACKBONE_MAPPING.keys(), f'block_type must be one of {list(ThreeDBACKBONE_MAPPING.keys())}'
            block_type = ThreeDBACKBONE_MAPPING[block_type]
            input_block = ThreeDBACKBONE_MAPPING['input']
            pooler =ThreeDBACKBONE_MAPPING['pool']
            gap = ThreeDBACKBONE_MAPPING['gap']
        else:
            assert block_type in BACKBONE_MAPPING.keys(), f'block_type must be one of {list(BACKBONE_MAPPING.keys())}'
            block_type = BACKBONE_MAPPING[block_type]
            input_block = BACKBONE_MAPPING['input']
            pooler = BACKBONE_MAPPING['pool']
            gap = BACKBONE_MAPPING['gap']
        if n_blocks < 2:
            raise ValueError(f'n_blocks must be at least 2, but got {n_blocks}')
        super().__init__()
        layers = [
            ('conv1', input_block(in_channels, n_basefilters)),
            ('pool1', pooler(3, stride=2)),
            ('block1_0', block_type(n_basefilters, n_basefilters))
        ]
        for i in range(1, layer_size):
            layers.append(
                (f'block1_{i}', block_type(n_basefilters, n_basefilters))
            )
        n_filters = n_basefilters
        for i in range(n_blocks-1):
            layers.append(
                (f'block{i+2}_0', block_type(n_filters, 2 * n_filters, stride=2))
            )
            for j in range(1, layer_size):
                layers.append(
                    (f'block{i+2}_{j}', block_type(2 * n_filters, 2 * n_filters))
                )
            n_filters = 2 * n_filters
        self.n_filters_out = n_filters
        self.blocks = nn.Sequential(OrderedDict(layers))
        self.gap = gap(1)

    def forward(self, x):
        out = x
        feature_maps = []
        # use GAP for all intermediate layers for potential debiasing (debatable)
        for i, layer in enumerate(self.blocks):
            out = layer(out)
            feature_maps.append(self.gap(out).squeeze())
        return feature_maps

    def get_n_filters_out(self):
        return self.n_filters_out


class Network(nn.Module):
    def __init__(
        self,
        n_outputs,
        network_name: str,
        **kwargs,
    ):
        super().__init__()
        self.backbone: CustomBackbone | ResNetBackbone
        if network_name == "resnet-50-pretrained":
            self.backbone = ResNetBackbone()
            self.n_filters_out = 2048
        elif network_name == "custom":
            self.backbone = CustomBackbone(**kwargs)
        else:
            raise ValueError(f'network_name must be one of ["resnet-50-pretrained", "custom"], but got {network_name}')
        n_out = 1 if (n_outputs is None) or (n_outputs <= 2) else n_outputs
        # used by adversarial
        self.n_filters_out = self.backbone.get_n_filters_out()
        self.fc = nn.Linear(self.backbone.get_n_filters_out(), n_out)

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_clf_params(self):
        return self.fc.parameters()

    def forward(self, x, y=None):
        out: List[torch.Tensor] = self.backbone(x)
        out.append(self.fc(out[-1]))
        return out


class YaleBNetwork(nn.Module):
    def __init__(self, n_outputs, **kwargs):
        super().__init__()
        self.featurizer = torch.nn.Sequential(*(list(
            torchvision.models.resnet18(pretrained=True).children())[:-1]
        ))
        n_out = 1 if (n_outputs is None) or (n_outputs <= 2) else n_outputs
        self.n_filters_out = 512
        self.fc = nn.Sequential(
            nn.Linear(self.n_filters_out, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.Linear(64, n_out),
        )
        # self.gap = nn.AdaptiveAvgPool2d(1)

    def get_backbone_params(self):
        return self.featurizer.parameters()

    def get_clf_params(self):
        return self.fc.parameters()

    def forward(self, x, y=None):
        out = x
        feature_maps = []
        for i, layer in enumerate(self.featurizer):
            out = layer(out)
        out = torch.flatten(out, start_dim=1)
        feature_maps.append(out)  # only use the layers after the encoder to debias
        for i, layer in enumerate(self.fc):
            out = layer(out)
            feature_maps.append(out)
        return feature_maps


class SimulationNetwork(nn.Module):
    def __init__(self, n_outputs, in_channels,
                 n_basefilters,
                 n_blocks,
                 block_type,
                 layer_size,
                 threed=True, **kwargs):
        super().__init__()
        self.featurizer = CustomBackbone(in_channels, n_basefilters, n_blocks, block_type, layer_size, threed=threed)
        n_out = 1 if (n_outputs is None) or (n_outputs <= 2) else n_outputs
        self.n_filters_out = self.featurizer.get_n_filters_out()
        self.fc = nn.Sequential(
            nn.Linear(self.n_filters_out, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.Linear(64, n_out),
        )

    def get_backbone_params(self):
        return self.featurizer.parameters()

    def get_clf_params(self):
        return self.fc.parameters()

    def forward(self, x, y=None):
        out = x
        feature_maps = self.featurizer(out)
        out = feature_maps[-1]
        out = torch.flatten(out, start_dim=1)
        feature_maps.append(out)  # only use the layers after the encoder to debias
        for i, layer in enumerate(self.fc):
            out = layer(out)
            feature_maps.append(out)
        return feature_maps


class DspritesNetwork(nn.Module):
    def __init__(self, n_outputs=None, **kwargs):
        super().__init__()
        
        # Using a custom featurizer based on the original configuration
        self.featurizer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Adjust output based on the n_outputs parameter
        n_out = 1 if (n_outputs is None) or (n_outputs <= 2) else n_outputs
        
        # Custom FC layers based on the original configuration
        self.fc = nn.Sequential(
            nn.Linear(256, 128),  # Assuming the output dimension of the featurizer to be 64
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.Linear(64, n_out)
        )

    def get_backbone_params(self):
        return self.featurizer.parameters()

    def get_clf_params(self):
        return self.fc.parameters()

    def forward(self, x, y=None):
        out = x
        feature_maps = []
        for i, layer in enumerate(self.featurizer):
            out = layer(out)
            feature_maps.append(out)  # Collect feature maps for visualization or further processing
        
        # Flatten the output of the last Conv layer
        out = torch.flatten(out, start_dim=1)
        
        # Apply the fully connected layers
        for i, layer in enumerate(self.fc):
            out = layer(out)
            feature_maps.append(out)
        
        return feature_maps
