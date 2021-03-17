from enum import Enum

import torch
import torch.nn as nn

from slam.common.utils import assert_debug


class ACTIVATIONS(Enum):
    relu = nn.ReLU()
    gelu = nn.GELU()
    sinus = torch.sin
    sigmoid = nn.Sigmoid()
    softplus = nn.Softplus()

    @staticmethod
    def get(activation: str):
        assert_debug(activation in ACTIVATIONS.__members__, f"activation {activation} not implemented")
        return ACTIVATIONS.__members__[activation].value


class NORM_LAYERS(Enum):
    group = nn.GroupNorm
    batch2d = nn.BatchNorm2d
    instance2d = nn.InstanceNorm2d
    none = nn.Identity

    @staticmethod
    def get(norm_layer: str, num_groups: int = None, num_channels: int = None):
        assert_debug(norm_layer in NORM_LAYERS.__members__)
        pytorch_ctor = NORM_LAYERS.__members__[norm_layer].value
        if norm_layer == "group":
            assert_debug(num_groups is not None and num_channels is not None)
            return pytorch_ctor(num_channels=num_channels, num_groups=num_groups)
        elif norm_layer == "batch2d":
            assert_debug(num_channels is not None)
            return pytorch_ctor(num_channels)
        elif norm_layer == "instance2d":
            return pytorch_ctor(num_channels)
        elif norm_layer == "none":
            return pytorch_ctor()


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, activation: str = "relu"):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.activation = ACTIVATIONS.get(activation)

    def forward(self, x):
        return self.activation(self.fc(x))
