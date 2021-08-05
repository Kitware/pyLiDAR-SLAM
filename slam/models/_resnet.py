import torch.nn as nn
import torchvision.models.resnet as models

from slam.common.utils import assert_debug, check_tensor
from slam.models.layers import ACTIVATIONS


class CustomBasicBlock(models.BasicBlock):
    """
    ResNet basic block where the ReLU activation is replaced by a Custom activation
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None, activation: str = "relu"):
        super().__init__(inplanes, planes, stride, downsample)
        self.relu = ACTIVATIONS.get(activation)

    def forward(self, x):
        return super().forward(x)


class CustomBottleneck(models.Bottleneck):
    """
    ResNet Bottleneck block where the ReLU activation is replaced by a Custom activation
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None, activation: str = "relu"):
        super().__init__(inplanes, planes, stride, downsample)
        self.relu = ACTIVATIONS.get(activation)

    def forward(self, x):
        return super().forward(x)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


class _ResNetEncoder(nn.Module):
    """
    Constructs a resnet encoder with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, num_input_channels: int, block, layers, zero_init_residual=False, activation: str = "relu"):
        super().__init__()
        self.layers = layers
        self.planes = [num_input_channels, 64, 64, 128, 256, 512]

        self.inplanes = 64
        self.dilation = 1
        self.num_input_channels = num_input_channels
        self.conv1 = nn.Conv2d(num_input_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.relu = ACTIVATIONS.get(activation)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, CustomBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, CustomBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv1x1(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_layers(self, x):
        # See note [TorchScript super()]
        check_tensor(x, [-1, self.num_input_channels, -1, -1])
        x = self.conv1(x)
        x0 = self.relu(x)

        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x0, x1, x2, x3, x4

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.forward_layers(x)
        return x4


class ResNetEncoder(_ResNetEncoder):
    """
    A ResNet Encoder consist of the 4 layers of a ResNet which can be returned and decoded
    """

    def __init__(self, num_input_channels, model: int = 18, activation: str = "relu", pretrained: bool = False):
        model_to_params = {
            18: ([2, 2, 2, 2], CustomBasicBlock),
            34: ([3, 4, 6, 3], CustomBasicBlock),
            50: ([3, 4, 6, 3], CustomBottleneck)
        }
        assert_debug(model in model_to_params)

        super().__init__(num_input_channels,
                         model_to_params[model][1],
                         model_to_params[model][0],
                         activation=activation)

        # Load pretrained model
