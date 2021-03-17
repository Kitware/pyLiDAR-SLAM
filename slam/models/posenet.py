import torch
import torch.nn as nn
from enum import Enum

# Hydra and OmegaConf
from omegaconf import DictConfig
from hydra.conf import ConfigStore

# Project Imports
from ._resnet import ResNetEncoder
from .layers import ACTIVATIONS
from slam.common.pose import Pose
from slam.common.utils import assert_debug, check_sizes


# ----------------------------------------------------------------------------------------------------------------------
# POSERESNET
# noinspection PyAbstractClass
class PoseResNet(nn.Module):
    """
    PoseResNet is a network regressing the 6 parameters of rigid transformation
    From a pair of images or a single image
    """

    def __init__(self,
                 config: DictConfig,
                 pose: Pose = Pose("euler")):
        nn.Module.__init__(self)
        self.config = config
        self.num_input_channels = self.config.num_input_channels
        self.sequence_len = self.config.sequence_len
        self.num_out_poses = self.config.get("num_out_poses", 1)
        model = self.config.get("resnet_model", 18)

        self.resnet_encoder = ResNetEncoder(self.num_input_channels * self.sequence_len,
                                            model=model,
                                            activation=self.config.get("activation", "relu"))
        self.activation_str = self.config.get("regression_activation", "relu")
        self.activation = ACTIVATIONS.get(self.activation_str)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._last_resnet_layer = 512
        self.pose = pose

        self.fc_rot = nn.Linear(self._last_resnet_layer, pose.num_rot_params() * self.num_out_poses, bias=False)
        self.fc_trans = nn.Linear(self._last_resnet_layer, 3 * self.num_out_poses)

        # Initialize Scale to allow stable training
        torch.nn.init.xavier_uniform_(self.fc_rot.weight, 0.01)
        torch.nn.init.xavier_uniform_(self.fc_trans.weight, 0.01)

    def forward(self, frames) -> torch.Tensor:
        if isinstance(frames, list):
            assert_debug(len(frames) == 1, "Multiple input not supported for current PoseNet version")
            frames = frames[0]
        check_sizes(frames, [-1, self.sequence_len, self.num_input_channels, -1, -1])
        features = self.resnet_encoder(frames.reshape(-1, self.num_input_channels * self.sequence_len,
                                                      frames.size(3),
                                                      frames.size(4)))
        x = self.avgpool(features).flatten(1)

        rot_params = 0.1 * self.fc_rot(x)  # scaling which allows stable training
        trans_params = self.fc_trans(x)

        pose_params = torch.cat([trans_params, rot_params], dim=-1)
        pose_params = pose_params.reshape(-1, self.num_out_poses, self.pose.num_params())
        return pose_params


# ----------------------------------------------------------------------------------------------------------------------
class POSENET(Enum):
    poseresnet = PoseResNet

    @staticmethod
    def load(config: DictConfig, pose: Pose = Pose("euler")):
        assert_debug("type" in config)
        assert_debug(config.type in POSENET.__members__)

        return POSENET.__members__[config.type].value(config, pose=pose)
