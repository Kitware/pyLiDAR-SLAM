from typing import Optional, Dict, Any

from hydra.core.config_store import ConfigStore
from torch import nn as nn

# Hydra and OmegaConf
from omegaconf import OmegaConf
from hydra.conf import dataclass, MISSING

# Project Imports
from slam.common.pose import Pose
from slam.models.posenet import POSENET


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class PredictionConfig:
    """PoseNet Prediction Config"""
    num_input_channels: int = MISSING
    sequence_len: int = MISSING
    posenet_config: Optional[Dict[str, Any]] = None


# Hydra -- Create a group for the Prediction Config
cs = ConfigStore.instance()
cs.store(group="training/prediction", name="posenet", node=PredictionConfig)


# ----------------------------------------------------------------------------------------------------------------------
# POSENET PREDICTION MODULE
class _PoseNetPredictionModule(nn.Module):
    """
    Posenet Module
    """

    def __init__(self,
                 config: PredictionConfig,
                 pose: Pose):
        nn.Module.__init__(self)
        self.config = PredictionConfig(**config)
        self.pose = pose

        self.num_input_channels = self.config.num_input_channels
        self.sequence_len: int = self.config.sequence_len
        config.posenet_config["sequence_len"] = self.sequence_len
        config.posenet_config["num_input_channels"] = self.num_input_channels
        self.posenet: nn.Module = POSENET.load(OmegaConf.create(config.posenet_config), pose=self.pose)

    def forward(self, data_dict: dict):
        vertex_map = data_dict["vertex_map"]
        pose_params = self.posenet(vertex_map)[:, 0]
        data_dict["pose_params"] = pose_params
        data_dict["pose_matrix"] = self.pose.build_pose_matrix(pose_params)

        if "absolute_pose_gt" in data_dict:
            gt = data_dict["absolute_pose_gt"]
            relative_gt = gt[:, 0].inverse() @ gt[:, 1]
            data_dict["ground_truth"] = relative_gt

        return data_dict
