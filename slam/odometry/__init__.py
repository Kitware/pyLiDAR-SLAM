from enum import Enum

# Hydra and OmegaConf
from omegaconf import OmegaConf

# Project Import
from .icp_odometry import ICPFrameToModel, ICPFrameToModelConfig, assert_debug
from .posenet_odometry import PoseNetOdometry, PoseNetOdometryConfig
from .odometry import OdometryConfig
from slam.common.utils import ObjectLoaderEnum


# ----------------------------------------------------------------------------------------------------------------------


class ODOMETRY(ObjectLoaderEnum, Enum):
    """A Convenient Enum which allows to load the proper Odometry Algorithm"""
    icp_F2M = (ICPFrameToModel, ICPFrameToModelConfig)
    posenet = (PoseNetOdometry, PoseNetOdometryConfig)

    @classmethod
    def type_name(cls):
        return "algorithm"
