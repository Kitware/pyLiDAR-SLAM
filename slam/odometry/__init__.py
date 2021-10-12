from abc import ABC, abstractmethod
from enum import Enum

# Hydra and OmegaConf
from omegaconf import OmegaConf, MISSING, DictConfig

# Project Import
from .icp_odometry import ICPFrameToModel, ICPFrameToModelConfig, assert_debug
from .posenet_odometry import PoseNetOdometry, PoseNetOdometryConfig
from .odometry import OdometryConfig
from slam.common.utils import ObjectLoaderEnum

# ----------------------------------------------------------------------------------------------------------------------
from .ct_icp_odometry import _with_ct_icp
from slam.common.utils import ObjectLoaderEnum

if _with_ct_icp:
    from .ct_icp_odometry import CT_ICPOdometryConfig, CT_ICPOdometry


# ----------------------------------------------------------------------------------------------------------------------

class ODOMETRY(ObjectLoaderEnum, Enum):
    """A Convenient Enum which allows to load the proper Odometry Algorithm"""
    icp_F2M = (ICPFrameToModel, ICPFrameToModelConfig)
    posenet = (PoseNetOdometry, PoseNetOdometryConfig)
    if _with_ct_icp:
        ct_icp = (CT_ICPOdometry, CT_ICPOdometryConfig)

    @classmethod
    def type_name(cls):
        return "algorithm"
