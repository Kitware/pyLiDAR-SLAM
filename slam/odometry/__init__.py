from enum import Enum

# Hydra and OmegaConf
from omegaconf import OmegaConf

# Project Import
from .icp_odometry import ICPFrameToModel, ICPFrameToModelConfig, assert_debug
from .posenet_odometry import PoseNetOdometry, PoseNetOdometryConfig
from .odometry import OdometryConfig


# ----------------------------------------------------------------------------------------------------------------------
class ODOMETRY(Enum):
    """A Convenient Enum which allows to load the proper Odometry Algorithm"""

    icp_F2M = (ICPFrameToModel, ICPFrameToModelConfig)
    posenet = (PoseNetOdometry, PoseNetOdometryConfig)

    @staticmethod
    def load(config: OdometryConfig, **kwargs):
        assert_debug("algorithm" in config, "The config does not contains the key : 'algorithm'")
        algorithm = config.algorithm
        assert_debug(algorithm in ODOMETRY.__members__, f"Unknown algorithm {algorithm}")

        _class, _config = ODOMETRY.__members__[algorithm].value

        return _class(_config(**config), **kwargs)
