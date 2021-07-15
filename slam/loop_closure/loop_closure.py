from abc import ABC
from enum import Enum
from typing import Union, Optional
import numpy as np

# Hydra and OmegaConf
from omegaconf import DictConfig
from hydra.conf import dataclass, MISSING

# Project Imports
from slam.common.utils import assert_debug


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class LoopClosureConfig:
    """Configuration for a LoopClosure Algorithm"""
    type: str = MISSING


class LoopClosure(ABC):
    """An abstract class for a LoopClosure Algorithm

    The Loop Closure Algorithm searches the data_dict for KeyFrames
    Which are typically aggregated PointCloud from Odometry Algorithms
    And Return optionally some constraints on the trajectory

    Loop Closure Algorithms typically need to store a lot of information
    """

    def __init__(self, config: LoopClosureConfig, **kwargs):
        self.config = config

    def init(self):
        """Cleans and Initializes the Loop Closure Algorithm"""
        self.clean()

    def clean(self):
        """Delete all previous data of the LoopClosure"""
        raise NotImplementedError("")

    def process_next_frame(self, data_dict: dict):
        raise NotImplementedError("")

    def update_positions(self, trajectory: np.ndarray):
        """Updates trajectory

        Args:
            trajectory (np.ndarray): The absolute poses making the trajectory `(N, 4, 4)`
        """
        pass

    @staticmethod
    def pointcloud_key() -> str:
        """Returns the key in the keyframe dict for a new pointcloud"""
        return "lc_pointcloud"

    @staticmethod
    def relative_pose_key() -> str:
        """Returns the key in the keyframe dict for a new pose"""
        return "lc_relative_pose"


# ----------------------------------------------------------------------------------------------------------------------
class LOOP_CLOSURE(Enum):
    none = (None, None)

    @staticmethod
    def load(config: Union[dict, DictConfig, LoopClosureConfig], **kwargs) -> Optional[LoopClosure]:
        if config is None:
            return None

        if isinstance(config, dict):
            config = DictConfig(config)
        elif isinstance(config, DictConfig) or isinstance(config, LoopClosureConfig):
            config = config
        else:
            raise NotImplementedError("Loop Closure is not Implemented")

        _type = config.type
        assert_debug(_type in LOOP_CLOSURE.__members__,
                     f"The LoopClosure type {_type} is not available or not implemented")

        _class, _config = LOOP_CLOSURE[_type].value

        if isinstance(config, LoopClosureConfig):
            return _class(config, **kwargs)
        else:
            return _class(_config(**config), **kwargs)
