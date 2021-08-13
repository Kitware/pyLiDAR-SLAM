from abc import ABC
from enum import Enum
from typing import Union, Optional
import numpy as np

# Hydra and OmegaConf
from omegaconf import DictConfig
from hydra.conf import dataclass, MISSING

# Project Imports
from slam.common.utils import assert_debug, ObjectLoaderEnum


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
class LOOP_CLOSURE(ObjectLoaderEnum, Enum):
    none = (None, None)

    @classmethod
    def type_name(cls):
        return "type"
