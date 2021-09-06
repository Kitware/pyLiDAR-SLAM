import time
from abc import abstractmethod, ABC

import numpy as np

# Hydra and OmegaConf
from hydra.conf import dataclass, MISSING

# Project Imports
from slam.odometry.local_map import LocalMapConfig


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class OdometryConfig:
    """Abstract class which should be extended as config by childs of OdometryAlgorithm"""
    algorithm: str = MISSING
    local_map: LocalMapConfig = MISSING


# ----------------------------------------------------------------------------------------------------------------------
class OdometryAlgorithm(ABC):
    """Abstract class which acts as an interface for a Slam algorithm implemented in pytorch"""

    def __init__(self, config: OdometryConfig, **kwargs):
        self.config = config

        # Keeps track of the elapsed time between the processing of each frame
        self.elapsed: list = []

    @abstractmethod
    def init(self):
        """
        An initialization procedure called at the start of each sequence
        """
        self.elapsed = []

    def process_next_frame(self, data_dict: dict):
        """
        Computes the new pose and stores it in memory

        Args:
            data_dict (dict): The new frame (consisting of a dictionary of data items) returned by the Dataset
        """
        beginning = time.time()
        self.do_process_next_frame(data_dict)
        self.elapsed.append(time.time() - beginning)

    @abstractmethod
    def do_process_next_frame(self, data_dict: dict):
        """
        Computes the new pose and stores it in memory

        Args:
            data_dict (dict): The new frame's data as a dictionary returned by the Dataset
        """
        raise NotImplementedError("")

    def get_relative_poses(self) -> np.ndarray:
        """
        Returns the poses of the sequence of frames started after init() was called
        """
        raise NotImplementedError("")

    def get_elapsed(self) -> float:
        """
        Returns the total elapsed time in calling process_next_frame
        """
        return sum(self.elapsed)

    @staticmethod
    def pointcloud_key() -> str:
        return "odometry_pc"

    @staticmethod
    def relative_pose_key() -> str:
        """The key of the expected relative pose filled by the odometry at each frame

        The SLAM algorithm expects a [4, 4] numpy pose transform array under this key
        In the data_dict after do_process_next_frame is called
        """
        return "odometry_pose"
