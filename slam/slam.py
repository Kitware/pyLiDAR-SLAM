from abc import abstractmethod
import time
from pathlib import Path
from typing import Optional
import logging

from scipy.spatial.transform import Rotation
import numpy as np

# Hydra and omegaconf
from hydra.conf import dataclass

# Project Imports
from slam.backend.backend import Backend, BackendConfig, BACKEND
from slam.common.utils import assert_debug
from slam.eval.eval_odometry import compute_absolute_poses
from slam.loop_closure.loop_closure import LoopClosure, LoopClosureConfig, LOOP_CLOSURE
from slam.odometry import ODOMETRY
from slam.odometry.odometry import OdometryAlgorithm, OdometryConfig


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class SLAMConfig:
    odometry: Optional[OdometryConfig] = None
    loop_closure: Optional[LoopClosureConfig] = None
    backend: Optional[BackendConfig] = None


class SLAM:
    """A SLAM Algorithm for Point Cloud data (typically LiDAR Data)

    A SLAM of pylidar-slam consists of three modules
        - Odometry:         The Scan Matching algorithm which iteratively estimate the trajectory
                            And produces frame-to-frame trajectory constraints i -> (i+1)
                            Required

        - Loop Closure:     A Loop Closure module constructs constraints between
                            Distant poses in the trajectory i -> j (such that i < j)
                            (Optional)

        - Backend:          The Backend estimate an optimal trajectory given the different constraints
                            (Optional)
    """

    def __init__(self, config: SLAMConfig, **kwargs):

        self.config = config

        # TODO -- Separate Processes for loop_closure and backend
        self.odometry: Optional[OdometryAlgorithm] = None
        self.loop_closure: Optional[LoopClosure] = None
        self.backend: Optional[Backend] = None
        self._frame_idx: int = 0

        # Keep track of time spent by each step
        self.elapsed_backend = []
        self.elapsed_loop_closure = []
        self.elapsed_odometry = []

        self.__kwargs = kwargs

    @abstractmethod
    def init(self):
        """
        An initialization procedure called at the start of each sequence
        """
        self._frame_idx = 0
        if self.odometry is None:
            assert self.config.odometry is not None
            self.odometry = ODOMETRY.load(self.config.odometry, **self.__kwargs)

        assert self.odometry is not None
        self.odometry.init()
        if self.loop_closure is None and self.config.loop_closure is not None:
            self.loop_closure = LOOP_CLOSURE.load(self.config.loop_closure, **self.__kwargs)
        if self.loop_closure is not None:
            self.loop_closure.init()
            if self.config.backend is not None:
                self.backend = BACKEND.load(self.config.backend, **self.__kwargs)
            if self.backend is not None:
                self.backend.init()
            else:
                logging.warning("[SLAMAlgorithm]Defined a Loop Closure Algorithm Without a Backend")

    def process_next_frame(self, data_dict: dict):
        """
        Args:
            data_dict (dict): The new frame (consisting of a dictionary of data items) returned by the Dataset
        """
        beginning = time.time()
        self.odometry.process_next_frame(data_dict)
        step_odometry = time.time()
        self.elapsed_loop_closure.append(step_odometry - beginning)

        odometry_pose = None
        if self.odometry.relative_pose_key() in data_dict:
            odometry_pose = data_dict[self.odometry.relative_pose_key()]

            # Convert to double and reproject to the manifold of Rotation matrices to minimize error cumulation
            odometry_pose = odometry_pose.astype(np.float64)
            odometry_pose[:3, :3] = Rotation.from_matrix(odometry_pose[:3, :3]).as_matrix()

        if self.loop_closure is not None:
            # Copy the variables for the appropriate names
            if odometry_pose is not None:
                data_dict[self.loop_closure.relative_pose_key()] = odometry_pose

            if self.odometry.pointcloud_key() in data_dict:
                data_dict[self.loop_closure.pointcloud_key()] = data_dict[self.odometry.pointcloud_key()]

            self.loop_closure.process_next_frame(data_dict)
            step_loop_closure = time.time()
            self.elapsed_loop_closure.append(step_loop_closure - step_odometry)

        if self.backend is not None:
            if odometry_pose is not None:
                measurement = (odometry_pose, None)
                data_dict[self.backend.se3_odometry_constraint(self._frame_idx - 1)] = measurement
            init_step = time.time()
            self.backend.next_frame(data_dict)
            step_backend = time.time()

            self.elapsed_backend.append(step_backend - init_step)

        self._frame_idx += 1

    def get_relative_poses(self):
        """Returns the computed relative poses along the trajectory"""
        if self.backend is not None:
            return self.backend.relative_odometry_poses()
        return self.odometry.get_relative_poses()

    def get_absolute_poses(self):
        """Returns the computed relative poses along the trajectory"""
        if self.backend is not None:
            return self.backend.absolute_poses()
        return compute_absolute_poses(self.odometry.get_relative_poses())

    def dump_all_constraints(self, log_dir: str):
        """Save the odometry, loop and absolute constraints on disk"""
        if self.backend is None:
            return

        dir_path = Path(log_dir)
        if not dir_path.exists():
            dir_path.mkdir()
        assert_debug(dir_path.exists())

        # Log Odometry Constraints
        self.save_constraints([(constraint[0], constraint[0] + 1, constraint[1]) for constraint in
                               self.backend.registered_odometry_constraints()],
                              str(dir_path / "odometry_constraints.txt"))

        self.save_constraints([(constraint[0], constraint[0], constraint[1]) for constraint in
                               self.backend.registered_absolute_constraints()],
                              str(dir_path / "absolute_constraints.txt"))

        self.save_constraints([(constraint[0], constraint[1], constraint[2]) for constraint in
                               self.backend.registered_loop_constraints()],
                              str(dir_path / "loop_constraints.txt"))

    @staticmethod
    def save_constraints(constraints, file_path: str):

        import pandas as pd
        constraints_list = [(constraint[0], constraint[1], *constraint[2].flatten().tolist()) for constraint in
                            constraints]
        constraint_df = pd.DataFrame(constraints_list, columns=["src", "tgt", *[str(i) for i in range(16)]])
        constraint_df.to_csv(file_path, sep=",")

    @staticmethod
    def load_constraints(file_path: str):
        """Loads trajectory constraints from disk"""
        import pandas as pd

        constraints_df: pd.DataFrame = pd.read_csv(file_path, sep=",")
        constraint_rows = constraints_df.values.tolist()
        return constraint_rows
