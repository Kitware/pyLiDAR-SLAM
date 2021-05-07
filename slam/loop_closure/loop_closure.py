from abc import ABC
from enum import Enum
from typing import Union, Optional
import numpy as np
import matplotlib.pyplot as plt

# Hydra and OmegaConf
from omegaconf import DictConfig
from dataclasses import asdict
from hydra.conf import dataclass, MISSING

# Project Imports
from slam.common.utils import assert_debug, check_sizes
from slam.common.registration import _with_cv2
from slam.backend.backend import Backend


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
if _with_cv2:

    import cv2
    from slam.common.registration import ElevationImageRegistration


    @dataclass
    class EILoopClosureConfig(LoopClosureConfig):
        type = "ei"

        map_size: int = 100  # The number of scans to aggregate before building an elevation image
        overlap: bool = True  # Whether to build overlapping scans
        overlap_size: int = 60  # The size (in number of scans) of the overlap

        # Image Registration Parameters
        im_height: int = 800
        im_width: int = 800
        inlier_threshold: int = 50
        distance_threshold: float = 2.0
        pixel_size: int = 0.2
        z_min: float = -3.0
        z_max: float = 15

        # Candidate search parameters
        min_num_frames: int = 20
        max_num_candidates: int = 30


    class ElevationImageLoopClosure(LoopClosure):
        """
        Attributes:
            sensor_absolute_pose (np.ndarray): The Pose of the last inserted frame `(4, 4)`
            poses_in_wc (list): The list of poses
            pointclouds (list): The list of pointclouds
        """

        def __init__(self, config: EILoopClosureConfig, **kwargs):
            super().__init__(config)
            self.last_frame_pose: Optional[np.ndarray] = None
            self.absolute_poses: Optional[np.ndarray] = None
            self.pointclouds: Optional[list] = None  # The list of pointclouds of the current frame being built
            self.pointcloud_ids: Optional[list] = None  # List of point cloud ids

            self.frame_pc_ids: Optional[np.ndarray] = None
            self._pose_idx: int = 0  # Keeps track of the poses added to define loop closure constraints
            self._frame_idx: int = 0  # Keeps track of the frames added

            # Keeps tracks of features per descriptor
            self.frames_features = None

            # 2D registration
            self.ei_algo = ElevationImageRegistration(DictConfig(asdict(config)))

            # TODO : SUPPRESS BEFORE COMMIT ViZ
            self._winname = "ei"
            self._winname_matches = "matches"
            self._win_initialized = False

        def __del__(self):
            if self._win_initialized:
                cv2.destroyWindow(self._winname)
                cv2.destroyWindow(self._winname_matches)

        def clean(self):
            self.last_frame_pose = np.eye(4, dtype=np.float64)
            self.absolute_poses = None
            self.pointclouds = []
            self.pointcloud_ids = []
            self.frames_features = dict()
            self.frame_pc_ids = None
            self._pose_idx: int = 1
            self._frame_idx: int = 0

            # cv2.destroyWindow(self._winname)
            cv2.namedWindow(self._winname, cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
            cv2.namedWindow(self._winname_matches, cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)

            self._win_initialized = True

        def process_next_frame(self, data_dict: dict):
            if self.relative_pose_key() in data_dict:
                mat_tgt_to_ref = data_dict[self.relative_pose_key()].astype(np.float64)
                self.last_frame_pose = self.last_frame_pose.dot(mat_tgt_to_ref)

                if self.pointcloud_key() in data_dict:
                    pointcloud = data_dict[self.pointcloud_key()]

                    # Transform the pointcloud in the reference frame of the world
                    pc_in_wc = np.einsum("ij,nj->ni", self.last_frame_pose[:3, :3], pointcloud) + \
                               self.last_frame_pose[:3, 3].reshape(1, 3)

                    self.pointclouds.append(pc_in_wc)
                    self.pointcloud_ids.append(self._pose_idx)

                if self.absolute_poses is None:
                    self.absolute_poses = self.last_frame_pose.reshape(1, 4, 4)
                else:
                    self.absolute_poses = np.concatenate([self.absolute_poses, self.last_frame_pose.reshape(1, 4, 4)],
                                                         axis=0)
                self._pose_idx += 1

            if len(self.pointclouds) >= self.config.map_size:
                # Process the frame
                frame_id, features = self.build_elevation_image()
                candidates = self.select_candidates()
                if len(candidates) > 0:
                    for candidate in candidates:
                        frame_to_candidate = self.match_candidates(candidate, features)
                        if frame_to_candidate is not None:
                            # Add a loop closure constraint
                            data_dict[Backend.se3_loop_closure_constraint(candidate, frame_id)] = (
                                frame_to_candidate, None)

                self.add_new_frame(frame_id, features)

        def build_elevation_image(self):
            """Builds an elevation image centered from the pointcloud"""
            assert len(self.pointclouds) > 0
            aggregate_pc = np.concatenate(self.pointclouds)
            middle_idx = len(self.pointclouds) // 2
            middle_pose_idx = self.pointcloud_ids[middle_idx]
            pose = np.linalg.inv(self.absolute_poses[middle_pose_idx])  # Pose of the middle pointcloud

            transformed_pc = np.einsum("ij,nj->ni", pose[:3, :3], aggregate_pc) + pose[:3, 3].reshape(1, 3)

            # Build Elevation Image
            image, kpts, desc = self.ei_algo.compute_features(transformed_pc)

            cv2.imshow(self._winname, image)
            cv2.waitKey(5)

            # TODO Build GLOBAL DESCRIPTOR
            return middle_pose_idx, (image, kpts, desc)

        def update_positions(self, trajectory: np.ndarray):
            check_sizes(trajectory, list(self.absolute_poses.shape))
            self.absolute_poses = trajectory
            self.last_frame_pose = trajectory[-1]

        def select_candidates(self) -> list:
            """Selects Candidates to be matched

            The candidates are selected as the closest points to the current frame
            """
            if self.frame_pc_ids is None:
                return []
            frame_poses = self.absolute_poses[self.frame_pc_ids[:-self.config.min_num_frames]]
            # Sort the frames by distance to the last inserted frame
            trajectory_distances = np.linalg.norm(frame_poses[:, :3, 3] - \
                                                  self.last_frame_pose[:3, 3].reshape(1, 3), axis=-1)

            # Select a fixed number of candidates
            candidate_ids = np.argsort(trajectory_distances)[:self.config.max_num_candidates]
            pc_candidate_ids = self.frame_pc_ids[candidate_ids]

            return pc_candidate_ids.tolist()

        def match_candidates(self, candidate_pc_id, features):
            image, kpts, desc = features
            ref_image, kpts_cd, desc_cd = self.frames_features[candidate_pc_id]

            # Match OpenCV Keypoints
            transform, points, inlier_matches = self.ei_algo.align_2d(kpts_cd, desc_cd, kpts, desc, None, None)

            if transform is None:
                return None

            # TODO DEBUG REMOVE
            image_matches = cv2.drawMatches(ref_image, kpts_cd, image, kpts, inlier_matches, None)

            cv2.imshow(self._winname_matches, image_matches)
            cv2.waitKey(5)

            print("Found Loop")  # DEBUG REMOVE

            return transform

        def add_new_frame(self, frame_pc_id, features):
            self.frames_features[frame_pc_id] = tuple(features)
            if self.frame_pc_ids is None:
                self.frame_pc_ids = np.array([frame_pc_id])
            else:
                self.frame_pc_ids = np.concatenate([self.frame_pc_ids, [frame_pc_id]], axis=0)
            self._frame_idx += 1

            # Cleans the last frames
            if self.config.overlap:
                start_idx = max(0, self.config.map_size - 1 - self.config.overlap_size)
                self.pointclouds = [self.pointclouds[i] for i in range(start_idx, self.config.map_size)]
                self.pointcloud_ids = [self.pointcloud_ids[i] for i in range(start_idx, self.config.map_size)]

            else:
                self.pointclouds = []
                self.pointcloud_ids = []


# ----------------------------------------------------------------------------------------------------------------------
class LOOP_CLOSURE(Enum):
    if _with_cv2:
        ei = (ElevationImageLoopClosure, EILoopClosureConfig)

    @staticmethod
    def load(config: Union[dict, DictConfig, LoopClosureConfig], **kwargs) -> LoopClosure:
        if isinstance(config, dict):
            config = DictConfig(config)
        elif isinstance(config, DictConfig) or isinstance(config, LoopClosureConfig):
            config = config
        else:
            raise NotImplementedError("")

        _type = config.type
        assert_debug(_type in LOOP_CLOSURE.__members__,
                     f"The LoopClosure type {_type} is not available or not implemented")

        _class, _config = LOOP_CLOSURE[_type].value

        if isinstance(config, LoopClosureConfig):
            return _class(config, **kwargs)
        else:
            return _class(_config(**config), **kwargs)
