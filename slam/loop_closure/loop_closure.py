import copy
import logging
from abc import ABC
from collections import namedtuple
from enum import Enum

import numpy as np

# Hydra and OmegaConf
from omegaconf import DictConfig, OmegaConf
from hydra.conf import dataclass, MISSING, ConfigStore, field

# Project Imports
from slam.backend.backend import Backend
from slam.common.pointcloud import grid_sample
from slam.common.modules import _with_cv2, _with_o3d
from slam.common.pose import transform_pointcloud
from slam.common.registration import ElevationImageRegistration
from slam.common.utils import assert_debug, check_tensor
from slam.common.utils import assert_debug, check_tensor, ObjectLoaderEnum

from slam.viz import _with_cv2

import open3d as o3d


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

    if _with_o3d:
        import open3d as o3d


    @dataclass
    class EILoopClosureConfig:
        """Configuration for a ElevationImageLoopClosure Algorithm"""
        type: str = "elevation_image"
        local_map_size: int = 50  # The number of frames in the stored local map
        overlap: int = 20  # The number of frames overlapping in the stored local map
        debug: bool = False
        max_num_candidates: int = 10  # Maximum number of candidates to inspect
        max_distance: float = 100  # Limit the maximum distance to search for loops
        min_id_distance: int = 200  # Do not try to detect loop closure between temporally close poses

        icp_distance_threshold: float = 1.0
        with_icp_refinement: bool = _with_o3d  # Only activated if open3d can be loaded

        ei_registration_config: DictConfig = field(default_factory=lambda: OmegaConf.create({
            "features": "akaze",
            "pixel_size": 0.1,
            "z_min": -3.0,
            "z_max": 5,
            "sigma": 0.1,
            "im_height": 1200,
            "im_width": 1200,
            "color_map": "jet",
            "inlier_threshold": 50,
            "distance_threshold": 2.0
        }))


    @dataclass
    class MapData:
        local_map_data: list = field(default_factory=lambda: [])
        last_inserted_pose: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
        current_frame_id: int = 0
        all_frames_absolute_poses: list = field(default_factory=lambda: [])
        maps_absolute_poses: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 4), dtype=np.float64))
        maps_frame_ids: list = field(default_factory=lambda: [])

        current_map_pcs: list = field(default_factory=lambda: [])  # Store the pointclouds
        current_map_poses: list = field(default_factory=lambda: [])  # Absolute poses
        current_map_frameids: list = field(default_factory=lambda: [])


    LocalMapData = namedtuple("LocalMapData", ['keypoints', 'descriptors', 'pointcloud', 'frame_id'])

    if _with_o3d:
        import open3d as o3d


        def draw_registration_result(source, target, transformation):
            source_temp = copy.deepcopy(source)
            target_temp = copy.deepcopy(target)
            source_temp.paint_uniform_color([1, 0.706, 0])
            target_temp.paint_uniform_color([0, 0.651, 0.929])
            source_temp.transform(transformation)
            o3d.visualization.draw_geometries([source_temp, target_temp],
                                              zoom=0.4459,
                                              front=[0.9288, -0.2951, -0.2242],
                                              lookat=[1.6784, 2.0612, 1.4451],
                                              up=[-0.3402, -0.9189, -0.1996])


    class ElevationImageLoopClosure(LoopClosure):
        """
        An Implementation of a Loop Detection and Estimation Algorithm
        """

        def __init__(self, config: EILoopClosureConfig, **kwargs):
            super().__init__(config, **kwargs)

            self.registration_2D = ElevationImageRegistration(config.ei_registration_config)
            self.with_window = config.debug
            self.winname = "Loop Closure Map"
            if config.debug:
                cv2.namedWindow(self.winname, cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)

            self.data = MapData()
            self.maps_saved_data: list = []

        def serialize(self) -> MapData:
            # Return MapData (convert cv2.KeyPoint which are not handled by the pickling protocol)
            def convert_tuple(nt: namedtuple):
                keypoints, descriptors, pointcloud, frame_id = nt
                return ([(kpt.pt, kpt.size, kpt.angle, kpt.response, kpt.octave,
                          kpt.class_id) for kpt in keypoints], descriptors, pointcloud, frame_id)

            self.data.local_map_data = [convert_tuple(nt) for nt in self.maps_saved_data]
            return self.data

        def update_positions(self, trajectory: np.ndarray):
            check_tensor(trajectory, [self.data.current_frame_id, 4, 4])
            if self.data.current_frame_id == 0:
                return

            num_saved_poses = len(self.data.all_frames_absolute_poses)
            self.data.all_frames_absolute_poses = [trajectory[idx] for idx in range(num_saved_poses)]
            self.data.maps_absolute_poses = np.array([trajectory[frame_id] for frame_id in self.data.maps_frame_ids])
            self.data.last_inserted_pose = trajectory[-1]

            num_poses_in_current_map = len(self.data.current_map_pcs)
            for idx in range(num_poses_in_current_map):
                self.data.current_map_poses[-idx] = trajectory[-idx]

        def load(self, map_data: MapData):
            # Return MapData (convert cv2.KeyPoint which are not handled by the pickling protocol)
            def convert_tuple(_tuple):
                keypoints, descriptors, pointcloud, frame_id = _tuple
                return LocalMapData(
                    [cv2.KeyPoint(kpt[0][0], kpt[0][1], kpt[1], kpt[2], kpt[3], kpt[4], kpt[5]) for kpt in keypoints],
                    descriptors, pointcloud, frame_id)

            self.data = map_data
            self.maps_saved_data = [convert_tuple(_tuple) for _tuple in map_data.local_map_data]

        def __del__(self):
            if self.with_window:
                cv2.destroyWindow(self.winname)

        def clean(self):
            self.data.current_map_pcs.clear()
            self.data.current_map_poses.clear()
            self.data.current_map_frameids.clear()
            self.data.all_frames_absolute_poses.clear()
            self.data.maps_frame_ids.clear()
            self.data.last_inserted_pose = np.eye(4, dtype=np.float64)
            self.data.current_frame_id = 0
            self.data.maps_absolute_poses = np.zeros((0, 4, 4), dtype=np.float64)
            self.maps_saved_data.clear()

        def _compute_transform(self, initial_transform, candidate_pc, target_pc):
            if not _with_o3d:
                return initial_transform

            assert isinstance(self.config, EILoopClosureConfig)
            # Refine the transform by an ICP on the point cloud
            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(candidate_pc)
            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(target_pc)

            result = o3d.pipelines.registration.registration_icp(
                source, target, self.config.icp_distance_threshold, initial_transform.astype(np.float64),
                o3d.pipelines.registration.TransformationEstimationPointToPoint())

            return np.linalg.inv(result.transformation), candidate_pc, target_pc

        def _match_candidates(self, candidate_ids, feat, desc, points, frame_id, data_dict: dict):
            assert isinstance(self.config, EILoopClosureConfig)
            for candidate in candidate_ids:
                cd_feat, cd_desc, cd_pc_image, cd_frame_id = self.maps_saved_data[candidate]
                transform, points_2D, inlier_matches = self.registration_2D.align_2d(feat, desc, cd_feat,
                                                                                     cd_desc, None, None)
                if self.config.debug:
                    logging.info(f"Found {len(inlier_matches)}")
                if transform is not None:
                    if self.config.with_icp_refinement and _with_o3d:
                        cd_points = cd_pc_image.reshape(-1, 3)
                        cd_points = cd_points[np.linalg.norm(cd_points, axis=1) > 0]
                        tgt_points = points.reshape(-1, 3)
                        tgt_points = tgt_points[np.linalg.norm(tgt_points, axis=1) > 0]

                        transform, source, target = self._compute_transform(transform, cd_points, tgt_points)

                        data_dict["transform"] = transform
                        data_dict["source"] = (cd_frame_id, source)
                        data_dict["target"] = (frame_id, target)

                    # Add the constraint to the data_dict
                    key = Backend.se3_loop_closure_constraint(cd_frame_id, frame_id)
                    logging.info(f"[LOOP CLOSURE] Found constraint between frame {cd_frame_id} and {frame_id}")
                    data_dict[key] = (transform, None)

        def process_next_frame(self, data_dict: dict):
            assert isinstance(self.config, EILoopClosureConfig)
            if self.data.current_frame_id > 0:
                assert_debug(self.relative_pose_key() in data_dict,
                             f"The Key `{self.relative_pose_key()}` must be defined at each time step for the loop closure to work."
                             f"Keys in the dictionary : {data_dict.keys()}")

                relative_pose = data_dict[self.relative_pose_key()]  # 4, 4
            else:
                relative_pose = np.eye(4, dtype=np.float64)

            # Update the absolute pose of the last inserted pointcloud
            self.data.last_inserted_pose = self.data.last_inserted_pose.dot(relative_pose)

            if self.pointcloud_key() not in data_dict:
                self.data.current_frame_id += 1
                return data_dict

            # Step 1: Add the aggregated pointcloud
            pointcloud = data_dict[self.pointcloud_key()]
            pointcloud = grid_sample(pointcloud, self.config.ei_registration_config.pixel_size * 2)[0]  # N, 3

            check_tensor(pointcloud, [-1, 3], np.ndarray)
            check_tensor(relative_pose, [4, 4], np.ndarray)

            self.data.current_map_pcs.append(transform_pointcloud(pointcloud, self.data.last_inserted_pose))
            self.data.current_map_poses.append(np.copy(self.data.last_inserted_pose))
            self.data.current_map_frameids.append(self.data.current_frame_id)

            # Step 2: Construct a local submap and run the loop closure
            if len(self.data.current_map_pcs) >= self.config.local_map_size:
                mid_pose_index = len(self.data.current_map_pcs) // 2
                aggregated_pc = np.concatenate(self.data.current_map_pcs, axis=0)
                mid_pose = self.data.current_map_poses[mid_pose_index]
                mid_pose_frame_id = self.data.current_map_frameids[mid_pose_index]
                mid_pose_I = np.linalg.inv(mid_pose)
                aggregated_pc = transform_pointcloud(aggregated_pc, mid_pose_I)

                # Project the pose in the image frame
                image, feat, desc, meta_data = self.registration_2D.compute_features(aggregated_pc)

                if self.with_window:
                    cv2.imshow(self.winname, image)
                    cv2.waitKey(10)

                # Step 3: Search for a match along the closest persisted poses
                local_map_id_distance = self.config.min_id_distance // (
                        self.config.local_map_size - self.config.overlap)
                if self.data.maps_absolute_poses.shape[0] > local_map_id_distance:
                    candidate_indices = np.arange(self.data.maps_absolute_poses.shape[0])[:-local_map_id_distance]
                    candidate_poses = self.data.maps_absolute_poses[:, :3, 3][:-local_map_id_distance]
                    distances_to_mid_pose = np.linalg.norm(candidate_poses -
                                                           mid_pose[:3, 3].reshape(1, 3), axis=1)
                    _filter_distance = distances_to_mid_pose < self.config.max_distance
                    candidate_indices = candidate_indices[_filter_distance]
                    distances_to_mid_pose = distances_to_mid_pose[_filter_distance]
                    if distances_to_mid_pose.shape[0] > 0:
                        indices = np.argsort(distances_to_mid_pose)[:self.config.max_num_candidates]
                        self._match_candidates(candidate_indices[indices], feat, desc, meta_data["points_3D"],
                                               mid_pose_frame_id, data_dict)

                # Step 4: Save the matches and the new pose to disk
                self.data.maps_absolute_poses = np.concatenate(
                    [self.data.maps_absolute_poses, mid_pose.reshape(1, 4, 4)],
                    axis=0)
                self.data.maps_frame_ids.append(mid_pose_frame_id)
                self.maps_saved_data.append(LocalMapData(feat, desc,
                                                         meta_data["points_3D"], mid_pose_frame_id))

                self.data.all_frames_absolute_poses += self.data.current_map_poses[:-self.config.overlap]

                # Remove old frames from the local map
                self.data.current_map_pcs = self.data.current_map_pcs[-self.config.overlap:]
                self.data.current_map_poses = self.data.current_map_poses[-self.config.overlap:]
                self.data.current_map_frameids = self.data.current_map_frameids[-self.config.overlap:]

            # Persist the information (Match) accross all images
            self.data.current_frame_id += 1
            return data_dict

# Add Elevation Image to the Config Store
cs = ConfigStore.instance()
cs.store(name="none", group="slam/loop_closure", node=None)
if _with_cv2:
    # Add Elevation Image to the Config Store
    cs = ConfigStore.instance()
    cs.store(name="elevation_image", group="slam/loop_closure", node=EILoopClosureConfig)
    cs.store(name="none", group="slam/loop_closure", node=None)


# ----------------------------------------------------------------------------------------------------------------------
class LOOP_CLOSURE(ObjectLoaderEnum, Enum):
    none = (None, None)
    if _with_cv2:
        elevation_image = (ElevationImageLoopClosure, EILoopClosureConfig)

    @classmethod
    def type_name(cls):
        return "type"
