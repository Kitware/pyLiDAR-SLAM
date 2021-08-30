import os
import numpy as np
from torch.utils.data import Dataset
import open3d as o3d
from pathlib import Path
from typing import Optional
from scipy.spatial.transform.rotation import Rotation as R, Slerp
import logging

# Hydra and OmegaConf
from hydra.core.config_store import ConfigStore
from hydra.conf import dataclass, MISSING, field

# Project Imports
from slam.common.projection import SphericalProjector
from slam.common.utils import assert_debug
from slam.dataset.configuration import DatasetLoader, DatasetConfig
from slam.eval.eval_odometry import compute_relative_poses


def read_ground_truth(file_path: str):
    assert_debug(Path(file_path).exists())

    ground_truth_df = np.genfromtxt(str(file_path), delimiter=',', dtype=np.float64)
    seconds = ground_truth_df[:, 0]
    nano_seconds = ground_truth_df[:, 1]
    xyz = ground_truth_df[:, 2:5]
    qxyzw = ground_truth_df[:, 5:]

    num_poses = qxyzw.shape[0]
    poses = np.eye(4, dtype=np.float64).reshape(1, 4, 4).repeat(num_poses, axis=0)
    poses[:, :3, :3] = R.from_quat(qxyzw).as_matrix()
    poses[:, :3, 3] = xyz

    T_CL = np.eye(4, dtype=np.float32)
    T_CL[:3, :3] = R.from_quat([0.0, 0.0, 0.924, 0.383]).as_matrix()
    T_CL[:3, 3] = np.array([-0.084, -0.025, 0.050], dtype=np.float32)
    poses = np.einsum("nij,jk->nik", poses, T_CL)

    poses_timestamps = seconds * 10e9 + nano_seconds
    poses = np.einsum("ij,njk->nik", np.linalg.inv(poses[0]), poses)
    return poses, poses_timestamps


def pointcloud_poses(poses, poses_timestamps, filenames):
    """Associate to a pointcloud (given by a filename)
    the closest pose (in terms of timestamps) to the ground truth
    """
    timestamps = []
    for filename in filenames:
        tokens = filename.replace('.', '_ ').split("_")
        secs = float(tokens[1])
        nsecs = float(tokens[2])
        timestamps.append(secs * 10e9 + nsecs)

    file_timestamps = np.array(timestamps)
    # Associate closest poses (in terms of timestamps)
    file_indices = np.searchsorted(poses_timestamps, file_timestamps)

    return poses[file_indices]


class NHCDOdometrySequence(Dataset):
    """
    Dataset for a Sequence of the New Handheld College Dataset
    see https://ori-drs.github.io/newer-college-dataset/

    Attributes:
        sequences_root_dir (str): The path to KITTI odometry benchmark's data
        sequence_id (str): The name id of the sequence in ["01_short_experiment", "02_long_experiment"]

        lidar_projector (SphericalProjector): The Spherical Projector, which projects pointclouds in the image plane
        ground_truth_channel (Optional[str]): The key in the dictionary for the ground truth absolute pose
    """

    @staticmethod
    def num_frames(sequence_id: str):
        if sequence_id == "01_short_experiment":
            return 15302
        elif sequence_id == "02_long_experiment":
            # Remove the last 600 frames which correspond to the arrival of the sensor to the garage
            # And includes very abrupt, motions
            return 26000

    def __init__(self,
                 sequences_root_dir: str,
                 sequence_id: str,
                 lidar_projector: SphericalProjector = None,
                 pointcloud_channel: str = "numpy_pc",
                 ground_truth_channel: Optional[str] = "absolute_pose_gt",
                 with_numpy_pc: bool = False):
        self.dataset_root: Path = Path(sequences_root_dir)
        self.sequence_id: str = sequence_id
        self.ground_truth_channel = ground_truth_channel
        self.id = self.sequence_id
        self.lidar_projector = lidar_projector
        self._with_numpy_pc = with_numpy_pc
        self.pcd_paths: Path = self.dataset_root / sequence_id / "raw_format" / "ouster_scan"
        assert_debug(self.pcd_paths.exists(), "The path to the folders of the pcd files does not exist")
        self.file_names = [filename for filename in sorted(os.listdir(str(self.pcd_paths))) if "(1)" not in filename]
        self._size = self.num_frames(self.sequence_id)
        self.pointcloud_channel = pointcloud_channel

        ground_truth_path = self.dataset_root / sequence_id / "ground_truth" / "registered_poses.csv"
        self.has_gt = False
        self.poses = None
        self.poses_seconds = None
        self.poses_nanoseconds = None

        if ground_truth_path.exists():
            self.has_gt = True
            poses, poses_timestamps = read_ground_truth(str(ground_truth_path))

            # poses = np.linalg.inv(poses)
            self.poses = pointcloud_poses(poses, poses_timestamps, self.file_names)

            # For each file : associate the closest pose (in terms of timestamps)
            self.poses = np.einsum("ij,njk->nik", np.linalg.inv(poses[0]), poses)

    def __len__(self):
        return self._size

    def __getitem__(self, idx) -> dict:
        """
        Returns:
            A dictionary with the mapping defined in the constructor
        """
        assert_debug(idx < self._size)

        file_path = self.pcd_paths / self.file_names[idx]
        assert_debug(file_path.exists(), "Could not open the file " + str(file_path))

        data_dict = dict()
        pointcloud: o3d.geometry.PointCloud = o3d.io.read_point_cloud(str(file_path), "pcd")
        xyz = np.asarray(pointcloud.points).copy()
        del pointcloud

        # Read timestamps
        data_dict[self.pointcloud_channel] = xyz.astype(np.float32)
        N_rows = int(xyz.shape[0] / 64)
        timestamps = np.arange(N_rows).reshape(N_rows, 1).repeat(64, axis=1).reshape(-1, ).astype(np.float64)
        min_t = timestamps.min()
        max_t = timestamps.max()

        timestamps = (timestamps - min_t) / (max_t - min_t) + idx
        data_dict[f"{self.pointcloud_channel}_timestamps"] = timestamps

        if self.has_gt:
            data_dict[self.ground_truth_channel] = self.poses[idx]

        return data_dict


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class NHCDConfig(DatasetConfig):
    """A configuration object read from a yaml conf"""
    # -------------------
    # Required Parameters
    root_dir: str = MISSING
    dataset: str = "nhcd"

    # ------------------------------
    # Parameters with default values
    lidar_height: int = 64
    lidar_width: int = 1024
    up_fov: int = 25
    down_fov: int = -25
    train_sequences: list = field(default_factory=lambda: ["02_long_experiment", "01_short_experiment"])
    test_sequences: list = field(default_factory=lambda: ["01_short_experiment"])
    eval_sequences: list = field(default_factory=lambda: ["01_short_experiment"])


# Hydra -- stores a NHCDConfig `nhcd` in the `dataset` group
cs = ConfigStore.instance()
cs.store(group="dataset", name="nhcd", node=NHCDConfig)


# ----------------------------------------------------------------------------------------------------------------------
class NHCDDatasetLoader(DatasetLoader):
    """
    Dataset Loader for NHCD's dataset
    see https://ori-drs.github.io/newer-college-dataset/
    """

    def __init__(self, config: NHCDConfig):
        super().__init__(config)
        self.root_dir = Path(config.root_dir)
        assert_debug(self.root_dir.exists())

    def projector(self) -> SphericalProjector:
        """Default SphericalProjetor for NHCD (projection of a pointcloud into a Vertex Map)"""
        lidar_height = self.config.lidar_height
        lidar_with = self.config.lidar_width
        up_fov = self.config.up_fov
        down_fov = self.config.down_fov
        # Vertex map projector
        projector = SphericalProjector(lidar_height, lidar_with, 3, up_fov, down_fov)
        return projector

    def get_ground_truth(self, sequence_name):
        """Returns the ground truth poses"""
        assert_debug(sequence_name in ["01_short_experiment", "02_long_experiment"])
        poses_file = self.root_dir / sequence_name / "ground_truth" / "registered_poses.csv"
        if not poses_file.exists():
            return None
        poses, poses_timestamps = read_ground_truth(str(poses_file))
        scans_dir = self.root_dir / sequence_name / "raw_format" / "ouster_scan"
        if not scans_dir.exists():
            logging.log(logging.ERROR,
                        f"The folder containing the ouster scan does not exist on disk at location {scans_dir}. "
                        "Cannot read the ground truth")
            return None

        absolute_poses = pointcloud_poses(poses, poses_timestamps, sorted(os.listdir(str(scans_dir))))
        absolute_poses = absolute_poses[:NHCDOdometrySequence.num_frames(sequence_name)]
        return compute_relative_poses(absolute_poses)

    def sequences(self):
        """
        Returns
        -------
        (train_dataset, eval_dataset, test_dataset, transform) : tuple
        train_dataset : (list, list)
            A list of dataset_config (one for each sequence of KITTI's Dataset),
            And the list of sequences used to build them
        eval_dataset : (list, list)
            idem
        test_dataset : (list, list)
            idem
        transform : callable
            A transform to be applied on the dataset_config
        """

        gt_pose_channel = self.config.absolute_gt_key

        # Sets the path of the kitti benchmark
        train_sequence_ids = self.config.train_sequences
        eval_sequence_ids = self.config.eval_sequences
        test_sequence_ids = self.config.test_sequences

        def __get_datasets(sequences: list):
            if sequences is None or len(sequences) == 0:
                return None

            datasets = [NHCDOdometrySequence(
                str(self.root_dir),
                sequence_id,
                self.projector(), self.config.numpy_pc_key, gt_pose_channel,
                with_numpy_pc=self.config.with_numpy_pc) for sequence_id in sequences]
            return datasets

        train_datasets = __get_datasets(train_sequence_ids)
        eval_datasets = __get_datasets(eval_sequence_ids)
        test_datasets = __get_datasets(test_sequence_ids)

        return (train_datasets, train_sequence_ids), \
               (eval_datasets, eval_sequence_ids), \
               (test_datasets, test_sequence_ids), lambda x: x
