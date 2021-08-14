import logging
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.spatial.transform.rotation import Rotation as R, Slerp
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

# Hydra and OmegaConf
from hydra.core.config_store import ConfigStore
from hydra.conf import dataclass, MISSING, field

# Project Imports
from slam.common.projection import SphericalProjector
from slam.common.utils import assert_debug
from slam.dataset.configuration import DatasetLoader, DatasetConfig
from slam.dataset.kitti_dataset import kitti_read_scan
from slam.eval.eval_odometry import compute_relative_poses


def kitti_360_poses(file_path: str):
    df = pd.read_csv(file_path, sep=" ", header=None)
    poses = df.values  # [N, 13] array

    frame_indices = poses[:, 0].astype(np.int32)
    pose_data = poses[:, 1:]

    n = pose_data.shape[0]
    pose_data = np.concatenate((pose_data, np.zeros((n, 3), dtype=np.float32), np.ones((n, 1), dtype=np.float32)),
                               axis=1)
    poses = pose_data.reshape((n, 4, 4))  # [N, 4, 4]
    return frame_indices, poses


def read_timestamps(file_path: str):
    """Read a timestamps file and convert it to float64 values
    """
    df = pd.read_csv(file_path, header=None, sep=",", names=["instants"],
                     dtype={"instants": "str"}, parse_dates=["instants"])
    timestamps = df.values.astype(np.int64).astype(np.float64)
    return timestamps.reshape(-1)


def drive_foldername(drive_id: int):
    return f"2013_05_28_drive_{drive_id:04}_sync"


__cam0_to_pose = np.array([[0.0371783278, -0.0986182135, 0.9944306009, 1.5752681039],
                           [0.9992675562, -0.0053553387, -0.0378902567, 0.0043914093],
                           [0.0090621821, 0.9951109327, 0.0983468786, -0.6500000000],
                           [0, 0, 0, 1]], dtype=np.float64)

__velo_to_cam0 = np.linalg.inv(np.array([[0.04307104361, -0.08829286498, 0.995162929, 0.8043914418],
                                         [-0.999004371, 0.007784614041, 0.04392796942, 0.2993489574],
                                         [-0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824],
                                         [0, 0, 0, 1]], dtype=np.float64))


def get_sequence_poses(root_dir: str, drive_id: int):
    """Returns the poses of a given drive"""
    assert_debug(drive_id in [0, 2, 3, 4, 5, 6, 7, 9, 10])
    sequence_foldername = drive_foldername(drive_id)
    root_dir = Path(root_dir)
    velodyne_path = root_dir / "data_3d_raw" / sequence_foldername / "velodyne_points"
    timestamps_path = velodyne_path / "timestamps.txt"
    gt_file = root_dir / "data_poses" / sequence_foldername / "poses.txt"

    gt_poses = None
    if gt_file.exists():
        index_frames, poses = kitti_360_poses(str(gt_file))
        timestamps = read_timestamps(str(timestamps_path))

        poses_key_times = timestamps[index_frames]
        rotations = R.from_matrix(poses[:, :3, :3])
        slerp = Slerp(poses_key_times, rotations)

        # Clamp timestamps at key times to allow interpolation
        timestamps = timestamps.clip(min=poses_key_times.min(), max=poses_key_times.max())

        frame_orientations = slerp(timestamps)
        frame_translations = interp1d(poses_key_times, poses[:, :3, 3], axis=0)(timestamps)

        # Compute one pose per frame by interpolating of the ground truth (there is less than a frame per pose)
        gt_poses = np.zeros((timestamps.shape[0], 4, 4), dtype=np.float64)
        gt_poses[:, :3, :3] = frame_orientations.as_matrix()
        gt_poses[:, :3, 3] = frame_translations
        gt_poses[:, 3, 3] = 1.0

        # Convert poses to the poses in the frame of the lidar
        gt_poses = np.einsum("nij,jk->nik", gt_poses, __cam0_to_pose.dot(__velo_to_cam0))

    else:
        logging.warning(f"[KITTI-360]The ground truth filepath {gt_file} does not exist")
    return gt_poses


class KITTI360Sequence(Dataset):
    """
    Dataset for a Sequence of KITTI-360 lidar dataset

    Attributes:
        kitti360_root_dir (str): The path to KITTI-360 data
        drive_id (int): The name id of drive [0, 2, 3, 4, 5, 6, 7, 9, 10]
    """

    __sequence_size = {
        0: 11518,
        2: 19240,
        3: 1031,
        4: 11587,
        5: 6743,
        6: 9699,
        7: 3396,
        9: 14056,
        10: 3836
    }

    def __init__(self,
                 kitti360_root_dir: str,
                 drive_id: int):
        self.root_dir: Path = Path(kitti360_root_dir)

        assert_debug(drive_id in [0, 2, 3, 4, 5, 6, 7, 9, 10])
        sequence_foldername = drive_foldername(drive_id)
        velodyne_path = self.root_dir / "data_3d_raw" / sequence_foldername / "velodyne_points"
        self.lidar_path = velodyne_path / "data"

        assert_debug(self.lidar_path.exists(), f"The drive directory {self.lidar_path} does not exist")
        self.size: int = self.__sequence_size[drive_id]
        self.gt_poses = get_sequence_poses(kitti360_root_dir, drive_id)
        self.gt_poses = np.einsum("ij,njk->nik", np.linalg.inv(self.gt_poses[0]), self.gt_poses)

    def __len__(self):
        return self.size

    def __getitem__(self, idx) -> dict:
        """
        Returns:
            A dictionary with the mapping defined in the constructor
        """
        assert_debug(idx < self.size)
        data_dict = {}

        xyz_r = kitti_read_scan(str(self.lidar_path / f"{idx:010}.bin"))
        data_dict["numpy_pc"] = xyz_r[:, :3]
        data_dict["numpy_reflectance"] = xyz_r[:, 3:]
        if self.gt_poses is not None:
            data_dict["absolute_pose_gt"] = self.gt_poses[idx]

        return data_dict


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class KITTI360Config(DatasetConfig):
    """A configuration object read from a yaml conf"""
    # -------------------
    # Required Parameters
    root_dir: str = MISSING
    dataset: str = "kitti_360"

    # ------------------------------
    # Parameters with default values
    lidar_height: int = 64
    lidar_width: int = 1024
    up_fov: int = 3
    down_fov: int = -24
    train_sequences: list = field(default_factory=lambda: [0, 2, 3, 4, 5, 6, 7, 9, 10])
    test_sequences: list = field(default_factory=lambda: [0, 2, 3, 4, 5, 6, 7])
    eval_sequences: list = field(default_factory=lambda: [9, 10])


# Hydra -- stores a KITTIConfig `kitti_360` in the `dataset` group
cs = ConfigStore.instance()
cs.store(group="dataset", name="kitti_360", node=KITTI360Config)


# ----------------------------------------------------------------------------------------------------------------------
class KITTI360DatasetLoader(DatasetLoader):
    """
    Configuration for KITTI-360 dataset
    see http://www.cvlibs.net/datasets/kitti-360/
    """

    def __init__(self, config: KITTI360Config):
        super().__init__(config)
        self.root_dir = Path(self.config.root_dir)
        assert_debug(self.root_dir.exists())

    def projector(self) -> SphericalProjector:
        """Default SphericalProjetor for KITTI (projection of a pointcloud into a Vertex Map)"""
        assert isinstance(self.config, KITTI360Config)
        lidar_height = self.config.lidar_height
        lidar_with = self.config.lidar_width
        up_fov = self.config.up_fov
        down_fov = self.config.down_fov
        # Vertex map projector
        projector = SphericalProjector(lidar_height, lidar_with, 3, up_fov, down_fov)
        return projector

    def get_ground_truth(self, drive_id: str):
        """Returns the relative ground truth poses associated to a sequence of KITTI-360"""
        drive_id = int(drive_id)
        return compute_relative_poses(get_sequence_poses(self.root_dir, drive_id))

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
        # Sets the path of the kitti benchmark
        train_sequence_ids = [str(_id) for _id in self.config.train_sequences]
        eval_sequence_ids = [str(_id) for _id in self.config.eval_sequences]
        test_sequence_ids = [str(_id) for _id in self.config.test_sequences]

        def __get_datasets(sequences: list):
            if sequences is None or len(sequences) == 0:
                return None

            datasets = [KITTI360Sequence(str(self.root_dir),
                                         int(sequence_id)) for sequence_id in sequences]
            return datasets

        train_datasets = __get_datasets(train_sequence_ids)
        eval_datasets = __get_datasets(eval_sequence_ids)
        test_datasets = __get_datasets(test_sequence_ids)

        return (train_datasets, train_sequence_ids), \
               (eval_datasets, eval_sequence_ids), \
               (test_datasets, test_sequence_ids), lambda x: x
