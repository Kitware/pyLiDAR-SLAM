from pathlib import Path
from typing import Optional, List
import os

import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from torch.utils.data import Dataset
from scipy.io import loadmat

# Hydra and OmegaConf
from hydra.conf import dataclass, MISSING, field

# Project Imports
from slam.common.pose import Pose
from slam.common.projection import SphericalProjector
from slam.common.utils import assert_debug
from slam.dataset import DatasetLoader, DatasetConfig
from slam.eval.eval_odometry import compute_relative_poses


# ----------------------------------------------------------------------------------------------------------------------
class FordCampusSequence(Dataset):
    """
    Ford Campus Sequence
    """

    def __init__(self,
                 sequence_dir: str,
                 projector: Optional[SphericalProjector] = None,
                 with_gt: bool = True,
                 vertex_map_channel: str = "vertex_map",
                 gt_channel: str = "trajectory_gt",
                 pc_channel: str = "numpy_pc"):
        self.sequence_dir = Path(sequence_dir) / "SCANS"
        assert_debug(self.sequence_dir.exists())
        self.list_of_files = list(sorted(os.listdir(str(self.sequence_dir))))
        self.projector = projector

        self._with_gt = with_gt
        self._vmap_channel = vertex_map_channel
        self._gt_channel = gt_channel
        self._pc_channel = pc_channel
        self._pose = Pose("euler")
        self.__np_sensor_to_vehicule = np.array([[0.0, 1.0, 0.0],
                                                 [-1.0, 0.0, 0.0],
                                                 [0.0, 0.0, 1.0]], dtype=np.float32)
        self._sensor_to_vehicule = torch.tensor([[0.0, 1.0, 0.0, 0.0],
                                                 [-1.0, 0.0, 0.0, 0.0],
                                                 [0.0, 0.0, 1.0, 0.0],
                                                 [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)

    def __len__(self):
        return len(self.list_of_files)

    def __read_scan(self, idx):
        scan_file = str(self.sequence_dir / self.list_of_files[idx])
        mat_content = loadmat(scan_file)
        return mat_content["SCAN"]

    def __getitem__(self, idx):
        assert_debug(0 <= idx < self.__len__())
        mat_content = self.__read_scan(idx)

        pc_sensor = mat_content["XYZ"][0, 0].T
        pc_sensor = pc_sensor[np.linalg.norm(pc_sensor, axis=-1) > 8]
        pc_vehicule = np.einsum("ij,nj->ni", self.__np_sensor_to_vehicule, pc_sensor)

        # Put the PC into the list, so that pytorch does not try to aggregate uncompatible pointcloud dimensions
        data_dict = {self._pc_channel: pc_vehicule}
        if self.projector:
            torch_pc = torch.from_numpy(pc_vehicule).unsqueeze(0)
            vmap = self.projector.build_projection_map(torch_pc)[0]
            data_dict[self._vmap_channel] = vmap.to(torch.float32)

        if self._with_gt:
            gt_params = mat_content["X_wv"][0, 0].T
            vehicule_to_world = self._pose.build_pose_matrix(torch.from_numpy(gt_params))[0].to(torch.float32)
            data_dict[self._gt_channel] = vehicule_to_world

        return data_dict


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class FordCampusConfig(DatasetConfig):
    """A Configuration object read from a yaml conf"""

    # --------------------
    root_dir: str = MISSING
    dataset: str = "ford_campus"

    # -------------------
    up_fov: float = 3
    down_fov: float = -25
    lidar_height: int = 64
    lidar_width: int = 720

    train_sequences: List[str] = field(default_factory=lambda: ["dataset-1", "dataset-2"])
    test_sequences: List[str] = field(default_factory=lambda: ["dataset-1", "dataset-2"])
    eval_sequences: List[str] = field(default_factory=lambda: [])


# Hydra -- stores a FordCampusConfig `ford_campus` in the `dataset` group
cs = ConfigStore.instance()
cs.store(group="dataset", name="ford_campus", node=FordCampusConfig)


class FordCampusDatasetLoader(DatasetLoader):
    """
    Configuration for Ford Dataset
    """

    def __init__(self, config: FordCampusConfig, **kwargs):
        super().__init__(config)
        self.root_dir = Path(self.config.root_dir)

    def projector(self) -> SphericalProjector:
        up_fov = self.config.up_fov
        down_fov = self.config.down_fov
        lidar_height = self.config.lidar_height
        lidar_width = self.config.lidar_width
        return SphericalProjector(up_fov=up_fov, down_fov=down_fov, height=lidar_height, width=lidar_width)

    _sequence_name_to_prefix = {"dataset-1": "IJRR-Dataset-1",
                                "dataset-2": "IJRR-Dataset-2"}

    def sequences(self):
        """Returns the tuples (dataset_config, sequence_name) for train, eval and test split on FordCampus"""
        train_sequences = self.config.train_sequences
        test_sequences = self.config.test_sequences
        gt_pose_channel = self.config.absolute_gt_key

        def __get_datasets(sequences):
            if sequences is None:
                return None
            datasets = []
            for sequence in sequences:
                assert_debug(sequence in self._sequence_name_to_prefix)
                dir = self.root_dir / self._sequence_name_to_prefix[sequence]
                datasets.append(FordCampusSequence(str(dir),
                                                   projector=self.projector(),
                                                   gt_channel=gt_pose_channel,
                                                   vertex_map_channel=self.config.vertex_map_key,
                                                   pc_channel=self.config.numpy_pc_key))
            return datasets

        return (__get_datasets(train_sequences), train_sequences), \
               None, \
               (__get_datasets(test_sequences), test_sequences), lambda x: x

    def get_ground_truth(self, sequence_name):
        poses_path = self.root_dir / self._sequence_name_to_prefix[sequence_name] / "poses_gt.npy"
        if poses_path.exists():
            absolute_gt = np.load(str(poses_path))
            relative = compute_relative_poses(absolute_gt)
            return relative
        return None
