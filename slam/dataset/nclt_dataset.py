import logging
import os
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import scipy.interpolate
from hydra.core.config_store import ConfigStore
from scipy.spatial.transform import Rotation

import torch
from torch.utils.data import Dataset

from omegaconf import MISSING
from hydra.conf import field, dataclass

# Project Imports
from slam.common.pose import PosesInterpolator
from slam.common.projection import SphericalProjector
from slam.common.utils import assert_debug, remove_nan
from slam.dataset import DatasetLoader, DatasetConfig


# ----------------------------------------------------------------------------------------------------------------------
def _convert(x_s, y_s, z_s):
    # Copied from http://robots.engin.umich.edu/nclt/python/read_vel_sync.py
    scaling = 0.005
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset
    return x, y, z


# ----------------------------------------------------------------------------------------------------------------------
class NCLTSequenceDataset(Dataset):
    """
    A Torch Dataset for a sequence of the NCLT Dataset

    see http://robots.engin.umich.edu/nclt for the link to the Dataset's main page

    TODO:
        - Use the velodyne_hits.bin file (much better data)
        - Allow the expression of the data in the different reference frames (body, velodyne, velodyne_inverted)
    """

    def __init__(self, root_dir: str, sequence_id: str, vertex_map_key: str, projector: SphericalProjector):
        super().__init__()
        self.sequence_dir = Path(root_dir) / sequence_id
        assert_debug(self.sequence_dir.exists(), f"The directory {self.sequence_dir} does not exist on disk")

        # Prepare Velodyne files
        self.velodyne_dir = self.sequence_dir / "velodyne_sync"

        timestamps, files, gt = self.timestamps_files_and_gt(root_dir, sequence_id)

        self.velodyne_files = files
        self.timestamps = timestamps
        self._with_gt = gt is not None
        self._gt = gt
        self.gt_channel = "gt_absolute_pose"

        self._size = self.velodyne_files.shape[0]
        self.vertex_map_key = vertex_map_key
        self.projector = projector

    @staticmethod
    def timestamps_files_and_gt(root_path: str, sequence_id: str) -> Tuple[np.ndarray, np.ndarray,
                                                                           Optional[np.ndarray]]:
        root_path = Path(root_path)
        assert_debug(root_path.exists())
        sequence_path = root_path / sequence_id
        assert_debug(sequence_path.exists())
        velodyne_dir = sequence_path / "velodyne_sync"

        velodyne_files = np.array(sorted(os.listdir(str(velodyne_dir))), dtype=np.str)
        timestamps = np.array([file.split(".")[0] for file in velodyne_files], dtype=np.int64)
        ground_truth_file = sequence_path / f"groundtruth_{sequence_id}.csv"

        gt = None
        if ground_truth_file.exists():
            ground_truth = NCLTSequenceDataset.read_ground_truth(str(ground_truth_file))

            # Ground truth timestamps and LiDARs don't match, interpolate
            gt_t = ground_truth[:, 0]
            t_min = np.min(gt_t)
            t_max = np.max(gt_t)
            inter = scipy.interpolate.interp1d(ground_truth[:, 0], ground_truth[:, 1:], kind='nearest', axis=0)

            # Limit the sequence to timestamps for which a ground truth exists
            filter_ = (timestamps > t_min) * (timestamps < t_max)
            timestamps = timestamps[filter_]
            velodyne_files = velodyne_files[filter_]

            gt = inter(timestamps)
            gt_tr = gt[:, :3]
            gt_euler = gt[:, 3:][:, [2, 1, 0]]
            gt_rot = Rotation.from_euler("ZYX", gt_euler).as_matrix()

            gt = np.eye(4, dtype=np.float32).reshape(1, 4, 4).repeat(gt.shape[0], axis=0)
            gt[:, :3, :3] = gt_rot
            gt[:, :3, 3] = gt_tr

            gt = np.einsum("nij,jk->nik", gt, np.array([[1.0, 0.0, 0.0, 0.0],
                                                        [0.0, -1.0, 0.0, 0.0],
                                                        [0.0, 0.0, -1.0, 0.0],
                                                        [0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
            gt = np.einsum("ij,njk->nik", np.array([[1.0, 0.0, 0.0, 0.0],
                                                    [0.0, -1.0, 0.0, 0.0],
                                                    [0.0, 0.0, -1.0, 0.0],
                                                    [0.0, 0.0, 0.0, 1.0]], dtype=np.float32), gt)

        return timestamps, velodyne_files, gt

    @staticmethod
    def read_ground_truth(gt_file: str):
        gt = pd.read_csv(gt_file, sep=",").values
        return gt

    @staticmethod
    def interpolate_ground_truth(ground_truth: np.ndarray,
                                 timestamps: np.ndarray,
                                 reference_frame: Optional[str] = None):
        """Interpolates the poses from the ground truth using the provided timestamps

        Parameters:
            ground_truth (np.ndarray): The ground truth data as provided by `read_ground_truth` `(-1, 7)`
            timestamps (np.ndarray): The point cloud poses
            reference_frame (str): The frame of reference to which the data should return
                                   In the "imu" frame (provided by the dataset),
                                   In the "velodyne" frame of the LiDAR sensor
                                   In the "velodyne_inverted" modifying the velodyne frame to have the z axis point up
        """
        if reference_frame is None:
            reference_frame = "velodyne_inverted"
        assert_debug(reference_frame in ["body", "velodyne", "velodyne_inverted"])
        gt_t = ground_truth[:, 0]
        gt_t, _filter = remove_nan(gt_t)
        gt = ground_truth[:, 1:][_filter]
        t_min = np.min(gt_t)
        t_max = np.max(gt_t)
        gt_tr = gt[:, :3]
        gt_euler = gt[:, 3:][:, [2, 1, 0]]
        gt_rot = Rotation.from_euler("ZYX", gt_euler).as_matrix()

        gt = np.eye(4, dtype=np.float32).reshape(1, 4, 4).repeat(gt.shape[0], axis=0)
        gt[:, :3, :3] = gt_rot
        gt[:, :3, 3] = gt_tr

        if reference_frame == "velodyne_inverted":
            gt = np.einsum("nij,jk->nik", gt, np.array([[1.0, 0.0, 0.0, 0.0],
                                                        [0.0, -1.0, 0.0, 0.0],
                                                        [0.0, 0.0, -1.0, 0.0],
                                                        [0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
            gt = np.einsum("ij,njk->nik", np.array([[1.0, 0.0, 0.0, 0.0],
                                                    [0.0, -1.0, 0.0, 0.0],
                                                    [0.0, 0.0, -1.0, 0.0],
                                                    [0.0, 0.0, 0.0, 1.0]], dtype=np.float32), gt)
        if reference_frame == "velodyne":
             gt = np.einsum("nij,jk->nik", gt, np.array([[0.0, 1.0, 0.0, 0.0],
                                                         [-1.0, 0.0, 0.0, 0.0],
                                                         [0.0, 0.0, 1.0, 0.0],
                                                         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
             gt = np.einsum("ij,njk->nik", np.array([[0.0, -1.0, 0.0, 0.0],
                                                    [1.0, 0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 1.0, 0.0],
                                                    [0.0, 0.0, 0.0, 1.0]], dtype=np.float32), gt)
        interpolator = PosesInterpolator(gt, gt_t)
        if timestamps.min() < t_min or timestamps.max() > timestamps.max():
            logging.info("[NCLT] Found timestamps outside of the window of ground truth poses. "
                         "Timestamps will be clipped to the match the window.")
            timestamps = timestamps.clip(min=t_min, max=t_max)
        return interpolator(timestamps)

    def __len__(self):
        return self._size

    @staticmethod
    def read_velodyne_file(file: str):
        # Custom File reader
        binary = np.fromfile(file, dtype=np.int16)
        x = np.ascontiguousarray(binary[::4])
        y = np.ascontiguousarray(binary[1::4])
        z = np.ascontiguousarray(binary[2::4])
        x = x.astype(np.float32).reshape(-1, 1)
        y = y.astype(np.float32).reshape(-1, 1)
        z = z.astype(np.float32).reshape(-1, 1)
        x, y, z = _convert(x, y, z)
        # Flip to have z pointing up
        pc = np.concatenate([x, -y, -z], axis=1)
        return pc

    def __getitem__(self, idx: int):
        assert_debug(0 <= idx < self._size)
        data_dict = dict()
        pc_file = self.sequence_dir / "velodyne_sync" / str(self.velodyne_files[idx])
        numpy_pc = self.read_velodyne_file(str(pc_file))
        numpy_pc = numpy_pc[np.linalg.norm(numpy_pc, axis=-1) < 100.0]
        torch_pc = torch.from_numpy(numpy_pc).unsqueeze(0)
        vertex_map = self.projector.build_projection_map(torch_pc)[0]

        data_dict["numpy_pc"] = numpy_pc
        data_dict[self.vertex_map_key] = vertex_map

        if self._gt is not None:
            data_dict[self.gt_channel] = self._gt[idx]
        return data_dict


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class NCLTConfig(DatasetConfig):
    """A configuration object read from a yaml configuration file"""

    # ------------------
    # Required parameters
    root_dir: str = MISSING
    dataset: str = "nclt"

    # ------------------------------
    # Parameters with default values
    train_sequences: List[str] = field(default_factory=lambda: ["2012-01-22", "2012-02-02", "2012-02-04",
                                                                "2012-02-05", "2012-02-12", "2012-02-18",
                                                                "2012-02-19", "2012-03-17", "2012-03-25",
                                                                "2012-03-31"])
    test_sequences: List[str] = field(default_factory=lambda: ["2012-01-08", "2012-01-15"])
    eval_sequences: List[str] = field(default_factory=lambda: [])

    # LiDAR Spherical projection configuration
    lidar_height: int = 40
    lidar_width: int = 720
    up_fov: float = 30.0
    down_fov: float = -5.0


# Hydra -- stores a NCLTConfig `nclt` in the `dataset` group
cs = ConfigStore.instance()
cs.store(group="dataset", name="nclt", node=NCLTConfig)


# ----------------------------------------------------------------------------------------------------------------------
class NCLTDatasetLoader(DatasetLoader):
    """
    A Configuration for the NCLT Dataset

    The configuration allows to build sequence datasets from the different acquisition sequences
    Of the Dataset.

    The NCLT Dataset consists of 27 sequences acquired with a HDL-32 Velodyne LiDAR sensor.
    This configuration expects the Dataset to be installed at a `root_dir` location disk.
    Each sequence should have its own directory, and the corresponding `groundtruth.csv` should be located
    In the sequence's directory as follows :

    <root_dir>/2012-01-08/velodyne_sync/
                                        1335704127712909.bin
                                        1335704127712912.bin
                                        ...
                          groundtruth_2012-01-08.csv

    See:
     - http://robots.engin.umich.edu/nclt for the link to the Dataset's main page
    """

    def __init__(self, config: NCLTConfig, **kwargs):
        super().__init__(config)
        self.root_dir = Path(self.config.root_dir)

    def projector(self) -> SphericalProjector:
        lidar_height = self.config.lidar_height
        lidar_width = self.config.lidar_width
        up_fov = self.config.up_fov
        down_fov = self.config.down_fov
        return SphericalProjector(lidar_height, lidar_width, up_fov=up_fov, down_fov=down_fov)

    def sequences(self):
        assert_debug(self.root_dir.exists())
        train_sequences_ids = self.config.train_sequences
        test_sequences_ids = self.config.test_sequences
        eval_sequences_ids = self.config.eval_sequences
        vertex_map_key = self.config.vertex_map_key
        projector = self.projector()

        def __seqid_to_datasets(sequences):
            if sequences is None or len(sequences) == 0:
                return None
            return [NCLTSequenceDataset(str(self.root_dir), str(seq_id), vertex_map_key, projector)
                    for seq_id in sequences]

        train_datasets = __seqid_to_datasets(train_sequences_ids)
        test_datasets = __seqid_to_datasets(test_sequences_ids)
        eval_datasets = __seqid_to_datasets(eval_sequences_ids)

        return (train_datasets, train_sequences_ids), \
               (eval_datasets, eval_sequences_ids), \
               (test_datasets, test_sequences_ids), lambda x: x

    def get_ground_truth(self, sequence_name):
        _, _, gt = NCLTSequenceDataset.timestamps_files_and_gt(self.root_dir, sequence_name)
        return gt
