from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# Hydra and OmegaConf
from hydra.core.config_store import ConfigStore
from hydra.conf import dataclass, MISSING, field

# Project Imports
from slam.common.projection import SphericalProjector
from slam.common.utils import assert_debug
from slam.dataset.configuration import DatasetLoader, DatasetConfig
from slam.eval.eval_odometry import compute_relative_poses


def kitti_read_scan(file_path: str) -> np.ndarray:
    """
    Reads and Returns a Lidar Scan from kitti's benchmark

    Args:
        file_path (str): The file path of kitti's binary file

    Returns:
        A `(N, 4)` `np.ndarray` point cloud of `N` points and 4 channels (in order) : X, Y, Z, r (reflectance)
        Read from the file
    """
    try:
        scan = np.fromfile(file_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return scan
    except (Exception, ValueError) as e:
        print(f"Error reading scan : {file_path}")
        raise e


def read_calib_file(file_path: str) -> dict:
    """
    Reads a calibration file from KITTI

    KITTI's calibration files map keys to transform / projection / calibration matrices,

    Args:
        file_path (str): The file path of kitti's calibration file

    Returns
        A dictionary mapping a key to a np.ndarray of the calibration matrix
        Each matrix has shape n, where n is the number of float read for the
        corresponding line.
    """
    calib_dict = {}
    with open(file_path, "r") as calib_file:
        for line in calib_file.readlines():
            tokens = line.split(" ")
            if tokens[0] == "calib_time:":
                continue
            # Only read with float data
            if len(tokens) > 0:
                values = [float(token) for token in tokens[1:]]
                values = np.array(values, dtype=np.float32)

                # The format in KITTI's file is <key>: <f1> <f2> <f3> ...\n -> Remove the ':'
                key = tokens[0][:-1]
                calib_dict[key] = values
    return calib_dict


def read_ground_truth_file(file_path: str) -> np.ndarray:
    """
    Reads a ground truth file from KITTI odometry's benchmark

    Returns poses in the coordinate system of the left camera
    With the z axis facing forward, use the calib.txt of the
    sequence, to obtain the calib file

    Parameters
    ----------
    file_path : str
        The path to the ground truth file
    """
    df = pd.read_csv(file_path, sep=" ", header=None)
    poses = df.values  # [N, 12] array

    n = poses.shape[0]
    poses = np.concatenate((poses, np.zeros((n, 3), dtype=np.float32), np.ones((n, 1), dtype=np.float32)), axis=1)
    poses = poses.reshape((n, 4, 4))  # [N, 4, 4]
    return poses


class KITTIOdometrySequence(Dataset):
    """
    Dataset for a Sequence of KITTI odometry's benchmark

    Attributes:
        sequences_root_dir (str): The path to KITTI odometry benchmark's data
        sequence_id (str): The name id of the sequence in [00-21]
        corrected_lidar_channel (str): The key in the dictionary for the vertex map
        lidar_projector (SphericalProjector): The Spherical Projector, which projects pointclouds in the image plane
        ground_truth_channel (Optional[str]): The key in the dictionary for the ground truth absolute pose
        with_numpy (bool): Whether to add the numpy pc to the data_dict
    """

    def __init__(self,
                 sequences_root_dir: str,
                 sequence_id: str,
                 corrected_lidar_channel: str = "vertex_map",
                 lidar_projector: SphericalProjector = None,
                 ground_truth_channel: Optional[str] = None,
                 unrectified_dir: Optional[str] = None,
                 unrectified_lidar_channel: str = "raw_vertex_map",
                 with_numpy_pc: bool = False):

        self.sequence_dir: Path = Path(sequences_root_dir)
        self.sequence_id: str = sequence_id
        self.ground_truth_channel = ground_truth_channel
        self.corrected_lidar_channel = corrected_lidar_channel
        self.id = self.sequence_id
        self.lidar_projector = lidar_projector
        self._with_numpy_pc = with_numpy_pc

        self.size: int = self.__mapping_sequence_id_to_info[sequence_id][2]

        # Initialize datasets paths
        self.velodyne_path: Path = self.sequence_dir / "sequences" / sequence_id / "velodyne"
        self.raw_velodyne_path: Optional[Path] = None
        self.raw_start = -1
        self._with_raw_lidar = False
        self.raw_lidar_key = unrectified_lidar_channel
        if unrectified_dir is not None:
            self._with_raw_lidar = True
            folder, start, end = self.__mapping_sequence_id_to_info[sequence_id]
            if folder is not None:
                self.raw_start = start
                self.raw_velodyne_path = Path(unrectified_dir) / f"{folder}_extract" / "velodyne_points" / "data"

        # Read calibration data
        calibration_file_path: Path = self.sequence_dir / "sequences" / sequence_id / "calib.txt"
        assert_debug(calibration_file_path.exists() and calibration_file_path.is_file())
        calibration_data = read_calib_file(str(calibration_file_path))
        self.calibration_matrices: dict = {}
        for id, array in calibration_data.items():
            matrix = None
            if id == "Tr":
                matrix = np.concatenate((np.reshape(array, (3, 4)), np.array([[0, 0, 0, 1]], dtype=np.float32)), axis=0)

            if matrix is not None:
                self.calibration_matrices[id] = matrix

        # Read ground truth
        self.poses_gt: (None, np.ndarray) = None
        if self.ground_truth_channel:
            gt_file = self.sequence_dir / "poses" / f"{sequence_id}.txt"
            if gt_file.exists() and gt_file.is_file():
                poses_gt_in_image = read_ground_truth_file(str(gt_file))
                self.poses_gt = self.__lidar_pose_gt(poses_gt_in_image)

    __mapping_sequence_id_to_info: dict = {
        # '<seq_id>' : ('<raw_drive_folder>', <raw_start_index:int>, <size:int>)
        "00": ("2011_10_03/2011_10_03_drive_0027", 0, 4541),
        "01": ("2011_10_03/2011_10_03_drive_0042", 0, 1101),
        "02": ("2011_10_03/2011_10_03_drive_0034", 0, 4661),
        "03": (None, 0, 801),
        "04": ("2011_09_30/2011_09_30_drive_0016", 0, 271),
        "05": ("2011_09_30/2011_09_30_drive_0018", 0, 2761),
        "06": ("2011_09_30/2011_09_30_drive_0020", 0, 1101),
        "07": ("2011_09_30/2011_09_30_drive_0027", 0, 1101),
        "08": ("2011_09_30/2011_09_30_drive_0028", 1100, 4071),
        "09": ("2011_09_30/2011_09_30_drive_0033", 0, 1591),
        "10": ("2011_09_30/2011_09_30_drive_0034", 0, 1201),
        "11": (None, 0, 921),
        "12": (None, 0, 1061),
        "13": (None, 0, 3281),
        "14": (None, 0, 631),
        "15": (None, 0, 1901),
        "16": (None, 0, 1731),
        "17": (None, 0, 491),
        "18": (None, 0, 1801),
        "19": (None, 0, 4981),
        "20": (None, 0, 831),
        "21": (None, 0, 2721)
    }

    def __raw_id_from_frame_idx(self, idx):
        return idx + self.__mapping_sequence_id_to_info[self.sequence_id][1]

    def __lidar_pose_gt(self, poses_gt: np.ndarray):
        if "Tr" in self.calibration_matrices:
            # Apply Tr on poses_gt
            tr = self.calibration_matrices["Tr"]
            left = np.einsum("...ij,...jk->...ik", np.linalg.inv(tr), poses_gt)
            right = np.einsum("...ij,...jk->...ik", left, tr)
            return right
        return poses_gt

    def __len__(self):
        return self.size

    @staticmethod
    def correct_scan(scan: np.ndarray):
        """
        Corrects the calibration of KITTI's HDL-64 scan
        """
        xyz = scan[:, :3]
        n = scan.shape[0]
        z = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (n, 1))
        axes = np.cross(xyz, z)
        # Normalize the axes
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        theta = 0.205 * np.pi / 180.0

        # Build the rotation matrix for each point
        c = np.cos(theta)
        s = np.sin(theta)

        u_outer = axes.reshape(n, 3, 1) * axes.reshape(n, 1, 3)
        u_cross = np.zeros((n, 3, 3), dtype=np.float32)
        u_cross[:, 0, 1] = -axes[:, 2]
        u_cross[:, 1, 0] = axes[:, 2]
        u_cross[:, 0, 2] = axes[:, 1]
        u_cross[:, 2, 0] = -axes[:, 1]
        u_cross[:, 1, 2] = -axes[:, 0]
        u_cross[:, 2, 1] = axes[:, 0]

        eye = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
        rotations = c * eye + s * u_cross + (1 - c) * u_outer
        corrected_scan = np.einsum("nij,nj->ni", rotations, xyz)

        return corrected_scan

    def __getitem__(self, idx) -> dict:
        """
        Returns:
            A dictionary with the mapping defined in the constructor
        """
        assert_debug(idx < self.size)
        data_dict = {}
        if self.corrected_lidar_channel:
            scan_path = self.velodyne_path / f"{idx:06}.bin"
            assert_debug(scan_path.exists() and scan_path.is_file(), f"The file {scan_path} does not exist")
            scan = kitti_read_scan(str(scan_path))
            # Apply Rectification on the scan
            scan = self.correct_scan(scan)
            if self._with_numpy_pc:
                data_dict["numpy_pc"] = scan.copy()
            scan = torch.from_numpy(scan[:, :3]).unsqueeze(0)
            data_dict[self.corrected_lidar_channel] = self.lidar_projector.build_projection_map(scan)[0]

        if self._with_raw_lidar:
            scan_path = self.raw_velodyne_path / f"{self.raw_start + idx:010}.txt"
            raw_scan = pd.read_csv(str(scan_path), sep=" ", header=None).values.astype(np.float32)
            # Apply Rectification on the scan
            # raw_scan = self._correct_scan(raw_scan)
            if self._with_numpy_pc:
                data_dict["raw_numpy_pc"] = raw_scan
            torch_scan = torch.from_numpy(raw_scan[:, :3]).unsqueeze(0)
            vmap = self.lidar_projector.build_projection_map(torch_scan)[0]
            data_dict[self.raw_lidar_key] = vmap

            c, h, w = vmap.shape
            timestamps_map = (np.arange(w, dtype=np.float32) / w).reshape(1, 1, w) - 0.5
            timestamps_map = timestamps_map.repeat(h, axis=1)
            numpy_timestamps = timestamps_map.reshape(-1, 1)
            raw_numpy_pc = vmap.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
            filter = np.linalg.norm(raw_numpy_pc, axis=-1) != 0.0

            raw_numpy_pc = raw_numpy_pc[filter]
            numpy_timestamps = numpy_timestamps[filter]
            data_dict["timestamps_map"] = timestamps_map
            if self._with_numpy_pc:
                data_dict["raw_numpy_pc"] = raw_numpy_pc
                data_dict["raw_numpy_timestamps"] = numpy_timestamps

        if self.ground_truth_channel and self.poses_gt is not None:
            data_dict[self.ground_truth_channel] = torch.from_numpy(self.poses_gt[idx, :, :])

        return data_dict


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class KITTIConfig(DatasetConfig):
    """A configuration object read from a yaml conf"""
    # -------------------
    # Required Parameters
    kitti_sequence_dir: str = MISSING
    dataset: str = "kitti"

    # ------------------------------
    # Parameters with default values
    kitti_raw_dir: Optional[str] = None  # By default the dataset is expected to be at the root of the project
    lidar_key: str = "vertex_map"
    lidar_height: int = 64
    lidar_width: int = 1024
    up_fov: int = 3
    down_fov: int = -24
    train_sequences: list = field(
        default_factory=lambda: ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"])
    test_sequences: list = field(default_factory=lambda: [f"{i:02}" for i in range(22)])
    eval_sequences: list = field(default_factory=lambda: ["09", "10"])


# Hydra -- stores a KITTIConfig `kitti` in the `dataset` group
cs = ConfigStore.instance()
cs.store(group="dataset", name="kitti", node=KITTIConfig)


# ----------------------------------------------------------------------------------------------------------------------
class KITTIDatasetLoader(DatasetLoader):
    """
    Configuration for KITTI's dataset odometry benchmark
    see http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    """

    def __init__(self, config: KITTIConfig):
        super().__init__(config)
        self.odometry_sequence_dir = Path(self.config.kitti_sequence_dir)
        self.raw_dir = self.config.kitti_raw_dir
        assert_debug(self.odometry_sequence_dir.exists())

    def projector(self) -> SphericalProjector:
        """Default SphericalProjetor for KITTI (projection of a pointcloud into a Vertex Map)"""
        lidar_height = self.config.lidar_height
        lidar_with = self.config.lidar_width
        up_fov = self.config.up_fov
        down_fov = self.config.down_fov
        # Vertex map projector
        projector = SphericalProjector(lidar_height, lidar_with, 3, up_fov, down_fov)
        return projector

    def get_ground_truth(self, sequence_name):
        """Returns the ground truth poses associated to a sequence of KITTI's odometry benchmark"""
        if sequence_name in [f"{i:02}" for i in range(11)]:
            poses = read_ground_truth_file(str(self.odometry_sequence_dir / "poses" / f"{sequence_name}.txt")).astype(
                np.float64)
            calib_file = read_calib_file(str(self.odometry_sequence_dir / "sequences" / sequence_name / "calib.txt"))
            _tr = calib_file["Tr"].reshape(3, 4)
            tr = np.eye(4, dtype=np.float64)
            tr[:3, :4] = _tr

            left = np.einsum("...ij,...jk->...ik", np.linalg.inv(tr), poses)
            right = np.einsum("...ij,...jk->...ik", left, tr)
            relative = compute_relative_poses(right)
            return relative
        return None

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

        lidar_channel = self.config.lidar_key
        gt_pose_channel = self.config.absolute_gt_key

        # Sets the path of the kitti benchmark
        train_sequence_ids = self.config.train_sequences
        eval_sequence_ids = self.config.eval_sequences
        test_sequence_ids = self.config.test_sequences

        def __get_datasets(sequences: list):
            if sequences is None or len(sequences) == 0:
                return None

            datasets = [KITTIOdometrySequence(
                str(self.odometry_sequence_dir),
                sequence_id,
                lidar_channel,
                self.projector(), gt_pose_channel, unrectified_dir=self.raw_dir,
                with_numpy_pc=self.config.with_numpy_pc) for sequence_id in sequences]
            return datasets

        train_datasets = __get_datasets(train_sequence_ids)
        eval_datasets = __get_datasets(eval_sequence_ids)
        test_datasets = __get_datasets(test_sequence_ids)

        return (train_datasets, train_sequence_ids), \
               (eval_datasets, eval_sequence_ids), \
               (test_datasets, test_sequence_ids), lambda x: x
