"""
UrbanLoco Dataset (cf https://advdataset2019.wixsite.com/urbanloco/data-collection-platforma)

The dataset consists of a set of 11 rosbags containing multiple data (including GPS ground truth)

TODO:
    - (pdell-kitware) The ground truth is currently wrong (need a bit more work to extract the correct pose matrix)
"""
import logging
from pathlib import Path

from hydra.conf import field, dataclass
from omegaconf import MISSING
from tqdm import tqdm

from slam.common.io import read_poses_from_disk, write_poses_to_disk
from slam.common.pose import PosesInterpolator
from slam.common.projection import SphericalProjector
from slam.common.utils import assert_debug
from slam.dataset import DatasetConfig
from typing import Optional

import numpy as np
import numba as nb
from scipy.spatial.transform.rotation import Rotation as R
from enum import Enum

from slam.dataset import DatasetLoader
from slam.dataset.rosbag_dataset import RosbagConfig, RosbagDataset
from slam.eval.eval_odometry import compute_relative_poses


@nb.jit(nopython=True)
def compute_ring_ids(theta_bins, unique):
    """Compute ring ids by grouping points by polar angle bins (in spherical projection)"""
    ring_ids = -1 * np.ones_like(theta_bins)
    # convert thetas_bins to ring indices
    for idx in range(theta_bins.shape[0]):
        value = theta_bins[idx]
        for rid in range(32):
            bin_value = unique[rid]
            if value == bin_value:
                ring_ids[idx] = rid
                break

    return ring_ids


@nb.jit(nopython=True)
def packet_ids(ring_ids: np.ndarray):
    """Extract velodyne packet_ids from the array of ring ids (which can be used to simulate timestamps)"""
    array = -1 * np.ones((ring_ids.shape[0],), dtype=np.int64)
    ring_ids_set = set()
    packet_id = 0
    col_id = 0
    for idx in range(ring_ids.shape[0]):
        ring_id = int(ring_ids[idx])
        if ring_id < 0:
            continue

        if ring_id in ring_ids_set:
            # Finished a column
            col_id += 1
            ring_ids_set.clear()
            if col_id == 12:
                # Finished a packet
                col_id = 0
                packet_id += 1

        ring_ids_set.add(ring_id)
        array[idx] = packet_id
    return array


_california_ext_to_lidar = np.array([[0., -1., 0., -5.245e-01],
                                     [-1., 0., 0., 1.06045],
                                     [0., 0., -1., 7.98576e-01],
                                     [0, 0, 0, 1]], dtype=np.float64)

_hk_body_to_lidar = np.array([[2.67949e-08, -1, 0, 0],
                              [1, 2.67949e-08, 0, 0],
                              [0, 0, 1, -0.28],
                              [0., 0., 0., 1]], dtype=np.float64)

_hk_body_to_span = np.array([[2.67949e-08, -1, 0, 0],
                             [1, 2.67949e-08, 0, 0],
                             [0, 0, 1, -0.36],
                             [0., 0., 0., 1]], dtype=np.float64)

_hk_span_to_lidar = _hk_body_to_lidar.dot(np.linalg.inv(_hk_body_to_span))


class UrbanLocoDataset(RosbagDataset):
    """Sequence of the UrbanLoco Dataset wrapping a Rosbag

    Note: As the dataset is a rosbag dataset, random access is not authorized
    """

    class ACQUISITION(Enum):
        HONG_KONG = 0,
        CALIFORNIA = 1

    __span_to_lidar_california = np.array([[0., -1., 0., -5.245e-01],
                                           [-1., 0., 0., 1.06045],
                                           [0., 0., -1., 7.98576e-01],
                                           [0, 0, 0, 1]], dtype=np.float64)

    __span_to_lidar_hk = np.array([[2.67949e-08, -1, 0, 0],
                                   [1, 2.67949e-08, 0, 0],
                                   [0, 0, 1, -0.36],
                                   [0., 0., 0., 1]], dtype=np.float64)

    def __init__(self, config: RosbagConfig, acquisition: ACQUISITION,
                 absolute_gt_poses: Optional[np.ndarray] = None) -> object:
        super().__init__(config, config.file_path, self.pointcloud_topic(acquisition),
                         1, self._topics_mapping(acquisition))

        # Build the conversion from GPS coordinates (lat, long, alt) to global xyz
        self.acquisition = acquisition
        self.ground_truth_poses = absolute_gt_poses

    @staticmethod
    def pointcloud_topic(acquisition: ACQUISITION):
        if acquisition == UrbanLocoDataset.ACQUISITION.HONG_KONG:
            return "/velodyne_points"
        else:
            return "/rslidar_points"

    @staticmethod
    def ground_truth_topic():
        return "/novatel_data/inspvax"

    @staticmethod
    def _topics_mapping(acquisition: ACQUISITION):
        return {UrbanLocoDataset.ground_truth_topic(): DatasetLoader.absolute_gt_key(),
                (UrbanLocoDataset.pointcloud_topic(acquisition)): "numpy_pc",
                "/novatel_data/inspvax": "gps_pose",
                "/navsat/odom": "odom"}

    def span_to_lidar(self):
        if self.acquisition == self.ACQUISITION.HONG_KONG:
            return self.__span_to_lidar_hk
        else:
            return self.__span_to_lidar_california

    def llu_to_ecef(self, llu: np.ndarray):
        ecef = np.zeros((3,), dtype=np.float64)
        a = 6378137.0
        b = 6356752.314

        lon = llu[0] * 3.1415926 / 180.0
        lat = llu[1] * 3.1415926 / 180.0
        alt = llu[2]
        n = a * a / np.sqrt(a * a * np.cos(lat) * np.cos(lat) + b * b * np.sin(lat) * np.sin(lat))
        Rx = (n + alt) * np.cos(lat) * np.cos(lon)
        Ry = (n + alt) * np.cos(lat) * np.sin(lon)
        Rz = (b * b / (a * a) * n + alt) * np.sin(lat);
        ecef[0] = Rx
        ecef[1] = Ry
        ecef[2] = Rz
        return ecef

    def decode_data(self, _key: str, data_dict: dict, msg_list: list, **kwargs):
        super().decode_data(_key, data_dict, msg_list, **kwargs)
        if len(msg_list) == 0:
            return
        elem = msg_list[0][1]

        if "INSPVAX" in elem._type:
            odom_items = []
            for timestamp, msg in msg_list:
                roll = msg.roll / 180 * np.pi
                pitch = msg.pitch / 180 * np.pi
                yaw = msg.azimuth / 180 * np.pi
                rotation = R.from_euler("ZYX", np.array([yaw, pitch, roll], dtype=np.float64)).as_matrix()
                pose = np.eye(4, dtype=np.float64)
                pose[:3, :3] = rotation

                latitude = msg.latitude
                longitude = msg.longitude
                altitude = msg.altitude
                ecef = self.llu_to_ecef(np.array([latitude, longitude, altitude]))
                pose[:3, 3] = ecef

                span_to_lidar = self.span_to_lidar()
                pose = pose.dot(span_to_lidar)
                odom_items.append((timestamp.secs * 10e9 + timestamp.nsecs, pose))

            data_dict["gps_pose"] = odom_items

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)

        del data_dict["numpy_pc_timestamps"]

        if self.acquisition == self.ACQUISITION.CALIFORNIA:
            pc = data_dict["numpy_pc"]

            num_packets = pc.shape[0] / (12 * 32)
            timestamps = np.arange(num_packets).reshape(1, int(num_packets), 1)
            timestamps = timestamps.repeat(32, axis=0).repeat(12, axis=2).reshape(-1).astype(np.float64)
            timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
            data_dict["numpy_pc_timestamps"] = timestamps + index
        else:
            numpy_pc = data_dict["numpy_pc"]
            thetas = np.arctan2(np.linalg.norm(numpy_pc[:, :2], axis=1), numpy_pc[:, 2])
            bin_size = 0.1 / 180. * np.pi
            thetas_bins = (thetas / bin_size).astype(np.int32)

            unique, indices, counts = np.unique(thetas_bins, return_index=True, return_counts=True)
            _sorted_indices = np.argsort(- counts)
            _sorted_indices = _sorted_indices[:32]
            unique = unique[_sorted_indices]
            ring_ids = compute_ring_ids(thetas_bins, unique)
            _packet_ids = packet_ids(ring_ids)
            _filter = _packet_ids >= 0
            _packet_ids = _packet_ids[_filter]
            t_min = _packet_ids.min()
            t_max = _packet_ids.max()
            timestamps = (_packet_ids - t_min) / (t_max - t_min) + index

            data_dict["numpy_pc"] = numpy_pc[_filter]
            data_dict["numpy_pc_timestamps"] = timestamps

        if self.ground_truth_poses is not None:
            pose_gt = self.ground_truth_poses[index]

            if self.acquisition == self.ACQUISITION.HONG_KONG:
                lidar_to_lidar_0 = pose_gt
            else:
                lidar_to_lidar_0 = pose_gt
            data_dict["absolute_pose_gt"] = lidar_to_lidar_0

        return data_dict


@dataclass
class UrbanLocoConfig(DatasetConfig):
    dataset: str = "urban_loco"
    root_dir: str = MISSING

    train_sequences: list = field(
        default_factory=lambda: ["CAChinaTown", "CAColiTower", "CALombardStreet", "CAGoldenBridge",
                                 "CABayBridge", "CAMarketStreet", "CARussianHill", "HK-Data20190426-2",
                                 "HK-Data20190426-2", "HK-Data20190426-1",
                                 "HK-Data20190316-2", "HK-Data20190316-1"])
    test_sequences: list = field(default_factory=lambda: [])
    eval_sequences: list = field(default_factory=lambda: [])

    lidar_height: int = 32
    lidar_width: int = 1024
    up_fov: int = 25
    down_fov: int = -24


class UrbanLocoDatasetLoader(DatasetLoader):

    def span_to_lidar_calib(self, acquisition: UrbanLocoDataset.ACQUISITION):
        if acquisition == UrbanLocoDataset.ACQUISITION.CALIFORNIA:
            return self.__california_ext_to_lidar
        else:
            return self.__hk_body_to_lidar.dot(np.linalg.inv(self.__hk_body_to_span))

    __seqname_to_filename = {
        "CABayBridge": "CA-20190828151211_blur_align.bag",
        "CAMarketStreet": "CA-20190828155828_blur_align.bag",
        "CARussianHill": "CA-20190828173350_blur_align.bag",
        "CAChinaTown": "CA-20190828180248_blur_align.bag",
        "CAColiTower": "CA-20190828184706_blur_align.bag",
        "CALombardStreet": "CA-20190828190411_blur_align.bag",
        "CAGoldenBridge": "CA-20190828191451_blur_align.bag",
        "HK-Data20190426-2": "20190331WH.bag",
        "HK-Data20190426-1": "20190331HH.bag",
        "HK-Data20190316-2": "20190331_NJ_LL.bag",
        "HK-Data20190316-1": "20190331_NJ_SL.bag"
    }

    __seqname_to_acquisition = {
        "CABayBridge": UrbanLocoDataset.ACQUISITION.CALIFORNIA,
        "CAMarketStreet": UrbanLocoDataset.ACQUISITION.CALIFORNIA,
        "CARussianHill": UrbanLocoDataset.ACQUISITION.CALIFORNIA,
        "CAChinaTown": UrbanLocoDataset.ACQUISITION.CALIFORNIA,
        "CAColiTower": UrbanLocoDataset.ACQUISITION.CALIFORNIA,
        "CALombardStreet": UrbanLocoDataset.ACQUISITION.CALIFORNIA,
        "CAGoldenBridge": UrbanLocoDataset.ACQUISITION.CALIFORNIA,
        "HK-Data20190426-2": UrbanLocoDataset.ACQUISITION.HONG_KONG,
        "HK-Data20190426-1": UrbanLocoDataset.ACQUISITION.HONG_KONG,
        "HK-Data20190316-2": UrbanLocoDataset.ACQUISITION.HONG_KONG,
        "HK-Data20190316-1": UrbanLocoDataset.ACQUISITION.HONG_KONG
    }

    def __init__(self, config: UrbanLocoConfig, **kwargs):
        super().__init__(config)
        self.root_dir = Path(config.root_dir)

    @classmethod
    def groundtruth_filename(cls, sequence: str):
        assert_debug(sequence in cls.__seqname_to_filename, "Unrecognised sequence from UrbanLoco dataset")
        return f"{sequence}.poses.txt"

    def generate_ground_truth(self, sequences: list):
        """Estimates the ground truth poses for the LiDAR frames

        The output files are saved as text file under self.root_dir / <sequence_name>.poses.txt

        Note: That this will require to play all the rosbag in full.
              So this will very likely be very long
        """
        for sequence in sequences:
            assert_debug(sequence in self.__seqname_to_filename)
            rosbag_path = self.root_dir / self.__seqname_to_filename[sequence]
            if not rosbag_path.exists():
                logging.warning(f"[UrbanLocoDataset]Could not find the rosbag {str(rosbag_path)}")
                continue
            config = self.rosbag_config(sequence)
            acquisition = self.__seqname_to_acquisition[sequence]
            dataset = UrbanLocoDataset(config, acquisition, None)

            timestamps_pointclouds = []
            timestamps_odom_poses = []
            odom_poses = []

            timestamp_0 = None
            for b_idx, data_dict in tqdm(enumerate(dataset), ncols=100, total=len(dataset), ascii=True):
                if "odom" in data_dict:
                    poses_data = data_dict["odom"]
                    for timestamp, pose in poses_data:
                        timestamps_odom_poses.append(timestamp)
                        odom_poses.append(pose)

                if "numpy_pc_timestamps" in data_dict:
                    timestamps = data_dict["numpy_pc_timestamps"]
                    timestamp_max = timestamps.max()
                    if timestamp_0 is None:
                        timestamp_0 = timestamp_max
                    timestamps_pointclouds.append(timestamp_max)

            timestamps_pointclouds = np.array(timestamps_pointclouds).reshape(-1)  # [N]
            timestamps_odom_poses = np.array(timestamps_odom_poses).reshape(-1)  # [N]
            odom_poses = np.array(odom_poses)  # [N, 4, 4]

            interpolator = PosesInterpolator(odom_poses, timestamps_odom_poses)
            span_lidar_poses = interpolator(timestamps_pointclouds)
            span_lidar_poses = np.einsum("ij,njk->nik", np.linalg.inv(span_lidar_poses[0]), span_lidar_poses)

            poses_filename = str(self.root_dir / self.groundtruth_filename(sequence))
            write_poses_to_disk(poses_filename, span_lidar_poses)

    def rosbag_config(self, sequence_name: str) -> RosbagConfig:
        """Returns the Rosbag config for the given sequence"""
        assert isinstance(self.config, UrbanLocoConfig)
        assert_debug(sequence_name in self.__seqname_to_filename)
        rosbag_file = self.root_dir / self.__seqname_to_filename[sequence_name]
        assert_debug(rosbag_file.exists(),
                     f"The rosbag for sequence `{sequence_name}` does not exist at {str(rosbag_file)}")
        config = RosbagConfig()
        config.file_path = rosbag_file

        config.lidar_width = self.config.lidar_width
        config.lidar_height = self.config.lidar_height
        config.up_fov = self.config.up_fov
        config.down_fov = self.config.down_fov
        config.accumulate_scans = False
        config.frame_size = 1

        return config

    def projector(self) -> SphericalProjector:
        """Default SphericalProjetor for UrbanLoco dataset (projection of a pointcloud into a Vertex Map)"""
        assert isinstance(self.config, UrbanLocoConfig)
        lidar_height = self.config.lidar_height
        lidar_with = self.config.lidar_width
        up_fov = self.config.up_fov
        down_fov = self.config.down_fov
        # Vertex map projector
        projector = SphericalProjector(lidar_height, lidar_with, 3, up_fov, down_fov)
        return projector

    def sequences(self):
        # Sets the path of the kitti benchmark
        assert isinstance(self.config, UrbanLocoConfig)
        train_sequence_ids = [str(_id) for _id in self.config.train_sequences]
        eval_sequence_ids = [str(_id) for _id in self.config.eval_sequences]
        test_sequence_ids = [str(_id) for _id in self.config.test_sequences]

        def __get_datasets(sequences: list):
            if sequences is None or len(sequences) == 0:
                return None

            datasets = []
            for sequence in sequences:
                assert_debug(sequence in self.__seqname_to_acquisition, f"The sequence {sequence} does not exist")
                acquisition = self.__seqname_to_acquisition[sequence]
                config = self.rosbag_config(sequence)
                poses = self.get_ground_truth(sequence, relative=False)
                dataset = UrbanLocoDataset(config, acquisition, poses)
                datasets.append(dataset)

            return datasets

        train_datasets = __get_datasets(train_sequence_ids)
        eval_datasets = __get_datasets(eval_sequence_ids)
        test_datasets = __get_datasets(test_sequence_ids)

        return (train_datasets, train_sequence_ids), \
               (eval_datasets, eval_sequence_ids), \
               (test_datasets, test_sequence_ids), lambda x: x

    def get_ground_truth(self, sequence_name: str, relative: bool = True):
        gt_filename = self.groundtruth_filename(sequence_name)
        file_path = self.root_dir / gt_filename
        if file_path.exists():
            absolute_poses = read_poses_from_disk(file_path)
            absolute_poses = np.einsum("ij,njk->nik", np.linalg.inv(absolute_poses[0]), absolute_poses)
            return compute_relative_poses(absolute_poses) if relative else absolute_poses
        logging.warning("[URBAN LOCO]The ground truth for sequence was not found.")
        return None
