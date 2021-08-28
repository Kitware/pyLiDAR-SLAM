from slam.dataset.rosbag_dataset import _with_rosbag

if _with_rosbag:
    from slam.dataset.rosbag_dataset import RosbagConfig, RosbagDataset

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
    from slam.common.projection import SphericalProjector, SphericalProjector, SphericalProjector
    from slam.common.utils import assert_debug
    from slam.dataset import DatasetConfig
    from typing import Optional

    import numpy as np
    import numba as nb
    from scipy.spatial.transform.rotation import Rotation as R
    from enum import Enum

    from slam.dataset import DatasetLoader
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


    def llu_to_ecef(llu: np.ndarray):
        """Converts a LLU GPS position to ECEF frame"""
        ecef = np.zeros((3,), dtype=np.float64)
        a = 6378137.0
        b = 6356752.314
        lon = llu[0] * 3.1415926 / 180.0
        lat = llu[1] * 3.1415926 / 180.0
        alt = llu[2]
        n = a * a / np.sqrt(a * a * np.cos(lat) * np.cos(lat) + b * b * np.sin(lat) * np.sin(lat))
        Rx = (n + alt) * np.cos(lat) * np.cos(lon)
        Ry = (n + alt) * np.cos(lat) * np.sin(lon)
        Rz = (b * b / (a * a) * n + alt) * np.sin(lat)
        ecef[0] = Rx
        ecef[1] = Ry
        ecef[2] = Rz
        return ecef


    def ecef_to_enu(origin_llu, ecef):
        """Converts a ECEF GPS position to a LLU frame"""
        pi = 3.1415926
        DEG2RAD = pi / 180.0

        # Eigen::MatrixXd enu; // the enu for output
        enu = np.zeros((3,), dtype=np.float64)  # enu.resize(3, 1); // resize to 3X1
        # oxyz.resize(3, 1); // resize to 3X1
        # double x, y, z; // save the x y z in ecef
        x = ecef[0]
        y = ecef[1]
        z = ecef[2]

        # double ox, oy, oz; // save original reference position in ecef
        oxyz = llu_to_ecef(origin_llu)
        ox = oxyz[0]
        oy = oxyz[1]
        oz = oxyz[2]

        dx = x - ox
        dy = y - oy
        dz = z - oz

        # lonDeg, latDeg
        lonDeg = origin_llu[0]
        latDeg = origin_llu[1]
        lon = lonDeg * DEG2RAD
        lat = latDeg * DEG2RAD
        enu[0] = -np.sin(lon) * dx + np.cos(lon) * dy
        enu[1] = -np.sin(lat) * np.cos(lon) * dx - np.sin(lat) * np.sin(lon) * dy + np.cos(lat) * dz
        enu[2] = np.cos(lat) * np.cos(lon) * dx + np.cos(lat) * np.sin(lon) * dy + np.sin(lat) * dz
        return enu


    class UrbanLocoDataset(RosbagDataset):
        """Sequence of the UrbanLoco Dataset wrapping a Rosbag

        Note: As the dataset is a rosbag dataset, random access is not authorized
        """

        class ACQUISITION(Enum):
            HONG_KONG = 0,
            CALIFORNIA = 1

        def __init__(self, config: RosbagConfig, acquisition: ACQUISITION,
                     absolute_gt_poses: Optional[np.ndarray] = None, synchronise_azimuth: bool = True,
                     azimuth_bin: int = -179) -> object:
            super().__init__(config, config.file_path, self.pointcloud_topic(acquisition),
                             1, self._topics_mapping(acquisition))

            # Build the conversion from GPS coordinates (lat, long, alt) to global xyz
            self.acquisition = acquisition
            self.ground_truth_poses = absolute_gt_poses

            self.synchronise_azimuth = synchronise_azimuth
            self.azimuth_bin = azimuth_bin
            self.current_frame = []
            self.current_timestamps = []
            self.skip_frame = False
            self.use_first_id = False

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

        def estimate_timestamps(self, frame_index: int, pc: np.ndarray):
            if self.acquisition == self.ACQUISITION.CALIFORNIA:
                num_packets = pc.shape[0] / (12 * 32)
                _packet_ids = np.arange(num_packets).reshape(1, int(num_packets), 1)
                _packet_ids = _packet_ids.repeat(32, axis=0).repeat(12, axis=2).reshape(-1).astype(np.float64)
                timestamps = (_packet_ids - _packet_ids.min()) / (_packet_ids.max() - _packet_ids.min())
                return pc, timestamps + frame_index, _packet_ids
            else:
                thetas = np.arctan2(np.linalg.norm(pc[:, :2], axis=1), pc[:, 2])
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
                timestamps = (_packet_ids - t_min) / (t_max - t_min) + frame_index

                return pc[_filter], timestamps, _packet_ids

        def _save_topic(self, data_dict, key, topic, msg, timestamp, frame_index: int = -1, **kwargs):
            if "PointCloud2" in msg._type:
                data, timestamps = self.decode_pointcloud(msg, timestamp)
                pc, timestamps, _packet_ids = self.estimate_timestamps(frame_index, data)

                if self.synchronise_azimuth:
                    sorted_indices = np.argsort(timestamps)
                    pc = pc[sorted_indices]
                    timestamps = timestamps[sorted_indices]
                    _packet_ids = _packet_ids[sorted_indices]

                    azimuth_bins = (np.arctan2(pc[:, 1], pc[:, 0]) * 180 / np.pi).astype(np.int32)
                    indices = np.nonzero(azimuth_bins == self.azimuth_bin)[0]
                    first_id = indices[0]
                    last_id = indices[-1]

                    first_packet_id = _packet_ids[first_id]
                    last_packet_id = _packet_ids[last_id]

                    self.use_first_id = len(self.current_frame) > 0 and self.current_frame[0].shape[0] > 30000

                    set_current_frame: bool = True
                    if last_packet_id <= 1 and len(self.current_frame) == 0:
                        # No previous saved points : return the full pointcloud
                        current_frame_filter = np.ones((_packet_ids.shape[0],), dtype=np.bool)
                        set_current_frame = False
                    else:
                        current_frame_filter = _packet_ids <= (
                            first_packet_id if self.use_first_id else last_packet_id)

                    self.current_frame.append(pc[current_frame_filter])
                    self.current_timestamps.append(timestamps[current_frame_filter])

                    current_frame = np.concatenate(self.current_frame, axis=0)
                    current_timestamps = np.concatenate(self.current_timestamps, axis=0)

                    self.current_timestamps.clear()
                    self.current_frame.clear()
                    if set_current_frame:
                        self.current_frame.append(pc[~current_frame_filter])
                        self.current_timestamps.append(timestamps[~current_frame_filter])
                        if (self.use_first_id and abs(last_packet_id - first_packet_id) > 50) or \
                                (last_packet_id <= 1 and self.current_frame[0].shape[0] > 50000):
                            self.skip_frame = True  # The next frame returned in the remaining points

                    frame_size = current_frame.shape[0]
                    if frame_size < 3000:
                        print(f"[ERROR] {frame_size} is small")

                else:
                    current_frame = pc
                    current_timestamps = timestamps
                    self.skip_frame = False

                data_dict[key].append(current_frame)
                timestamps_key = f"{key}_timestamps"
                if timestamps_key not in data_dict:
                    data_dict[timestamps_key] = []
                data_dict[timestamps_key].append(current_timestamps)

            if "INSPVAX" in msg._type:
                roll = msg.roll
                pitch = msg.pitch
                yaw = msg.azimuth
                rotation = R.from_euler("ZYX", np.array([yaw, pitch, roll], dtype=np.float64)).as_matrix()
                pose = np.eye(4, dtype=np.float64)
                pose[:3, :3] = rotation

                latitude = msg.latitude
                longitude = msg.longitude
                altitude = msg.altitude
                llu = np.array([latitude, longitude, altitude])
                ypr = np.array([yaw, pitch, roll])

                data_dict[key].append((timestamp.secs * 10e9 + timestamp.nsecs, (llu, ypr)))
            return data_dict

        def __getitem__(self, index):
            if self.skip_frame:
                data_dict = dict()
                data_dict["numpy_pc"] = np.concatenate(self.current_frame, axis=0)
                data_dict["numpy_pc_timestamps"] = np.concatenate(self.current_timestamps, axis=0)
                self.current_timestamps.clear()
                self.current_frame.clear()
                self.skip_frame = False
                return data_dict

            data_dict = super().__getitem__(index)
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

        synchronise_azimuth: bool = True
        synchronisation_angle: int = -179


    class UrbanLocoDatasetLoader(DatasetLoader):
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
                init_enu = None
                init_llu = None
                for b_idx, data_dict in tqdm(enumerate(dataset), ncols=100, total=len(dataset), ascii=True):
                    if "odom" in data_dict:
                        poses_data = data_dict["gps_pose"]
                        for timestamp, pose_items in poses_data:
                            llu, ypr = pose_items
                            timestamps_odom_poses.append(timestamp)

                            longitude = llu[0] * np.pi / 180
                            latitude = llu[1] * np.pi / 180
                            altitude = llu[2] * np.pi / 180
                            yaw = ypr[0] * np.pi / 180
                            pitch = ypr[1] * np.pi / 180
                            roll = ypr[2] * np.pi / 180
                            R_enu = R.from_euler("zyx", np.array([-yaw, pitch, roll])).as_matrix()

                            if init_llu is None:
                                init_llu = llu

                            ecef = llu_to_ecef(np.array([longitude, latitude, altitude]))
                            enu = ecef_to_enu(init_llu, ecef)

                            if init_enu is None:
                                init_enu = enu

                            enu_to_enu0 = np.eye(4, dtype=np.float64)
                            enu_to_enu0[:3, 3] = (enu - init_enu).reshape(3)
                            enu_to_enu0[:3, :3] = R_enu

                            # Convert ENU to  NWU North / West / Up
                            enu_to_nwu = np.array([[0.0, 1.0, 0.0, 0.0],
                                                   [-1.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 1.0, 0.0],
                                                   [0.0, 0.0, 0.0, 1.0]])
                            nwu_pose = enu_to_nwu.dot(enu_to_enu0).dot(np.linalg.inv(enu_to_nwu))

                            odom_poses.append(nwu_pose)

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
                    dataset = UrbanLocoDataset(config, acquisition, poses,
                                               synchronise_azimuth=self.config.synchronise_azimuth,
                                               azimuth_bin=int(self.config.synchronisation_angle))
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
