import logging
from dataclasses import MISSING
from pathlib import Path
from typing import Optional, Tuple
import os

from torch.utils.data import IterableDataset
import numpy as np

from hydra.conf import field, dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from slam.common.projection import SphericalProjector
from slam.common.utils import assert_debug, remove_nan
from slam.dataset import DatasetLoader, DatasetConfig

try:
    import rosbag
    import sensor_msgs.point_cloud2 as pc2
    from sensor_msgs.msg import PointCloud2, PointField

    _with_rosbag = True
except ImportError:
    _with_rosbag = False

if _with_rosbag:

    @dataclass
    class RosbagConfig(DatasetConfig):
        """Config for a Rosbag Dataset"""
        dataset: str = "rosbag"
        file_path: str = field(
            default_factory=lambda: "" if not "ROSBAG_PATH" in os.environ else os.environ["ROSBAG_PATH"])
        main_topic: str = "numpy_pc"  # The Key of the main topic (which determines the number of frames)
        xyz_fields: str = "xyz"

        accumulate_scans: bool = False  # Whether to accumulate the pointcloud messages (in case of raw sensor data)
        frame_size: int = 60  # The number of accumulated message which constitute a frame

        topic_mapping: dict = field(default_factory=lambda: {})

        lidar_height: int = 720
        lidar_width: int = 720
        up_fov: float = 45.
        down_fov: float = -45.


    class RosbagDataset(IterableDataset):
        """A Dataset which wraps a RosBag

        Note:
            The dataset can only read data sequentially, and will raise an error when two calls are not consecutives

        Args:
            file_path (str): The path on disk to the rosbag
            main_topic (str): The name of the main topic (which sets the number of frames to be extracted)
            frame_size (int): The number of messages to accumulate in a frame
            topic_mapping (dict): The mapping topic name to key in the data_dict
        """

        def __init__(self, config: RosbagConfig, file_path: str, main_topic: str, frame_size: int,
                     topic_mapping: Optional[dict] = None):
            self.config = config
            self.rosbag = None
            assert_debug(Path(file_path).exists(), f"The path to {file_path} does not exist.")
            logging.info(f"Loading ROSBAG {file_path}")
            self.rosbag = rosbag.Bag(file_path, "r")
            logging.info(f"Done.")

            self.topic_mapping = topic_mapping if topic_mapping is not None else {}
            if main_topic not in self.topic_mapping:
                self.topic_mapping[main_topic] = "numpy_pc"

            topic_info = self.rosbag.get_type_and_topic_info()
            for topic in self.topic_mapping:
                assert_debug(topic in topic_info.topics,
                             f"{topic} is not a topic of the rosbag "
                             f"(existing topics : {list(topic_info.topics.keys())}")

            self.main_topic = main_topic
            self._frame_size: int = frame_size if self.config.accumulate_scans else 1
            self._len = self.rosbag.get_message_count(self.main_topic) // self._frame_size

            self.__idx = 0
            self._topics = list(topic_mapping.keys())
            self.__iter = None

        def __iter__(self):
            self.__iter = self.rosbag.read_messages(self._topics)
            self.__idx = 0
            return self

        @staticmethod
        def decode_pointcloud(msg: pc2.PointCloud2, timestamp, xyz_fieldname: str = "xyz") -> Tuple[
            Optional[np.ndarray], Optional[np.ndarray]]:
            assert_debug("PointCloud2" in msg._type)
            pc = np.array(list(pc2.read_points(msg, field_names=xyz_fieldname)))
            timestamps = np.ones((pc.shape[0],),
                                 dtype=np.float64) * (float(timestamp.secs * 10e9) + timestamp.nsecs)
            return pc, timestamps

        def aggregate_messages(self, data_dict: dict):
            """Aggregates the point clouds of the main topic"""
            main_key = self.topic_mapping[self.main_topic]
            pcs = data_dict[main_key]
            data_dict[main_key] = np.concatenate(pcs, axis=0)
            timestamps_topic = f"{main_key}_timestamps"
            if timestamps_topic in data_dict:
                data_dict[timestamps_topic] = np.concatenate(data_dict[timestamps_topic], axis=0)
            return data_dict

        def _save_topic(self, data_dict, key, topic, msg, t, **kwargs):
            if "PointCloud2" in msg._type:
                data, timestamps = self.decode_pointcloud(msg, t)
                data_dict[key].append(data)
                timestamps_key = f"{key}_timestamps"
                if timestamps_key not in data_dict:
                    data_dict[timestamps_key] = []
                data_dict[timestamps_key].append(timestamps)

        def __getitem__(self, index) -> dict:
            assert_debug(index == self.__idx, "A RosbagDataset does not support Random access")
            assert isinstance(self.config, RosbagConfig)
            if self.__iter is None:
                self.__iter__()

            data_dict = {key: [] for key in self.topic_mapping.values()}
            main_topic_key = self.topic_mapping[self.main_topic]

            # Append Messages until the main topic has the required number of messages
            while len(data_dict[main_topic_key]) < self._frame_size:
                topic, msg, t = next(self.__iter)
                _key = self.topic_mapping[topic]
                self._save_topic(data_dict, _key, topic, msg, t, frame_index=index)

            self.__idx += 1
            # Aggregate data
            data_dict = self.aggregate_messages(data_dict)
            return data_dict

        def __next__(self):
            return self[self.__idx]

        def __len__(self):
            return self._len

        def __del__(self):
            if self.rosbag is not None:
                self.rosbag.close()


    # Hydra -- stores a RosbagConfig `rosbag` in the `dataset` group
    cs = ConfigStore.instance()
    cs.store(group="dataset", name="rosbag", node=RosbagConfig)


    class RosbagDatasetConfiguration(DatasetLoader):
        """Returns the configuration of a Dataset built for ROS"""

        def __init__(self, config: RosbagConfig, **kwargs):
            if isinstance(config, DictConfig):
                config = RosbagConfig(**config)
            super().__init__(config)

        def projector(self) -> SphericalProjector:
            return SphericalProjector(height=self.config.lidar_height, width=self.config.lidar_width,
                                      up_fov=self.config.up_fov, down_fov=self.config.down_fov)

        def sequences(self):
            assert isinstance(self.config, RosbagConfig)
            file_path = self.config.file_path
            dataset = RosbagDataset(self.config, file_path, self.config.main_topic,
                                    self.config.frame_size,
                                    OmegaConf.to_container(self.config.topic_mapping) if isinstance(
                                        self.config.topic_mapping, DictConfig) else self.config.topic_mapping)

            return ([dataset], [Path(file_path).stem]), None, None, lambda x: x

        def get_ground_truth(self, sequence_name):
            """No ground truth can be read from the ROSBAG"""
            return None
