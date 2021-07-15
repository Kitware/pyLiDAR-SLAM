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
from slam.common.utils import assert_debug
from slam.dataset import DatasetLoader, DatasetConfig

try:
    import rosbag
    import sensor_msgs.point_cloud2 as pc2
    from sensor_msgs.msg import PointCloud2, PointField

    _with_rosbag = True
except ImportError:
    _with_rosbag = False

if _with_rosbag:

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

        def __init__(self, file_path: str, main_topic: str, frame_size: int, topic_mapping: Optional[dict] = None):
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
            self._frame_size: int = frame_size
            self._len = self.rosbag.get_message_count(self.main_topic) // self._frame_size

            self.__idx = 0
            self._topics = list(topic_mapping.keys())
            self.__iter = None

        def __iter__(self):
            self.__iter = self.rosbag.read_messages(self._topics)
            self.__idx = 0
            return self

        @staticmethod
        def decode(msgs: list) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            if len(msgs) == 0:
                return None, None
            if "PointCloud2" in msgs[0][1]._type:
                pcs = [np.array(list(pc2.read_points(msg, field_names="xyz"))) for t, msg in msgs]
                timestamps = np.concatenate([np.ones((pcs[i].shape[0],),
                                                     dtype=np.int64) * msgs[i][0].nsecs for i in range(len(msgs))])
                pcs = np.concatenate(pcs)

                return pcs, timestamps
            else:
                return None, None

        def _convert(self, data_dict):
            new_dict = dict()
            for key, msgs in data_dict.items():
                data, timestamps = self.decode(msgs)
                _key = self.topic_mapping[key]
                if data is not None:
                    new_dict[_key] = data
                if timestamps is not None:
                    new_dict[f"{_key}_timestamps"] = timestamps

            return new_dict

        def __getitem__(self, index) -> dict:
            assert_debug(index == self.__idx, "A RosbagDataset does not support Random access")
            if self.__iter is None:
                self.__iter__()

            data_dict = {topic: [] for topic in self._topics}
            # Append Messages until the main topic has the required number of messages
            while len(data_dict[self.main_topic]) < self._frame_size:
                topic, msg, t = next(self.__iter)
                data_dict[topic].append((t, msg))

            self.__idx += 1
            return self._convert(data_dict)

        def __next__(self):
            return self[self.__idx]

        def __len__(self):
            return self._len

        def __del__(self):
            if self.rosbag is not None:
                self.rosbag.close()

if _with_rosbag:
    @dataclass
    class RosbagConfig(DatasetConfig):
        """Config for a Rosbag Dataset"""
        dataset = "rosbag"
        file_path: str = field(
            default_factory=lambda: "" if not "ROSBAG_PATH" in os.environ else os.environ["ROSBAG_PATH"])
        main_topic: str = "numpy_pc"  # The Key of the main topic (which determines the number of frames)
        frame_size: int = 60  # The number of accumulated message which constitute a frame
        topic_mapping: dict = field(default_factory=lambda: {})

        lidar_height: int = 720
        lidar_width: int = 720
        up_fov: float = 45.
        down_fov: float = -45.


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
            dataset = RosbagDataset(file_path, self.config.main_topic,
                                    self.config.frame_size,
                                    OmegaConf.to_container(self.config.topic_mapping))

            return ([dataset], [Path(file_path).stem]), None, None, lambda x: x

        def get_ground_truth(self, sequence_name):
            """No ground truth can be read from the ROSBAG"""
            return None
