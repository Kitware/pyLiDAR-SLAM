from enum import Enum

from slam.common.utils import assert_debug
from slam.dataset.configuration import DatasetLoader, DatasetConfig
from slam.dataset.kitti_dataset import KITTIDatasetLoader
from slam.dataset.nclt_dataset import NCLTDatasetLoader
from slam.dataset.ford_dataset import FordCampusDatasetLoader

from slam.dataset.rosbag_dataset import _with_rosbag


class DATASET(Enum):
    """
    The different datasets covered by the dataset_config configuration
    A configuration must have the field dataset_config pointing to one of these keys
    """
    kitti = KITTIDatasetLoader
    nclt = NCLTDatasetLoader
    ford_campus = FordCampusDatasetLoader
    if _with_rosbag:
        from slam.dataset.rosbag_dataset import RosbagDatasetConfiguration
        rosbag = RosbagDatasetConfiguration

    @staticmethod
    def load(config: DatasetConfig):
        """
        Loads and returns a dataset_config configuration from the config
        """
        dataset_config = config.dataset
        assert_debug(dataset_config in DATASET.__members__)
        return DATASET[dataset_config].value(config)
