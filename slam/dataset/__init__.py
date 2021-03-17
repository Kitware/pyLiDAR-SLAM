from enum import Enum

from slam.common.utils import assert_debug
from slam.dataset.configuration import DatasetConfiguration, DatasetConfig
from slam.dataset.kitti_dataset import KITTIDatasetConfiguration
from slam.dataset.nclt_dataset import NCLTDatasetConfiguration
from slam.dataset.ford_dataset import FordCampusDatasetConfiguration


class DATASET(Enum):
    """
    The different datasets covered by the dataset_config configuration
    A configuration must have the field dataset_config pointing to one of these keys
    """
    kitti = KITTIDatasetConfiguration
    nclt = NCLTDatasetConfiguration
    ford_campus = FordCampusDatasetConfiguration

    @staticmethod
    def load(config: DatasetConfig):
        """
        Loads and returns a dataset_config configuration from the config
        """
        dataset_config = config.dataset
        assert_debug(dataset_config in DATASET.__members__)
        return DATASET[dataset_config].value(config)
