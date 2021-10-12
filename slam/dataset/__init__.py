from enum import Enum

from slam.common.utils import ObjectLoaderEnum
from slam.dataset.configuration import DatasetLoader, DatasetConfig
from slam.dataset.kitti_dataset import KITTIDatasetLoader, KITTIConfig
from slam.dataset.nclt_dataset import NCLTDatasetLoader, NCLTConfig
from slam.dataset.ford_dataset import FordCampusDatasetLoader, FordCampusConfig
from slam.dataset.nhcd_dataset import NHCDDatasetLoader, NHCDConfig
from slam.dataset.kitti_360_dataset import (KITTI360Config, KITTI360DatasetLoader)

from slam.dataset.rosbag_dataset import _with_rosbag
from slam.dataset.ct_icp_dataset import _with_ct_icp


class DATASET(ObjectLoaderEnum, Enum):
    """
    The different datasets covered by the dataset_config configuration
    A configuration must have the field dataset_config pointing to one of these keys
    """

    kitti = (KITTIDatasetLoader, KITTIConfig)
    kitti_360 = (KITTI360DatasetLoader, KITTI360Config)
    nclt = (NCLTDatasetLoader, NCLTConfig)
    ford_campus = (FordCampusDatasetLoader, FordCampusConfig)
    nhcd = (NHCDDatasetLoader, NHCDConfig)
    if _with_rosbag:
        from slam.dataset.rosbag_dataset import RosbagDatasetConfiguration, RosbagConfig
        from slam.dataset.urban_loco_dataset import UrbanLocoConfig, UrbanLocoDatasetLoader
        rosbag = (RosbagDatasetConfiguration, RosbagConfig)
        urban_loco = (UrbanLocoDatasetLoader, UrbanLocoConfig)

    if _with_ct_icp:
        from slam.dataset.ct_icp_dataset import CT_ICPDatasetLoader, CT_ICPDatasetConfig
        ct_icp = (CT_ICPDatasetLoader, CT_ICPDatasetConfig)

    @classmethod
    def type_name(cls):
        return "dataset"
