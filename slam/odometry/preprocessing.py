from abc import ABC
from enum import Enum
from typing import Dict

import numpy as np

# Project Imports
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import torch

from slam.common.utils import assert_debug, check_sizes
from slam.common.pointcloud import voxel_hashing, voxelise, voxel_normal_distribution

# Hydra and OmegaConf
from hydra.conf import MISSING, dataclass


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class FilterConfig:
    """A Configuration for a filter"""
    filter_name: str = MISSING
    input_channel: str = MISSING


# ----------------------------------------------------------------------------------------------------------------------
class Filter(ABC):
    """A Filter on the input slam data"""

    def __init__(self, config: FilterConfig):
        self.config = config

    def filter(self, data_dict: dict):
        """Applies a filter which modifies the state of the data_dict"""
        raise NotImplementedError("")


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class VoxelizationConfig(FilterConfig):
    """The configuration for the `Voxelization` Filter"""
    filter_name = "voxelization"
    input_channel: str = "numpy_pc"

    voxel_covariances_key: str = "voxel_covariances"
    voxel_means_key: str = "voxel_means"
    voxel_size_key: str = "voxel_sizes"
    voxel_indices_key: str = "voxel_indices"
    voxel_hashes_key: str = "voxel_hashes"
    voxel_coordinates_key: str = "voxel_coordinates"

    with_normal_distribution: bool = True  # Whether to compute voxel statistics

    # Voxel sizes
    voxel_x_size: float = 0.1
    voxel_y_size: float = 0.1
    voxel_z_size: float = 0.2


# ----------------------------------------------------------------------------------------------------------------------
class Voxelization(Filter):
    """
    A Filter which voxelizes a given pointcloud, to reduce it's dimensionality

    Optionally, it computes the voxel's aggregate statistics (mean position, covariance)
    """

    def __init__(self, config: VoxelizationConfig, **kwargs):
        super().__init__(config)

    def filter(self, data_dict: dict):
        assert_debug(self.config.input_channel in data_dict,
                     f"The input channel {self.config.input_channel} was not in the input channel")
        assert isinstance(self.config, VoxelizationConfig)

        pointcloud = data_dict[self.config.input_channel]
        assert_debug(isinstance(pointcloud, np.ndarray))
        check_sizes(pointcloud, [-1, 3])

        voxel_coordinates = voxelise(pointcloud,
                                     self.config.voxel_x_size,
                                     self.config.voxel_y_size,
                                     self.config.voxel_z_size)
        voxel_hashes = np.zeros_like(voxel_coordinates[:, 0])
        voxel_hashing(voxel_coordinates, voxel_hashes)

        data_dict[self.config.voxel_hashes_key] = voxel_hashes
        data_dict[self.config.voxel_coordinates_key] = voxel_coordinates

        if self.config.with_normal_distribution:
            v_sizes, v_means, v_covs, v_indices = voxel_normal_distribution(pointcloud, voxel_hashes)

            data_dict[self.config.voxel_means_key] = v_means
            data_dict[self.config.voxel_covariances_key] = v_covs
            data_dict[self.config.voxel_size_key] = v_sizes
            data_dict[self.config.voxel_indices_key] = v_indices


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class ToTensorConfig(FilterConfig):
    """A Filter Config for numpy to pytorch conversion and renaming"""
    filter_name: str = "to_tensor"
    device: str = "cpu"

    keys: Dict[str, str] = MISSING  # The map which converts input numpy arrays into tensors


# ----------------------------------------------------------------------------------------------------------------------
class ToTensor(Filter):
    """Converts to tensor a set of numpy arrays"""

    def __init__(self, config: ToTensorConfig, device: str = "cpu", **kwargs):
        super().__init__(config)
        self.device = torch.device(device)

    def filter(self, data_dict: dict):
        assert isinstance(self.config, ToTensorConfig)

        for old_key, new_key in self.config.keys.items():
            assert_debug(old_key in data_dict)
            np_array = data_dict[old_key]
            assert_debug(isinstance(np_array, np.ndarray))
            data_dict[new_key] = torch.from_numpy(np_array).to(self.device)


# ----------------------------------------------------------------------------------------------------------------------
class FILTER(Enum):
    """Filters registered"""
    # ground_detection =
    # voxel_region_growth_clustering =
    # spherical_map_region_growth_clustering =
    # loam_keypoints_extraction =
    # random_sampling =
    # ground_point_sampling =
    # kdtree_neighborhood =
    voxelization = (Voxelization, VoxelizationConfig)
    to_tensor = (ToTensor, ToTensorConfig)

    @staticmethod
    def load(config: DictConfig, **kwargs) -> Filter:
        """Loads the configuration of the filter"""
        assert_debug("filter_name" in config)
        filter_name = config.filter_name
        assert_debug(filter_name in FILTER.__members__)

        _class, _config = FILTER[filter_name].value

        return _class(_config(**config), **kwargs)


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class PreprocessingConfig:
    """The configuration for `Preprocessing`"""
    filters: Dict[str, Dict] = MISSING


# -- Hydra add default configurations
cs = ConfigStore.instance()
cs.store(group="slam/odometry/preprocessing/filters", name="none", node=dict())


# ----------------------------------------------------------------------------------------------------------------------
class Preprocessing:
    """
    A `Preprocessing` instance applies a sequence of `Filter`(s) a data_dict
    """

    def __init__(self, preprocessing_config: PreprocessingConfig, **kwargs):
        self.config = preprocessing_config

        self.filters = []

        # Populate the filters
        filters_config = self.config.filters
        keys = filters_config.keys()
        keys = list(sorted(keys))

        for key in keys:
            self.filters.append(FILTER.load(OmegaConf.create(filters_config[key]), **kwargs))

    def forward(self, data_dict: dict):
        """Applies all filters sequentially"""
        for _filter in self.filters:
            _filter.filter(data_dict)
