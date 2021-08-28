from abc import ABC
from enum import Enum
from typing import Dict, Optional, Any

import numpy as np
from scipy.spatial.transform.rotation import Rotation as R, Slerp

# Project Imports
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import torch

from slam.common.utils import assert_debug, check_tensor
from slam.common.pointcloud import voxel_hashing, voxelise, voxel_normal_distribution, sample_from_hashes

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
    voxel_size: float = 0.2


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
        check_tensor(pointcloud, [-1, 3])

        voxel_coordinates = voxelise(pointcloud,
                                     self.config.voxel_size,
                                     self.config.voxel_size,
                                     self.config.voxel_size)
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
@dataclass
class CVDistortionConfig(FilterConfig):
    """A Filter Config a distortion of a frame"""
    filter_name: str = "cv_distortion"
    pointcloud_key: str = "numpy_pc"
    timestamps_key: str = "numpy_pc_timestamps"
    pose_key: str = "relative_pose"
    output_key: str = "input_data"


# ----------------------------------------------------------------------------------------------------------------------
class CVDistortion(Filter):
    """Distort a frame using the estimated initial motion"""

    def __init__(self, config: CVDistortionConfig, **kwargs):
        super().__init__(config)

    def filter(self, data_dict: dict):
        assert isinstance(self.config, CVDistortionConfig)
        pc = data_dict[self.config.pointcloud_key]
        assert_debug(isinstance(pc, np.ndarray), "Cannot Distort a non numpy frame")
        check_tensor(pc, [-1, 3])
        rpose = data_dict[self.config.pose_key]
        check_tensor(pc, [4, 4])
        timestamps = data_dict[self.config.timestamps_key]
        timestamps = timestamps.reshape(-1)
        assert_debug(isinstance(timestamps, np.ndarray))
        check_tensor(timestamps, [pc.shape[0]])

        rot_times = R.from_matrix(np.array([np.eye(3, dtype=np.float64), rpose[:3, :3].astype(np.float64)]))
        key_times = [0.0, 1.0]

        slerp = Slerp(rot_times, key_times)

        alpha_timestamps = (timestamps - np.min(timestamps)) - (np.max(timestamps) - np.min(timestamps))
        alpha_timestamps.reshape(-1)
        interpolated_rots: R = slerp(alpha_timestamps)

        interpolated_tr = alpha_timestamps.reshape(-1, 1) * rpose[:3, 3].reshape(1, 3)

        distorted_frame = np.einsum("nij,nj->ni", pc, interpolated_rots.as_matrix()) + interpolated_tr
        data_dict[self.config.output_key] = distorted_frame


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class GridSampleConfig(FilterConfig):
    """A Filter Config for the sampling of a frame"""
    filter_name: str = "grid_sample"
    voxel_size: float = 0.3

    pointcloud_key: str = "numpy_pc"
    output_indices_key: str = "sample_indices"
    output_sample_key: str = "sample_points"


# ----------------------------------------------------------------------------------------------------------------------
class GridSample(Filter):
    """Distort a frame using the estimated initial motion"""

    def __init__(self, config: GridSampleConfig, **kwargs):
        super().__init__(config)

    def filter(self, data_dict: dict):
        assert isinstance(self.config, GridSampleConfig)

        pc = data_dict[self.config.pointcloud_key]
        assert_debug(isinstance(pc, np.ndarray), "Cannot Distort a non numpy frame")
        check_tensor(pc, [-1, 3])

        voxel_coords = voxelise(pc, self.config.voxel_size, self.config.voxel_size, self.config.voxel_size)
        voxel_hashes = np.zeros((pc.shape[0]), dtype=np.int64)
        voxel_hashing(voxel_coords, voxel_hashes)

        sample, indices = sample_from_hashes(pc, voxel_hashes)
        data_dict[self.config.output_sample_key] = sample
        data_dict[self.config.output_indices_key] = indices


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
    cv_distortion = (CVDistortion, CVDistortionConfig)
    voxelization = (Voxelization, VoxelizationConfig)
    grid_sample = (GridSample, GridSampleConfig)
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
    filters: Optional[Dict[str, Any]] = None


# -- Hydra add default configurations
cs = ConfigStore.instance()
cs.store(group="slam/preprocessing", name="none", node=PreprocessingConfig(filters=OmegaConf.create(dict())))


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
        if filters_config is not None:
            keys = filters_config.keys()
            keys = list(sorted(keys))

            for key in keys:
                self.filters.append(FILTER.load(OmegaConf.create(filters_config[key]), **kwargs))

    def forward(self, data_dict: dict):
        """Applies all filters sequentially"""
        for _filter in self.filters:
            _filter.filter(data_dict)
