from typing import Optional

import torch
from torch.utils.data import Dataset

from slam.common.utils import assert_debug, check_sizes
from slam.viz.color_map import tensor_to_image
import numpy as np
import logging

try:
    import cv2

    _with_cv2 = True
except ModuleNotFoundError:
    logging.warning("OpenCV (cv2 python module) not found, visualization disabled")
    _with_cv2 = False

if _with_cv2:
    class _ImageVisualizer(object):
        """
        A Visualizer displays images tensors in OpenCV windows

        Parameters
        ----------
        channels : list of str
            The keys of image tensors in each iteration data_dict
        update_frequency : int
            The frequency of update for each image
        wait_key : int
            The number of milliseconds to wait for a key to be pressed before moving on
        """

        def __init__(self, channels: list, update_frequency: int = 10, wait_key: int = 1):
            self.channels: set = set(channels)

            self.update_frequency = update_frequency
            self.global_step = 0
            self.wait_key = wait_key
            self.old_wait_key = wait_key

            for channel in self.channels:
                self.prepare_channel(channel)

        def prepare_channel(self, channel_name: str):
            assert_debug(channel_name in self.channels)
            for channel in self.channels:
                cv2.namedWindow(channel, cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)

        def __del__(self):
            for channel in self.channels:
                cv2.destroyWindow(channel)

        def visualize(self, data_dict: dict, iter_: int):
            """
            Visualize the images in data_dict in OpenCV windows

            The data_dict must have all keys in in self.channels
            Each item for these keys must either be a torch.Tensor representing a batch of images of shape [B, C, H, W]
            Or a np.ndarray of shape [H, W, C]
            In both cases C must be in {1, 3}

            The images will only be displayed every self.update_frequency

            Parameters
            ----------
            data_dict : dict
                The dictionary of tensors, from which images
            iter_ : int
                The global step for the current data_dict
            """
            if iter_ % self.update_frequency == 0:
                for channel in self.channels:
                    try:
                        assert_debug(channel in data_dict, f"chanel {channel} not in data_dict")
                        image_data = data_dict[channel]
                        if isinstance(image_data, np.ndarray):
                            has_right_dimensions = len(image_data.shape) == 3 or len(image_data.shape) == 2
                            has_right_num_channels = len(image_data.shape) == 3 and (
                                    image_data.shape[2] != 3 and image_data[2] != 1)
                            assert has_right_dimensions and has_right_num_channels, \
                                f"[NodeVisualization] Bad shape for np.array (expect [H, W, C]), got {image_data.shape}"
                        elif isinstance(image_data, torch.Tensor):
                            image_data = tensor_to_image(image_data).transpose((1, 2, 0))
                        image_data = image_data[:, :, [2, 1, 0]]
                        cv2.imshow(channel, image_data)
                    except (Exception, AssertionError) as e:
                        print(f"Error trying to visualize channel {channel}")
                        raise e
                cv2.waitKey(self.wait_key)


def ImageVisualizer(channels: list, update_frequency: int = 10, wait_key: int = 1):
    """Returns an Image Visualizer based on OpenCV if the package cv2 was found"""
    if _with_cv2:
        return _ImageVisualizer(channels, update_frequency, wait_key)
    else:
        return None


# ----------------------------------------------------------------------------------------------------------------------
# Point cloud visualization
try:
    import open3d

    _with_open3d = True
    logging.info("Found Open3D")
except ImportError:
    _with_open3d = False
    logging.warning("Open3D (open3d python module) not found, visualization disabled")

if _with_open3d:

    class SLAMViewerOpen3D:
        """A Viewer which displays aggregated pointclouds in an Open3D window"""

        def __init__(self, dataset: Dataset, trajectory: Optional[np.ndarray] = None, pointcloud_key: str = "numpy_pc",
                     color_map: str = "viridis", point_size: int = 1, light_on: bool = True, with_color: bool = True):
            self.dataset = dataset
            self.pointcloud_key = pointcloud_key
            self._num_pcs = len(self.dataset)

            from matplotlib import cm
            self.color_map = cm.get_cmap(color_map)

            if trajectory is None:
                self.trajectory = np.eye(4).reshape(1, 4, 4).repeat(self._num_pcs, axis=0)
            else:
                check_sizes(trajectory, [self._num_pcs, 4, 4])
                self.trajectory = trajectory

            self.with_color = with_color
            self.point_size = point_size
            self.light_on = light_on

        def show(self, pc_ids: list, sample_ratio: float = 1.0):
            """
            Displays the pointclouds in an Open3D Window
            """
            pcs = []
            for idx in pc_ids:
                assert_debug(0 <= idx < self._num_pcs)
                pc = self.dataset[idx][self.pointcloud_key]
                pose = self.trajectory[idx]

                pcs.append(np.einsum("ij,nj->ni", pose[:3, :3], pc) + pose[:3, 3:4].T)

            pcs = np.concatenate(pcs)

            # Sample the pcs to display
            if 0.0 <= sample_ratio < 1.0:
                num_samples = int(pcs.shape[0] * sample_ratio)
                sample_indices = np.random.randint(0, pcs.shape[0], (num_samples,))
                pcs = pcs[sample_indices]

            # Draw Geometries
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(pcs)

            if self.with_color:
                zs = pcs[:, 2]
                zs_sorted = np.argsort(zs)
                z_min = zs[int(zs_sorted.shape[0] * 0.01)]  # Clamp min values below the 0.01
                z_max = zs[int(zs_sorted.shape[0] * 0.99)]
                normalized_zs = ((zs - z_min) / (z_max - z_min)).clip(0, 1)
                color = self.color_map(normalized_zs)[:, :3]
                pcd.colors = open3d.utility.Vector3dVector(color)

            vis = open3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd)
            vis.get_render_option().point_size = self.point_size
            vis.get_render_option().light_on = self.light_on
            vis.get_render_option().background_color = np.zeros((3,))
            vis.run()
            vis.destroy_window()
