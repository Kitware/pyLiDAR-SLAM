import torch

from slam.common.utils import assert_debug
from slam.viz.color_map import tensor_to_image
import numpy as np

from slam.common.modules import _with_cv2

if _with_cv2:
    import cv2


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
