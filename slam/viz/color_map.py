from typing import Union

import torch
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from slam.common.utils import assert_debug


def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i]) for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}


def gray_color_map(np_image: np.array, color_map="rainbow") -> np.array:
    """
    Returns a color image from a gray image

    Parameters:
        np_image (np.ndarray): a gray image `(H,W)`

    Returns:
        np.ndarray `(4,H,W)`
    """
    assert len(np_image.shape) == 2
    min_value = np_image.min()
    max_value = np_image.max()
    np_image = (np_image - min_value) / (max_value - min_value)
    output_image = COLORMAPS[color_map](np_image).astype(np.float32)
    output_image = output_image.transpose(2, 0, 1)
    return output_image


def scalar_gray_cmap(values: np.ndarray, cmap="rainbow") -> np.ndarray:
    """Returns an array of colors from an array of values

    Parameters:
        values (np.ndarray): an array of scalars `(N,)`
    Returns:
        np.ndarray `(N,3)`
    """
    colors = gray_color_map(values.reshape(-1, 1), cmap)[:3, :, 0].T
    return colors


def rescale_image_values(t_image: Union[torch.Tensor, np.ndarray]):
    """
    Rescales the values of an image, for more convenient visualization
    Gray Image will have values in [0, 1]
    RGB Images will have values in [0, 255]
    """
    assert_debug(len(t_image.shape) == 3)
    rescale_factor = 1.
    if t_image.shape[0] == 3:
        rescale_factor = 255.
    min_value = t_image.min()
    max_value = t_image.max()

    if -1 <= min_value < 0 and max_value <= 1:
        #  Values between [-1, 1], rescale to [0, 1] or [0, 255]
        return (t_image * 0.5 + 0.5) * rescale_factor
    if min_value >= 0 and max_value <= 1:
        return t_image * rescale_factor
    if 0 <= min_value and max_value <= rescale_factor:
        # If the image is already properly scaled, return
        return t_image
    if min_value == max_value:
        return t_image
    # CROP OR Reduce scale ?
    return (t_image - min_value) / (max_value - min_value) * rescale_factor


def tensor_to_image(t_image: torch.Tensor, colormap="magma", **kwargs):
    """
    Converts an image tensor to a np.array
    The image extracted from the tensor will obey the following rules, depending on the size of the tensor :
    [B, C, H, W] => Concatenates the images to an [3, H * B, W] np.array
    [3, H, W]    => Returns the color image [3, H, W] as an np.array (dealing with the scaling of the data)
    [H, W] | [1, H, W] => Returns a color map converting the gray image to a colored image

    If C = 2, returns an image

    Returns an RGB image [3, H | H * B, W]
    """
    l_t_image = []
    if len(t_image.size()) == 4:
        b, c, h, w = t_image.size()
        for i in range(b):
            l_t_image.append(t_image[i])
    elif len(t_image.size()) == 3:
        l_t_image.append(t_image)
    elif len(t_image.size()) == 2:
        l_t_image.append(t_image.unsqueeze(0))
    else:
        raise RuntimeError("Bad Input Shape")

    l_np_imgs = []
    for t_imh in l_t_image:
        c, h, w = t_imh.size()
        if c != 2:
            t_imh = rescale_image_values(t_imh)
        if c == 3:
            np_imh = t_imh.to(dtype=torch.uint8).cpu().detach().numpy()
            l_np_imgs.append(np_imh)
        elif c == 1:
            np_imh = (gray_color_map(t_imh[0].cpu().detach().numpy(), colormap) * 255.0).astype(np.uint8)
            l_np_imgs.append(np_imh)
        else:
            raise RuntimeError(f"Unrecognised shape {t_imh.shape}")
    concat_image = np.concatenate(l_np_imgs, axis=1)
    return concat_image
