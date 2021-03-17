from typing import Optional, Any

import torch
import numpy as np
from abc import ABC, abstractmethod

from functools import lru_cache
from slam.common.utils import check_sizes, assert_debug


def torch__spherical_projection(t_pointcloud: torch.Tensor,
                                height: int,
                                width: int,
                                min_vertical_fov: float,
                                max_vertical_fov: float,
                                min_horizontal_fov: float = -180.0,
                                max_horizontal_fov: float = 180.0) -> tuple:
    """
    Computes a spherical projection of the points of a point cloud

    It will compute the pixel values of the points in t_pointcloud

    Parameters
    ----------
    t_pointcloud: torch.Tensor [B, N, 3]
        A batch of tensor to projection to spherical coordinates
    height: int
        the height of the destination image
    width: int
        the width of the destination image
    min_vertical_fov: float (angle in degrees)
        the field of view up of the image
    max_vertical_fov: float (angle in degrees)
        the field of view down of the image
    min_horizontal_fov: float (optional)
        the horizontal field of view left (of the image)
    max_horizontal_fov: float (optional)
        the horizontal field of view right (of the image)

    Returns
    -------
    (t_row, t_col) : pair of torch.Tensor of size [B, N] (torch.float32)
        t_row : the pixels' rows as a float for each point in the point cloud
        t_col : the pixels' cols as a float for each point in the point cloud
    """
    check_sizes(t_pointcloud, [-1, -1, 3])
    fov_up = min_vertical_fov / 180.0 * np.pi
    fov_down = max_vertical_fov / 180.0 * np.pi
    fov = abs(fov_down) + abs(fov_up)

    # get depth of all points
    r = torch.norm(t_pointcloud, p=2, dim=2)

    # Define a mask of validity to avoid nan
    mask_0 = (r == 0.0).to(dtype=t_pointcloud.dtype)
    mask_valid = 1.0 - mask_0
    r = mask_0 * 0.001 + mask_valid * r

    x = t_pointcloud[:, :, 0]
    y = t_pointcloud[:, :, 1]
    z = t_pointcloud[:, :, 2]

    # compute angles
    theta = - torch.atan2(y, x)
    phi = torch.asin(z / r)

    proj_col = 0.5 * (theta / np.pi + 1.0)
    proj_row = 1.0 - (phi + abs(fov_down)) / fov

    proj_col *= width
    proj_row *= height

    return proj_row * mask_valid - mask_0, proj_col * mask_valid - mask_0, r * mask_valid


def xyz_conversion(t_point_cloud: (torch.Tensor, np.ndarray)) -> torch.Tensor:
    """
    Extracts the xyz fields of a point cloud

    Parameters
    ----------
    t_point_cloud : A [B, N, C >= 3] or a [N, C >= 3] array
        Extracts the first three channels of a tensor

    >>> assert (xyz_conversion(np.array([[1.0, 2.0, 3.0, 4.0]])) - np.array([[1.0, 2.0, 3.0]]) == 0.0).all()
    """
    if len(list(t_point_cloud.shape)) == 2:
        n, c = t_point_cloud.shape
        assert_debug(c >= 3)
        return t_point_cloud[:, :3]
    else:
        check_sizes(t_point_cloud, [-1, -1, -1])
        b, n, c = t_point_cloud.shape
        assert_debug(c >= 3)
        return t_point_cloud[:, :, :3]


def depth_conversion(t_point_cloud: (torch.Tensor, np.ndarray)) -> (torch.Tensor, np.ndarray):
    """
    Extracts the depth from a point cloud

    Parameters
    ----------
    t_point_cloud : (torch.Tensor, np.ndarray) [B, N, 3] of [N, 3]
        A Point cloud which can be either a tensor or a numpy ndarray

    Returns
    -------
    (torch.Tensor, np.ndarray) : [B, N, 1]
        A Tensor of the same type as the input tensor
    >>> check_sizes(depth_conversion(torch.randn(4, 10, 3)), [4, 10, 1])
    >>> check_sizes(depth_conversion(np.random.randn(4, 10, 3)), [4, 10, 1])
    >>> check_sizes(depth_conversion(np.random.randn(40, 3)), [40, 1])
    """
    if len(t_point_cloud.shape) == 2:
        assert_debug(isinstance(t_point_cloud, np.ndarray) and t_point_cloud.shape[1] >= 3)

        return np.linalg.norm(t_point_cloud, ord=2, axis=1, keepdims=True)

    else:
        check_sizes(t_point_cloud, [-1, -1, -1])
        if isinstance(t_point_cloud, np.ndarray):
            return np.linalg.norm(t_point_cloud[:, :, :3], ord=2, axis=2, keepdims=True)
        else:
            return torch.norm(t_point_cloud[:, :, :3], p=2, dim=2, keepdim=True)


def build_spherical_image(t_point_cloud: torch.Tensor,
                          destination: torch.Tensor,
                          min_vertical_fov: float,
                          max_vertical_fov: float,
                          min_horizontal_fov: float = -180.0,
                          max_horizontal_fov: float = 180.0,
                          conversion_function: callable = lambda x: x):
    """
    Builds a Spherical Image from a Point Cloud in place

    Parameters
    ----------
    t_point_cloud: torch.Tensor [B, N, C >= 3]
        The first 3 channels corresponding to the coordinates X, Y, Z
    destination: torch.Tensor [B, C_dest, H, W]
        In which the image will be projected. The projection is done in place
    min_vertical_fov: float in [0.0, 180.0]
        The angle in degrees of the upward boundary of the fov
    max_vertical_fov: float in [min_vertical_fov, 180.0]
        The angle in degrees of the downward boundary of the fov
    min_horizontal_fov: float in [-180.0, 180.0]
        The angle in degrees of the leftward boundary of the fov
    max_horizontal_fov: float in [min_horizontal_fov, 180.0]
        The angle in degrees of the rightward boundary of the fov
    conversion_function: callable
        The function to convert a point cloud [B, N, C] into
        a point cloud with the specific channels to put in the image [B, N, C_dest]

    """
    check_sizes(destination, [-1, 3, -1, -1])
    check_sizes(t_point_cloud, [-1, -1, -1])
    # Extract channels to put in destination
    channels_extracted = conversion_function(t_point_cloud)
    b, n, c = t_point_cloud.shape
    assert_debug(c >= 3, "The point cloud must have at least 3 channels")

    bp, c_dest, height, width = destination.shape
    assert_debug(bp == b, "Mismatch between the batch size of the destination and the source point cloud")

    proj_row, proj_col, depth = torch__spherical_projection(t_point_cloud[:, :, :3],
                                                            height,
                                                            width,
                                                            min_vertical_fov,
                                                            max_vertical_fov,
                                                            min_horizontal_fov,
                                                            max_horizontal_fov)
    proj_row = torch.floor(proj_row)
    proj_row = proj_row.clamp(min=0, max=height - 1)

    proj_col = torch.floor(proj_col)
    proj_col = proj_col.clamp(min=0, max=width - 1)

    b_idx = torch.arange(b, dtype=torch.int64, device=t_point_cloud.device).view(b, 1).expand(b, n).reshape(b * n)
    order = torch.argsort(depth, dim=1).reshape(b * n)
    proj_row = proj_row[b_idx, order].to(torch.int64)
    proj_col = proj_col[b_idx, order].to(torch.int64)
    destination[b_idx, :, proj_row, proj_col] = channels_extracted[b_idx, order, :]


class Projector(ABC):
    """
    A Projector is an object which can project a PointCloud in an image
    And construct a PointCloud from a Depth image
    """

    def __init__(self,
                 transform: callable = lambda x: x,
                 height: Optional[int] = None,
                 width: Optional[int] = None):
        # The transform mapping a pointcloud to a array or tensor of color values
        # Used to construct an image from the point cloud
        self.transform = transform

        self.height: Optional[int] = height
        self.width: Optional[int] = width

    @abstractmethod
    def project_pointcloud(self, pointcloud: torch.Tensor, **kwargs) -> torch.tensor:
        """
        Projects the points of a PointCloud tensor in the image plane

        Parameters
        ----------
        pointcloud : torch.Tensor [B, N, 3]
            A Pointcloud tensor (X, Y, Z) to be projected in the image plane

        kwargs :
            Additional arguments required for the projection

        Returns
        -------
        torch.Tensor
            The tensor of size [B, N, 2] of pixel values (as float) in the image plane
            The first coordinate is the pixel of the row, and the second the pixel coordinate in the columns
            When considering the image as a matrix. (The values can be outside of the image plane dimension)

        """
        raise NotImplementedError("")

    def project_normalized(self, pointcloud: torch.Tensor, height=None, width=None, **kwargs) -> torch.Tensor:
        """

        Parameters
        ----------
        pointcloud : torch.Tensor
            The point cloud tensor [B, N, 3] to project in the image plane
        height : int
            The optional height of the image
            Uses member height if it is None
        width :
            The optional width of the image
            Uses member width if it is None
        kwargs

        Returns
        -------
        torch.Tensor [B, N, 2]
            A Tensor of pixels normalized between -1, 1

        """
        height = self.swap(height=height)
        width = self.swap(width=width)
        pixels = self.project_pointcloud(pointcloud, height=height, width=width, **kwargs)
        rows = pixels[:, :, 0] * 2.0 / height
        cols = pixels[:, :, 1] * 2.0 / width
        pixels: torch.Tensor = (-1.0 + torch.cat([rows.unsqueeze(2), cols.unsqueeze(2)], dim=2))
        return pixels

    @abstractmethod
    def rescaled_projector(self, new_height: int, new_width: int):
        """
        Parameters
        ----------
        new_height : int
            The new height of the projector
        new_width
            The new width of the projector
        Returns
        -------
        Projector
            A similar Projector, with its dimension reset to new_height and new_width
            And its appropriate parameters reset (intrinsics)

        """
        raise NotImplementedError("")

    def rescale_intrinsics(self, new_height: int, new_width: int, **kwargs) -> Any:
        """
        Rescales the intrinsics parameters of the projection from the arguments

        Parameters
        ----------
        new_height : int
            The height of the new image
        new_width : int
            The width of the new image
        kwargs
            arguments to rescale
            (Depends on the type of the projector)

        Returns
        -------
        Any
            The intrinsics rescaled : depends on the type of Projector

        """
        raise NotImplementedError("")

    @abstractmethod
    def set_projection_params(self, height: int = None, width: int = None, transform: callable = None, **kwargs):
        """
        Reads projection params from the arguments and set the appropriate parameters
        All named arguments are optional, and will only be set if they are not None

        Parameters
        ----------
        height : int
            The height of the image created from a point cloud
        width : int
            The width of the image created from a point cloud
        transform : callable
            The transformation applied to a pointcloud to extract color channels to build
            The projection image from

        **kwargs : other variables

        """
        if height is not None:
            self.height = height
        if width is not None:
            self.width = width
        if transform is not None:
            self.transform = transform

    def swap(self, **kwargs):
        for key, value in kwargs.items():
            assert_debug(hasattr(self, key))
            if value is None:
                member_value = getattr(self, key)
                assert_debug(member_value is not None)
                value = member_value
            return value

    def build_projection_map(self,
                             pointcloud: torch.Tensor,
                             default_value: float = 0.0,
                             height: Optional[int] = None,
                             width: Optional[int] = None,
                             transform: Optional[callable] = None,
                             **kwargs) -> torch.Tensor:
        """
        Builds a projection image from a PointCloud (torch.Tensor)

        Parameters
        ----------
        pointcloud : torch.Tensor
            A [B, N, C>=3] torch.Tensor with the first 3 channels the cartesian coordinates X, Y, Z
        default_value : float
            The default value for the image being built
        height : int
            Optional value of the height of the image created
            (If it is None, the member height will be used)
        width : int
            Optional value of the width of the image created
            (If it is None, the member height will be used)
        transform : Optional callable
            The function called on a point cloud which maps the input pointcloud
            to the channels desired in the image created.
            Transforms a [B, N, C] pointcloud to a [B, N, C_dest] point cloud
        kwargs

        Returns
        -------
        torch.Tensor : [B, C_dest, height, width]
            An image of size (height, width)
            (Either the height and width of the parameters or the member height and width)

        """
        height = self.swap(height=height)
        width = self.swap(width=width)
        transform = self.swap(transform=transform)

        check_sizes(pointcloud, [-1, -1, -1])
        b, n, _ = pointcloud.shape
        image_channels = pointcloud
        if transform is not None:
            image_channels = transform(image_channels)
        c_dest = image_channels.shape[2]

        # Build destination tensor
        if default_value == 0.:
            destination_image = torch.zeros(pointcloud.size(0),
                                            c_dest,
                                            height,
                                            width,
                                            device=pointcloud.device,
                                            dtype=pointcloud.dtype)
        else:
            destination_image = torch.ones(pointcloud.size(0),
                                           c_dest,
                                           height,
                                           width,
                                           device=pointcloud.device,
                                           dtype=pointcloud.dtype) * default_value

        pixels = self.project_pointcloud(pointcloud, height=height, width=width, **kwargs)
        r = pointcloud.norm(dim=2)
        pixel_rows = pixels[:, :, 0].round()
        pixel_cols = pixels[:, :, 1].round()

        invalidity_mask = ~((pixel_rows[:] >= 0.0) * \
                            (pixel_rows[:] <= (height - 1)) * \
                            (pixel_cols[:] >= 0.0) * \
                            (pixel_cols[:] <= (width - 1)))

        b_idx = torch.arange(b, dtype=torch.int64, device=pointcloud.device).view(b, 1).expand(b, n)
        r[invalidity_mask] = -1.0
        order = torch.argsort(r, dim=1, descending=True)
        order = order.reshape(b, n)
        b_idx = b_idx.reshape(b, n)

        mask = r[b_idx, order] > 0.0

        order = order[mask]
        b_idx = b_idx[mask]
        proj_row = pixel_rows[b_idx, order].to(torch.int64)
        proj_col = pixel_cols[b_idx, order].to(torch.int64)
        destination_image[b_idx, :, proj_row, proj_col] = image_channels[b_idx, order, :]

        # TODO DEAL WITH [0, 0] coordinates clamping problem
        return destination_image


@lru_cache(maxsize=10)
def torch_ones(b: int, n: int, dtype: torch.dtype, device: torch.device):
    return torch.ones(b, n, 1, dtype=dtype, device=device)


class SphericalProjector(Projector):
    """
    A SphericalProjector projects a pointcloud in a spherical image

    Parameters
    ----------
    up_fov : float
        The field of view upward in degrees [-90, 90]
    down_fov : float
        The field of view downward in degrees [-90, up_vertical_fov]

    """

    def __init__(self,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 num_channels: Optional[int] = None,
                 up_fov: Optional[float] = None,
                 down_fov: Optional[float] = None,
                 conversion: Optional[callable] = xyz_conversion):
        super().__init__(transform=conversion, height=height, width=width)
        self.num_channels = num_channels
        self.up_fov = up_fov
        self.down_fov = down_fov
        self.conversion = conversion

    def project_pointcloud(self,
                           pointcloud: torch.Tensor,
                           height: Optional[int] = None,
                           width: Optional[int] = None,
                           up_fov: Optional[float] = None,
                           down_fov: Optional[float] = None, **kwargs) -> torch.tensor:
        """
        Project the pointcloud in the Spherical image

        Parameters
        ----------
        pointcloud : torch.Tensor [B, N, K>=3]
        height: Optional[int]
            The height of the spherical image built
        width:  Optional[int]
            The width of the spherical image built
        up_fov: Optional[float]
        down_fov: Optional[float]

        Returns
        -------
        pixel_tensor : torch.Tensor [B, N, 2]
            The pixel tensor of the pointcloud projected in the Spherical image plane
            First coordinates are the row values, Second are the column values

        """
        check_sizes(pointcloud, [-1, -1, -1])
        height: int = self.swap(height=height)
        width = self.swap(width=width)
        up_fov = self.swap(up_fov=up_fov)
        down_fov = self.swap(down_fov=down_fov)
        t_rows, t_cols, r = torch__spherical_projection(pointcloud[:, :, :3], height, width, up_fov, down_fov)
        return torch.cat([t_rows.unsqueeze(2), t_cols.unsqueeze(2)], dim=2)

    def rescaled_projector(self, new_height: int, new_width: int):
        """
        Returns a rescaled Spherical projector
        """
        return SphericalProjector(height=new_height,
                                  width=new_width,
                                  num_channels=self.num_channels,
                                  up_fov=self.up_fov,
                                  down_fov=self.down_fov,
                                  conversion=self.conversion)

    def rescale_intrinsics(self, new_height: int, new_width: int, **kwargs) -> Any:
        """
        The Spherical projection does not need to rescale its intrinsics parameters
        """
        raise NotImplementedError("")

    def set_projection_params(self, up_fov: float = None, down_fov: float = None, **kwargs):
        super().set_projection_params(**kwargs)
        if up_fov is not None:
            self.up_fov = up_fov
        if down_fov is not None:
            self.down_fov = down_fov
