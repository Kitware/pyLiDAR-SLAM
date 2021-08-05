from typing import Union, Optional

import torch
import torch.nn.functional as Fnn
from functools import lru_cache
import numpy as np

# Project Imports
from slam.common.utils import batched, assert_debug, check_tensor


# ----------------------------------------------------------------------------------------------------------------------
@batched([-1, 3])
def cross_op(vectors: Union[np.ndarray, torch.Tensor]):
    """
    Build the cross operator from a tensor of 3D vectors

    It is the 3x3 matrix A computed from a which verifies, for each vector b,
    A * b = a x b, where * is the matrix product and x the vector product
    """
    is_numpy = isinstance(vectors, np.ndarray)
    n = vectors.shape[0]
    if is_numpy:
        _cross = np.zeros((n, 3, 3), dtype=vectors.dtype)
    else:
        _cross = torch.zeros(n, 3, 3, dtype=vectors.dtype, device=vectors.device)
    _cross[:, 0, 1] = - vectors[:, 2]
    _cross[:, 1, 0] = vectors[:, 2]
    _cross[:, 0, 2] = vectors[:, 1]
    _cross[:, 2, 0] = -vectors[:, 1]
    _cross[:, 1, 2] = -vectors[:, 0]
    _cross[:, 2, 1] = vectors[:, 0]

    return _cross


# ----------------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def pixel_grid(batch_size: int,
               height: int,
               width: int,
               dtype=torch.float32,
               device=torch.device("cpu"),
               normalized=False):
    """
    Generates a pixel grid of size [B, H, W, 2]
    """
    rows_indices = torch.arange(0, height, dtype=dtype, device=device)
    col_indices = torch.arange(0, width, dtype=dtype, device=device)

    if normalized:
        rows_indices /= 0.5 * height
        col_indices /= 0.5 * width
        rows_indices -= 1.0
        col_indices -= 1.0

    rows, cols = torch.meshgrid([rows_indices, col_indices])
    pixels = torch.cat([rows.unsqueeze(2), cols.unsqueeze(2)], dim=2)
    pixels = pixels.view(1, height, width, 2).expand(batch_size, height, width, 2)

    return pixels


# ----------------------------------------------------------------------------------------------------------------------
def _adjoint(tensor: torch.Tensor):
    """
    Compute inverses without division by det; ...xv3xc3 input, or array of matrices assumed

    Parameters
    ----------
    tensor : torch.Tensor
    """
    AI = torch.empty_like(tensor)
    for i in range(3):
        AI[..., i, :] = torch.cross(tensor[..., i - 2, :], tensor[..., i - 1, :])
    return AI


# ----------------------------------------------------------------------------------------------------------------------
def _inverse_transpose(tensor: torch.Tensor, eps=1.e-6):
    """
    Efficiently computes the inverse-transpose for stack of 3x3 matrices

    Parameters
    ----------
    tensor : torch.Tensor or np.ndarray
    """
    I = _adjoint(tensor)
    det = _dot(I, tensor).mean(dim=-1)

    Ibis = torch.zeros_like(I)
    det_mat = det[..., None, None]
    mask = det_mat.abs() > eps
    det_mat[~mask] = 1.0
    maskI = mask.expand_as(Ibis)
    Ibis = I / det_mat
    Ibis[~maskI] = 0.0

    return Ibis, det


# ----------------------------------------------------------------------------------------------------------------------
def _inverse(tensor: torch.Tensor):
    """
    Inverse of a stack of 3x3 matrices

    Parameters
    ----------
    tensor : torch.Tensor

    """
    IA, det = _inverse_transpose(tensor)
    IA = IA.transpose(-1, -2)
    return IA, det


# ----------------------------------------------------------------------------------------------------------------------
def _dot(a: torch.Tensor, b: torch.Tensor):
    """
    Dot arrays of vecs; contract over last indices
    """
    return torch.einsum('...i,...i->...', a, b)


# ----------------------------------------------------------------------------------------------------------------------
def _multi_dot(a: torch.Tensor, b: torch.Tensor):
    """
    Computes matrix to vector product for a batch of tensors

    Parameters
    ----------
    a : torch.Tensor
    b : torch.Tensor

    """
    return torch.einsum('...ij,...j->...i', a, b)


# ----------------------------------------------------------------------------------------------------------------------
def _multi_dim_matrix_product(a: torch.Tensor, b: torch.Tensor):
    """
    Computes a matrix product for a batch of matrices

    Parameters
    ----------
    a : torch.Tensor [..., N, M]
    b : torch.Tensor [..., M, K]

    Returns
    -------
    torch.Tensor [..., N, K]
    """
    return torch.einsum('...ij,...jk->...ik', a, b)


# ----------------------------------------------------------------------------------------------------------------------
def mask_not_null(tensor: torch.Tensor, dim=1):
    """
    Parameters
    ----------
    tensor : torch.Tensor
       A tensor which to be filtered for null points.
       null points must be null across dimension dim
    dim : int
        The dimension across which all the entries must be null to consider a point to be null


    Returns
    -------
    A bool tensor of same dimension as 'tensor' except the dimension 'dim' which is reduced to 1
    Where each position is :
    true if at least one value along dimension 'dim' is not 0
    false otherwise


    """
    return tensor.abs().max(dim=dim, keepdim=True)[0] > 0


# ----------------------------------------------------------------------------------------------------------------------
def projection_map_to_points(pmap: torch.Tensor, dim=1, num_channels: int = 3) -> torch.Tensor:
    """
    Parameters
    ----------
    pmap: torch.Tensor [..., num_channels, H, W]
        A projection map tensor with :
            the last two dimensions the height H and width W of the image
            the dimension dim the dimension with the field channels
    dim: int
        The dimension of the num_channels different channels
    num_channels : int
        the number of channels to add to the last dimension

    Returns
    -------
    points: torch.Tensor [..., H * W, 3]
    """
    check_tensor(pmap, [*([-1] * dim), *[num_channels, -1, -1]])
    shape = pmap.shape
    permuted = pmap.permute(*[i for i in range(dim)], dim + 1, dim + 2, dim)
    reshaped: torch.Tensor = permuted.reshape(*[shape[i] for i in range(dim)],
                                              shape[dim + 1] * shape[dim + 2],
                                              num_channels)
    return reshaped


# ----------------------------------------------------------------------------------------------------------------------
def points_to_pmap(points: torch.Tensor,
                   h,
                   w,
                   num_channels: int = 3,
                   add_batch_dim: bool = True):
    """
    Reshapes a point cloud into a projection map

    Parameters
    ----------
    points : torch.Tensor [K * h * w, num_channels]
    h : the height of the image
    w : the width of the built image
    num_channels: the number of channels of the projection map
    add_batch_dim: whether to add a new dimension when K = 1

    Returns
    -------
    vertex_maps : torch.Tensor [K, num_channels, h, w]

    """
    check_tensor(points, [-1, num_channels])
    n = points.size(0)
    assert_debug(n % h == 0)
    assert_debug(n % (h * w) == 0)
    l = n // (h * w)
    if l == 1 and not add_batch_dim:
        return points.permute(1, 0).reshape(num_channels, h, w)
    return points.reshape(l, h, w, num_channels).permute(0, 3, 1, 2)


# ----------------------------------------------------------------------------------------------------------------------
def compute_normal_map(vertex_map: torch.Tensor, kernel_size: int = 5):
    """
    Computes the normals for a VertexMap
    (An image of X,Y,Z points projected into the image plane)

    Parameters
    ----------
    vertex_map : torch.Tensor [H, W, 3]
        A Vertex map, is an image where the channels are the cartesian coordinates of the points x, y, z
    kernel_size : int
        The size of the kernel for the box filter

    Returns
    -------
    normals_map : torch.Tensor [H, W, 3]
        A Vertex map, is an image where the channels are the coordinates of the normals nx, ny, nz

    """
    b, _, h, w = vertex_map.shape
    covariance = (vertex_map.unsqueeze(1) * vertex_map.unsqueeze(2))
    covariance = covariance.reshape(b * 9, 1, h, w)
    vmap_boxed = Fnn.conv2d(vertex_map.reshape(b * 3, 1, h, w),
                            torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.float32,
                                       device=vertex_map.device),
                            padding=(kernel_size // 2, kernel_size // 2)).reshape(b, 3, h, w).permute(0, 2, 3, 1)
    cov_boxed = Fnn.conv2d(covariance, torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.float32,
                                                  device=vertex_map.device),
                           padding=(kernel_size // 2, kernel_size // 2)) \
        .reshape(b, 3, 3, h, w).permute(0, 3, 4, 1, 2)

    IA, det = _inverse(cov_boxed)
    det = abs(det)
    mask = det > 1.e-6
    n = _multi_dot(IA, vmap_boxed)
    n_mask = n[mask]

    norms = torch.norm(n_mask, dim=1, keepdim=True)
    mask_null = torch.norm(vertex_map, dim=1) == 0.0

    # Insure that the operation is differentiable (not division by zero) [A]
    mask_norm = norms == 0.
    norms = norms + mask_norm.to(torch.float32)

    n[mask] /= norms
    n[~mask] = 0.0

    # Reset to 0 the normals artificially divided by 1. (cf [A])
    mask_norm = mask_norm.expand_as(n[mask])
    n[mask][mask_norm == 0.0] = 0.0

    n[mask_null] = 0.0

    n = n.permute(0, 3, 1, 2)

    assert_debug(not torch.isnan(n).any())
    return n


# ----------------------------------------------------------------------------------------------------------------------
@lru_cache()
def neighborhood_kernel(kernel_size: int,
                        groups: int,
                        dtype: torch.dtype = torch.float32,
                        device: torch.device = torch.device("cpu")):
    """
        Returns a neighborhood convolution kernel,
        It is a tensor which can be used to extract the K * K points in the neighborhood of a given pixel
        Typically used as a weight matrix in a conv2d operation

        Parameters
        ----------
        kernel_size : int (K)
            the size of the kernel
        groups : int
            the number of groups to replicate
        dtype : torch.dtype
            The type of the weight tensor
        device : torch.device
            The device on which to create the given kernel

        Returns
        -------
        kernel : torch.Tensor [groups * K * K, K, K]
            A weigh matrix for a convolution product which extracts all points in the neighbor

    """

    neighbors = kernel_size * kernel_size
    weights = torch.zeros(neighbors, 1, kernel_size, kernel_size, dtype=dtype, device=device)
    idx = 1
    for i in range(kernel_size):
        for j in range(kernel_size):
            if i == j and i == kernel_size // 2:
                weights[0, :, i, j] = 1.0
            else:
                weights[idx, :, i, j] = 1.0
                idx += 1

    weights_neighborhood = weights
    weights_neighbors = weights.unsqueeze(0).expand(groups, neighbors, 1, kernel_size, kernel_size).reshape(
        groups * neighbors, 1,
        kernel_size, kernel_size)

    return weights_neighborhood, weights_neighbors


# ----------------------------------------------------------------------------------------------------------------------
def conv_neighborhood(image_tensor: torch.Tensor, kernel_size: int = 3):
    """
        Computes the neighborhood of a given image tensor.
        More precisely, it extracts for each pixel location p, the K * K pixels in the neighborhood of size K
        Using a neighborhood convolution kernel.
        It also returns a neighborhood mask which is 1 if a neighbor was found, and 0 if not,
        For all pixels in the neighborhood.
        For a given pixel, if its value is 0 on every channel, it is not counted as a neighbor.

        Parameters
        ----------
        image_tensor : torch.Tensor [B, C, H, W]
            An image tensor from which to extract neighbors
        kernel_size : int (K)
            the size of the kernel

        Returns
        -------
        neighbors, neighborhood  : torch.Tensor, torch.Tensor [B, K * K, C, H, W], [B, K * K, 1, H, W]
            A weigh matrix for a convolution product which extracts all points in the neighbor
    """
    check_tensor(image_tensor, [-1, -1, -1, -1])
    b, c, h, w = image_tensor.shape
    weights_neighborhood, weight_neighbors = neighborhood_kernel(kernel_size,
                                                                 c,
                                                                 image_tensor.dtype,
                                                                 image_tensor.device)

    image_tensor_mask = (image_tensor.abs().max(dim=1, keepdim=True)[0] > 0.0).to(torch.float32)
    # A [B, #NEIGHBORS, H, W] size tensor where there is a one if there is a neighbor in the 1 dimension
    padding = kernel_size // 2
    neighborhood = Fnn.conv2d(image_tensor_mask, weight=weights_neighborhood, groups=1, stride=1,
                              padding=[padding, padding])
    neighborhood = (neighborhood * image_tensor_mask).to(torch.bool)
    neighborhood_size = neighborhood.size(1)

    neighbors = Fnn.conv2d(image_tensor, weight=weight_neighbors, groups=c, stride=1, padding=[padding, padding])
    neighbors = neighbors * image_tensor_mask

    return neighbors.reshape(b, c, neighborhood_size, h, w).permute(0, 2, 1, 3, 4), \
           neighborhood.reshape(b, neighborhood_size, 1, h, w)


# ----------------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=10)
def __ones(shape: tuple, device: torch.device, dtype: torch.dtype):
    return torch.ones(*shape, device=device, dtype=dtype)


# ----------------------------------------------------------------------------------------------------------------------
def compute_neighbors(vm_target,
                      vm_reference,
                      reference_fields: Optional[torch.Tensor] = None,
                      **kwargs):
    """
    Computes the nearest neighbors between a target vertex map and a batch of reference vertex maps

    Args:
        vm_target (torch.Tensor): The target vertex map `(1, 3, H, W)`
        vm_reference (torch.Tensor): The reference vertex map `(D, 3, H, W)`
        reference_fields (torch.Tensor): An optional field map to extract along the neighbor points `(D, C, H, W)`

    Returns:
        A neighbor vertex map consisting of the closest points to `vertex_map` points among the `D` candidates in the
        Batch dimension of the reference vertex maps
        And a optional field map which returns the neighbor's corresponding field taken from the `reference_fields`
        Map.
    """
    mask_target = mask_not_null(vm_target)
    mask_reference = mask_not_null(vm_reference, dim=1)

    mask_infty = (~mask_target).to(torch.float32)
    mask_infty[~mask_target] = float("inf")

    mask_ref_infty = (~mask_reference).to(torch.float32)
    mask_ref_infty[~mask_reference] = float("inf")

    diff = (vm_target - vm_reference).norm(dim=1, keepdim=True)
    diff += mask_ref_infty + mask_infty

    # Take the min among all neighbors
    _min, indices = torch.min(diff, dim=0, keepdim=True)
    indices: torch.Tensor = indices

    vm_neighbors = torch.gather(vm_reference, 0, indices.expand(1, *vm_reference.shape[1:]))
    vm_neighbors[vm_neighbors == float("inf")] = 0.0
    vm_neighbors[~mask_target.expand(*vm_neighbors.shape)] = 0.0

    if reference_fields is not None:
        reference_fields = torch.gather(reference_fields, 0, indices.expand(1, *reference_fields.shape[1:]))
        reference_fields[reference_fields == float("inf")] = 0.0

    return vm_neighbors, reference_fields


# ----------------------------------------------------------------------------------------------------------------------
def estimate_timestamps(numpy_pc: np.ndarray, clockwise: bool = True, phi_0: float = 0.0):
    """Computes an Estimate of timestamps for rotating lasers

    Each point are expressed in spherical coordinates,
    The timestamps are assigned based on their azimuthal angle (phi)

    Note: This is an imperfect estimation, as when the the vehicle is turning, objects near the frontier
          (ie from both sides of the phi_0 can still be doubled)

    Parameters:
        clockwise (bool): whether the lidar turns clockwise or counter clockwise
                          (clockwise when considering the x axis as right, and y axis as up in the 2D plane)
        numpy_pc (np.ndarray): the pointcloud expressed in the local lidar reference `(-1, 3)`
        phi_0 (float): an initial phi added to the azimuth angles
                       the resulting timestamps are computed starting from phi_0 as the initial timestamp
    """
    phis = np.arctan2(numpy_pc[:, 1], numpy_pc[:, 0]) * (-1.0 if clockwise else 1.0)
    phis -= phi_0
    phis[phis < 0.0] += 2 * np.pi

    min_phis = phis.min()
    max_phis = phis.max()

    return (phis - min_phis) / (max_phis - min_phis)
