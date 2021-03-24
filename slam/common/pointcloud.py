"""
Convenient Pointcloud processing functions

@note:  Most functions are accelerated by numba and thus compiled to machine code with LLVM,
        Thus the first call of functions with decorator @nb.njit is typically very long (up to a few seconds)
"""
from numba import prange
import numba as nb

import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
@nb.njit(fastmath=True)
def voxel_hash(x, y, z):
    """
    Computes a hash given the int coordinates of a 3D voxel

    Args:
        x (np.int64): the x coordinate of a voxel
        y (np.int64): the y coordinate of a voxel
        z (np.int64): the z coordinate of a voxel
    """
    return 73856093 * x + 19349669 * y + 83492791 * z


@nb.njit(fastmath=True, parallel=True)
def planar_hashing(voxel, hashvector):
    """
    Computes a pillar hash for each voxel in an array of voxel indices

    A pillar hash is a hash for the first two coordinate of a voxel (x, y)

    Args:
        voxel (np.ndarray): An array of voxels `(n, 3)` [np.int64]
    """
    for i in prange(voxel.shape[0]):
        hashvector[i] = 73856093 * voxel[i, 0] + 19349669 * voxel[i, 1]


@nb.njit(parallel=True)
def voxel_hashing(voxel, hashvector):
    """
    Computes voxel hashes for an array of voxel indices

    Args:
        voxel (np.ndarray): An array of voxels `(n, 3)` [np.int64]

    see https://niessnerlab.org/papers/2013/4hashing/niessner2013hashing.pdf
    """
    for i in prange(voxel.shape[0]):
        hashvector[i] = voxel_hash(voxel[i, 0], voxel[i, 1], voxel[i, 2])


@nb.njit(fastmath=True, parallel=True)
def voxelise(pointcloud, voxel_x: float = 0.2, voxel_y: float = -1.0, voxel_z: float = -1.0):
    """
    Computes voxel coordinates for a given pointcloud

    Args:
         pointcloud (np.ndarray): The input pointcloud `(n, 3)` [np.float32]
         voxel_x (float): The length of the voxel in the first coordinate
         voxel_y (float): The length of the voxel in the second coordinate (same as `voxel_x` by default)
         voxel_z (float): The length of the voxel in the third coordinate (same as `voxel_x` by default)
    """

    if voxel_y == -1.0:
        voxel_y = voxel_x
    if voxel_z == -1.0:
        voxel_z = voxel_x
    """Return voxel coordinates of a pointcloud"""
    out = np.zeros((pointcloud.shape[0], 3), dtype=np.int64)
    for i in prange(pointcloud.shape[0]):
        x = int(np.round_(pointcloud[i, 0] / voxel_x))
        y = int(np.round_(pointcloud[i, 1] / voxel_y))
        z = int(np.round_(pointcloud[i, 2] / voxel_z))
        out[i, 0] = x
        out[i, 1] = y
        out[i, 2] = z
    return out


# ----------------------------------------------------------------------------------------------------------------------
@nb.njit()
def __voxel_normal_distribution(pointcloud, voxel_hashes, is_sorted: bool = False):
    """
    Computes the normal distribution of points in each voxel

    Args:
         pointcloud (np.ndarray): The input pointcloud `(n, 3)` [np.float32]
         voxel_hashes (np.ndarray): The voxel hashes `(n,)` [np.int64]
         is_sorted (bool): Whether the pointcloud is sorted by hash value
                          (avoids performing this step)
    """
    if is_sorted:
        sorted_pointcloud = pointcloud
        sorted_hashes = voxel_hashes
    else:
        sort_indices = np.argsort(voxel_hashes)
        sorted_pointcloud = pointcloud[sort_indices]
        sorted_hashes = voxel_hashes[sort_indices]

    n = pointcloud.shape[0]

    _previous_hash = sorted_hashes[0]
    _voxel_start_idx = 0

    _idx = 0
    covs = []
    means = []
    voxel_sizes = []
    hashes = []
    voxel_ids = []
    voxel_id = 0

    while _idx < n + 1:
        if _idx == n:
            # Signal to aggregate the last voxel
            _new_hash = _previous_hash + 1
        else:
            _new_hash = sorted_hashes[_idx]

        if _new_hash == _previous_hash:
            _idx += 1
            voxel_ids.append(voxel_id)
            continue

        # Build the covariance matrices and mean for the voxel
        voxel_points = sorted_pointcloud[_voxel_start_idx:_idx]
        num_points = _idx - _voxel_start_idx
        mean = voxel_points.sum(axis=0).reshape(1, 3) / num_points
        centered = voxel_points - mean
        cov = (centered.reshape(-1, 3, 1) * centered.reshape(-1, 1, 3)).sum(axis=0)

        voxel_sizes.append(num_points)
        covs.append(cov)
        means.append(mean[0])
        hashes.append(_previous_hash)

        if _idx < n:
            _previous_hash = _new_hash
            _voxel_start_idx = _idx
            voxel_id += 1
            voxel_ids.append(voxel_id)

        _idx += 1

    voxel_ids = np.array(voxel_ids)

    if not is_sorted:
        indices = np.argsort(sort_indices)
        voxel_ids = voxel_ids[indices]

    return voxel_sizes, means, covs, voxel_ids


def voxel_normal_distribution(pointcloud, voxel_hashes, is_sorted: bool = False):
    """
    Computes the normal distribution of points in each voxel

    Args:
         pointcloud (np.ndarray): The input pointcloud `(n, 3)` [np.float32]
         voxel_hashes (np.ndarray): The voxel hashes `(n,)` [np.int64]
         is_sorted (bool): Whether the pointcloud is sorted by hash value
                          (avoids performing this step)
    """
    voxel_sizes, means, covs, voxel_ids = __voxel_normal_distribution(pointcloud, voxel_hashes, is_sorted)
    return np.array(voxel_sizes), np.array(means), np.array(covs), voxel_ids
