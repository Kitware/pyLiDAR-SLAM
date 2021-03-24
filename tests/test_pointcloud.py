import unittest

from slam.common.pointcloud import *


class PointCloudCase(unittest.TestCase):
    def test_voxelization(self):
        pointcloud = np.random.randn(100000, 3)
        voxel_size = 0.1

        # Voxelization
        voxel_coordinates = voxelise(pointcloud, voxel_size, voxel_size, voxel_size)
        self.assertTrue(voxel_coordinates.dtype == np.int64)

        # Hashing
        voxel_hashes = np.zeros_like(voxel_coordinates[:, 0])
        voxel_hashing(voxel_coordinates, voxel_hashes)

        # Statistics
        num_voxels, means, covs, voxel_indices = voxel_normal_distribution(pointcloud, voxel_hashes)

        voxel_means = means[voxel_indices]

        diff = np.linalg.norm(pointcloud - voxel_means, axis=-1).max()
        self.assertLess(diff, 0.18)


if __name__ == '__main__':
    unittest.main()
