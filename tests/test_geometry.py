import unittest
from slam.common.geometry import *


class GeometryTestCase(unittest.TestCase):
    def test_compute_neighbors(self):
        n = 10
        height = 10
        width = 10
        tgt = torch.randn(1, 3, height, width)
        ref = torch.randn(n, 3, height, width)

        tgt[0, :, 0, 0] = 0.0

        neighbors, _ = compute_neighbors(tgt, ref)
        for k in range(height):
            for l in range(width):
                if k == 0.0 and l == 0.0:
                    self.assertEqual(neighbors[0, :, 0, 0].norm(), 0.0)
                    continue
                norm_neighbor = (neighbors[0, :, k, l] - tgt[0, :, k, l]).norm()
                for i in range(n):
                    norm_ref = (ref[i, :, k, l] - tgt[0, :, k, l]).norm()
                    self.assertLessEqual(norm_neighbor, norm_ref)


if __name__ == '__main__':
    unittest.main()
