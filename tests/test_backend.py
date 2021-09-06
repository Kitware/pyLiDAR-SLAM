import unittest
import numpy as np
from scipy.spatial.transform import Rotation

# Project Imports
from slam.eval.eval_odometry import compute_relative_poses, compute_absolute_poses
from viz3d.window import OpenGLWindow


class BackendTestCase(unittest.TestCase):
    def test_graph_slam(self):
        # Construct Test problem
        from slam.backend import _with_g2o
        self.assertTrue(_with_g2o)

        from slam.backend import GraphSLAM, GraphSLAMConfig

        num_poses = 101
        thetas = 2 * np.pi * np.arange(num_poses) / num_poses

        c = np.cos(thetas)
        s = np.sin(thetas)
        poses = np.eye(4, dtype=np.float64).reshape(1, 4, 4).repeat(num_poses, axis=0)
        poses[:, 0, 0] = c
        poses[:, 1, 1] = c
        poses[:, 0, 1] = s
        poses[:, 1, 0] = -s
        poses[:, 0, 3] = 10 * c
        poses[:, 1, 3] = -10 * s

        relative_gt = compute_relative_poses(poses)
        # Apply noise to the relative poses
        noise = np.eye(4, dtype=np.float64).reshape(1, 4, 4).repeat(num_poses, axis=0)
        rotation_noise = np.random.randn(num_poses, 3) * 0.1
        trans_noise = np.random.randn(num_poses, 3) * 0.1
        noise[:, :3, 3] = trans_noise
        noise[:, :3, :3] = Rotation.from_euler("xyz", rotation_noise).as_matrix()

        relative_noisy = np.einsum("nij,njk->nik", relative_gt, noise)
        absolute_noisy = compute_absolute_poses(relative_noisy)

        # Show the problem
        window = OpenGLWindow()
        window.init()

        window.set_cameras(0, poses.astype(np.float32), scale=0.1, line_width=1)
        window.set_cameras(1, absolute_noisy.astype(np.float32), scale=0.1, line_width=1,
                           default_color=np.array([[1.0, 0.0, 0.0]]))

        # Solve with the Backend
        config = GraphSLAMConfig()
        graph_slam = GraphSLAM(config)

        # Construct i -> (i+1) relative constraints
        data_dict = dict()
        for idx in range(1, num_poses):
            data_dict[GraphSLAM.se3_odometry_constraint(idx - 1, idx)] = (relative_noisy[idx], None)

        # Loop closure constraint
        data_dict[GraphSLAM.se3_odometry_constraint(0, num_poses - 1)] = (np.eye(4, dtype=np.float64), 1.e6 * np.eye(6))

        graph_slam.init()
        graph_slam.next_frame(data_dict)

        for i in range(100):
            graph_slam.optimize(100)
            optimized = graph_slam.absolute_poses().astype(np.float32)
            window.set_cameras(2, optimized, scale=0.1, line_width=1, default_color=np.array([[0.0, 1.0, 0.0]]))

        window.close()


if __name__ == '__main__':
    unittest.main()
