import unittest
import torch

from slam.common.optimization import PointToPlaneCost, GaussNewton
from slam.common.pose import Pose


class MyTestCase(unittest.TestCase):
    def test_gauss_newton(self):
        tgt_points = torch.randn(2, 100, 3, dtype=torch.float64)
        ref_normals = torch.randn(2, 100, 3, dtype=torch.float64)
        ref_normals /= ref_normals.norm(dim=-1, keepdim=True)

        pose_params = torch.randn(2, 6, dtype=torch.float64) * torch.tensor([[0.01, 0.01, 0.01, 0.001, 0.001, 0.001]],
                                                                            dtype=torch.float64)

        pose = Pose("euler")
        ref_points = pose.apply_transformation(tgt_points, pose_params)

        gauss_newton = GaussNewton(max_iters=100, norm_stop_criterion=1.e-10, scheme="huber", sigma=0.0001)
        cost = PointToPlaneCost()
        residual_fun = cost.get_residual_fun(tgt_points, ref_points, ref_normals, pose)
        jac_fun = cost.get_residual_jac_fun(tgt_points, ref_points, ref_normals, pose)
        est_params, loss = gauss_newton.compute(torch.zeros_like(pose_params), residual_fun, jac_fun=jac_fun)

        diff_params = (est_params - pose_params).abs().max()
        est_points = pose.apply_transformation(tgt_points, est_params)
        diff_points = (ref_points - est_points).abs().max()

        self.assertLessEqual(diff_points, 1.e-7)
        self.assertLessEqual(diff_params, 1.e-7)
        self.assertLessEqual(loss.abs().sum().item(), 1.e-7)


if __name__ == '__main__':
    unittest.main()
