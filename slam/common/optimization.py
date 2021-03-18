import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.autograd.functional as Fa

# Hydra and OmegaConf
from omegaconf import DictConfig

# Project Imports
from slam.common.pose import Pose
from .utils import check_sizes, assert_debug


# ----------------------------------------------------------------------------------------------------------------------
class _WLSScheme(ABC):
    """
    A Weighted Least Square Scheme (WLS) defines the cost function to define a robust minimization objective

    The minimization objective is formulated as `$E_{ls} = \sum_{i=1}^n  (w_i \cdot{} r_i (x) ) ^ 2$`
    In this scheme, given a robust cost function $C(|r_i|)$, and the weights are computed as
    `$w_i=C(||r_i||)^{\frac{1}{2}} / ||r_i||_2$`

    Attributes:
        eps (float): The precision at which residuals are clamped
    """

    def __init__(self, eps: float = 1.e-4, **kwargs):
        self.eps = eps

    def __call__(self, residuals: torch.Tensor, **kwargs):
        """
        Returns the weighted tensor of residuals given a tensor of initial unweighted residuals

        Args:
            residuals (torch.Tensor): The tensor of residuals `(...,N)`

        Returns:
            The weighted least-square cost tensor `(...,N)`
        """
        return self.weights(residuals.detach()) * residuals

    def weights(self, residuals: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the Attenuation Factor used to define the weighted least square
        """
        clamped_residuals = residuals.abs().clamp(self.eps, float("inf"))
        return self.cost(residuals, **kwargs).sqrt() / clamped_residuals

    @abstractmethod
    def cost(self, residuals: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        The Cost function associated to the Least Square Scheme
        """
        raise NotImplementedError("")


# ----------------------------------------------------------------------------------------------------------------------
class _LeastSquareScheme(_WLSScheme):
    """The Standard least square Scheme, which is extremely susceptible to outliers"""

    def cost(self, residuals: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        The Standard cost associated to the residuals
        """
        return residuals ** 2

    def weights(self, residuals: torch.Tensor, **kwargs) -> torch.Tensor:
        """Returns a scalar weight of 1 for standard least square (to avoid) unnecessary computations"""
        return torch.tensor([1], dtype=residuals.dtype, device=residuals.device).reshape(*[1 for _ in residuals.shape])


# ----------------------------------------------------------------------------------------------------------------------
class _HuberScheme(_WLSScheme):
    """
    The Robust Huber Least Square cost function

    See: https://en.wikipedia.org/wiki/Huber_loss
    """

    def __init__(self, sigma: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self._sigma = sigma

    def cost(self, residuals: torch.Tensor, sigma: Optional[float] = None, **kwargs) -> torch.Tensor:
        """
        Huber cost computed from the residuals
        """
        if sigma is None:
            sigma = self._sigma
        abs_res = residuals.abs()
        is_squared = abs_res < sigma

        cost = is_squared * (residuals * residuals) + ~is_squared * (2 * sigma * abs_res - sigma ** 2)
        return cost


# ----------------------------------------------------------------------------------------------------------------------
class _ExponentialScheme(_WLSScheme):
    """
    Exponentially Weighted Cost function quickly kills residuals larger than `sigma`
    """

    def __init__(self, sigma: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self._sigma = sigma

    def cost(self, residuals: torch.Tensor, sigma: Optional[float] = None, **kwargs) -> torch.Tensor:
        """
        Returns the weighted squared residuals
        """
        if sigma is None:
            sigma = self._sigma
        cost = (residuals * residuals) * torch.exp(- residuals ** 2 / sigma ** 2)
        return cost


# ----------------------------------------------------------------------------------------------------------------------
class _NeighborhoodScheme(_WLSScheme):
    """
    Residuals are weighted by the confidence in the neighborhood association which led to the residual

    The confidence is given by the weights : $w(r) = exp (- ||q - p||^2_2 / sigma^2)$
    """

    def __init__(self, sigma: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self._sigma = sigma

    def cost(self, residuals: torch.Tensor, sigma: Optional[float] = None,
             target_points: Optional[torch.Tensor] = None,
             reference_points: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Returns the weighted squared residuals
        """
        if sigma is None:
            sigma = self._sigma
        assert_debug(target_points is not None and reference_points is not None)
        check_sizes(target_points, [residuals.shape[0], residuals.shape[1], 3])
        check_sizes(reference_points, [residuals.shape[0], residuals.shape[1], 3])
        weights = torch.exp(- (target_points - reference_points).norm(dim=-1) ** 2 / sigma ** 2)
        cost = residuals * residuals * weights
        return cost


# ----------------------------------------------------------------------------------------------------------------------
class _GemanMcClure(_WLSScheme):
    """
    The Geman-McClure robust cost function
    """

    def __init__(self, sigma: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self._sigma = sigma

    def cost(self, residuals: torch.Tensor, sigma: Optional[float] = None, **kwargs) -> torch.Tensor:
        """
        Returns the weighted squared residuals
        """
        if sigma is None:
            sigma = self._sigma
        res2 = residuals ** 2
        cost = sigma * res2 / (sigma + res2)
        return cost


class _LS_SCHEME(Enum):
    """Weighting Schemes which increase Least Square robustness"""
    default = _LeastSquareScheme
    least_square = _LeastSquareScheme
    huber = _HuberScheme
    exp = _ExponentialScheme
    neighborhood = _NeighborhoodScheme
    geman_mcclure = _GemanMcClure

    @staticmethod
    def get(scheme: str, **kwargs) -> _WLSScheme:
        """Returns the Least Square Weighting scheme parameters"""
        assert_debug(scheme in _LS_SCHEME.__members__)
        return _LS_SCHEME.__members__[scheme].value(**kwargs)


# ----------------------------------------------------------------------------------------------------------------------
class LeastSquare(ABC):
    """An Abstract class for Least Square Minimization"""

    def __init__(self, scheme: str = "default", **kwargs):
        self._ls_scheme: _WLSScheme = _LS_SCHEME.get(scheme, **kwargs)

    @abstractmethod
    def compute(self, x0: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimates the optimal set of parameters for a LeastSquare problem

        Args:
            x0 (torch.Tensor): The initial set of parameters `(D,)`

        Returns:
            The tuple `x, loss` where `x` is the optimized set of parameters,
            And `loss` is the sum of the squared residuals
        """
        raise NotImplementedError("")


# ----------------------------------------------------------------------------------------------------------------------
class LinearLeastSquare(LeastSquare):
    """Linear Least Square estimation"""

    def compute(self, x0: torch.Tensor, A: torch.Tensor, b: torch.Tensor, *args, **kwargs):
        """
        Solve the Linear Least Square estimation A * x = b


        Args:
            x0 (torch.Tensor): The initial set of parameters `(B, D,)`
            A (torch.Tensor): The matrix A `(B, N, D)'
            b (torch.Tensor): The matrix b `(B, N,)`

        Returns:
            The tuple `x, loss` where `x` is the optimized set of parameters,
            And `loss` is the sum of the squared residuals
        """
        check_sizes(A, [-1, -1])
        n, d = A.shape
        check_sizes(b, [n])
        check_sizes(x0, [d])

        residuals = A @ x0 - b
        weights = self._ls_scheme.weights(residuals, **kwargs)

        Aw = torch.einsum("nd,n->nd", A, weights)
        bw = b * weights

        x = torch.solve(bw, Aw)
        loss = ((Aw @ x - bw) ** 2).sum()

        return x, loss


# ----------------------------------------------------------------------------------------------------------------------
class GaussNewton(LeastSquare):
    """Gauss Newton algorithm for Least Square minimization"""

    def __init__(self, max_iters: int = 10,
                 norm_stop_criterion: float = 1.e-3, **kwargs):
        super().__init__(**kwargs)
        self._max_iters: int = max(max_iters, 1)
        self._norm_stop_criterion: float = norm_stop_criterion

    def compute(self, x0: torch.Tensor, res_fun: callable,
                jac_fun: Optional[callable] = None, num_iters: Optional[int] = None, **kwargs):
        """
        Estimates via Gauss Newton the non linear least-square objective

        Args:
            x0 (torch.Tensor): The initial set of parameters `(B, D,)`
            res_fun (callable): The function (or closure) mapping x to the tensor or residuals of dimension `(B, N,)`
            jac_fun (callable): The jacobian function ie computes the jacobian of the LS system
                                (matrix of dimension `(B, N, D)`) from the parameters.
                                If not defined, pytorch auto-differentiation on the res_fun is used

        Returns:
            The tuple `x, loss` where `x` is the optimized set of parameters,
            And `loss` is the sum of the squared residuals
        """
        if num_iters is None:
            num_iters = self._max_iters

        if jac_fun is None:
            indices = [i for i in range(x0.shape[0])]
            jac_fun = lambda _x: Fa.jacobian(res_fun, _x, vectorize=True)[indices, :, indices]

        x = x0
        for _ in range(num_iters):
            J: torch.Tensor = jac_fun(x.detach())
            res: torch.Tensor = res_fun(x)
            weights: torch.Tensor = self._ls_scheme.weights(res.detach(), **kwargs)
            res *= weights
            J *= weights.unsqueeze(-1)

            Jt = J.permute(0, 2, 1)
            H = Jt @ J
            if torch.any(H.det().abs() < 1.e-7):
                logging.error("Invalid Jacobian in Gauss Newton minimization, the hessian is not invertible")
                raise RuntimeError("Invalid Jacobian in Gauss Newton minimization")

            dx = - H.inverse() @ Jt @ res.unsqueeze(-1)
            x = x + dx[:, :, 0]

            if dx.detach().norm() < self._norm_stop_criterion:
                break

        return x, res * res


# ----------------------------------------------------------------------------------------------------------------------
class PointToPlaneCost:
    """Point-to-Plane Cost function"""

    def __init__(self, ls_scheme: str = "default", pose: Pose = Pose("euler"), **kwargs):
        self.ls_scheme: _WLSScheme = _LS_SCHEME.get(ls_scheme, **kwargs)
        self.pose = pose

    @staticmethod
    def get_residual_jac_fun(target_points: torch.Tensor,
                             ref_points: torch.Tensor,
                             ref_normals: torch.Tensor,
                             pose: Pose = Pose("euler"),
                             mask: Optional[torch.Tensor] = None, **kwargs):
        """
        Returns the Point-to-Plane residual jacobian closure

        The returned closure takes input a pose matrix or pose params tensor,
        And returns the jacobian of the residual tensor at the pose_matrix position

        Args:
            target_points (torch.Tensor): The tensor of target points `(B, N, 3)`
            ref_points (torch.Tensor): The tensor of reference points `(B, N, 3)`
            ref_normals (torch.Tensor): The tensor of reference normals `(B, N, 3)`
            pose (Pose): The Pose representation
            mask: (torch.Tensor): An optional mask to filter out points
        """
        check_sizes(target_points, [-1, -1, 3])
        b, n, _ = target_points.shape
        check_sizes(ref_normals, [b, n, 3])
        check_sizes(ref_points, [b, n, 3])
        if mask is not None:
            check_sizes(mask, [b, n, 1])

        def __jac_fun(params: torch.Tensor):
            check_sizes(params, [b, pose.num_params()])
            jac_pose_to_matrix = pose.pose_matrix_jacobian(params)  # [B, 6, 4, 4]

            jacobians_rot = jac_pose_to_matrix[:, :, :3, :3]
            jacobians_trans = jac_pose_to_matrix[:, :, :3, 3]
            residuals_jac = torch.einsum("bpij,bnj->bpni", jacobians_rot, target_points) + jacobians_trans.unsqueeze(2)
            normals_unsqueeze = ref_normals.unsqueeze(1)
            residuals_jac = (residuals_jac * normals_unsqueeze).sum(dim=3)
            residuals_jac = residuals_jac.reshape(b, pose.num_params(), n).permute(0, 2, 1)  # [B, N, 6]
            if mask is not None:
                residuals_jac *= mask.unsqueeze(1)

            return residuals_jac

        return __jac_fun

    @staticmethod
    def get_residual_fun(target_points: torch.Tensor,
                         ref_points: torch.Tensor,
                         ref_normals: torch.Tensor,
                         pose: Pose = Pose("euler"),
                         mask: Optional[torch.Tensor] = None, **kwargs):
        """
        Returns the Point-to-Plane residual closure

        The returned closure takes input a pose matrix or pose params tensor,
        And returns a tensor of corresponding residuals

        Args:
            target_points (torch.Tensor): The tensor of target points `(B, N, 3)`
            ref_points (torch.Tensor): The tensor of reference points `(B, N, 3)`
            ref_normals (torch.Tensor): The tensor of reference normals `(B, N, 3)`
            pose (Pose): The Pose representation
            mask: (torch.Tensor): An optional mask to filter out points
        """
        check_sizes(target_points, [-1, -1, 3])
        b, n, _ = target_points.shape
        check_sizes(ref_normals, [b, n, 3])
        check_sizes(ref_points, [b, n, 3])
        if mask is not None:
            check_sizes(mask, [b, n, 1])

        def __residual_fun(params: torch.Tensor):
            check_sizes(params, [b, pose.num_params()])
            matrices = pose.build_pose_matrix(params)
            transformed_points = pose.apply_transformation(target_points, matrices)

            residuals = (transformed_points - ref_points) * ref_normals
            if mask is not None:
                residuals *= mask
            residuals = residuals.sum(dim=-1)
            return residuals

        return __residual_fun

    def residuals(self, target_points: torch.Tensor, pose_params: torch.Tensor,
                  ref_points: torch.Tensor, ref_normals: torch.Tensor,
                  mask: Optional[torch.Tensor] = None, **kwargs):
        """Returns the point to plane residuals"""
        residuals = PointToPlaneCost.get_residual_fun(target_points,
                                                      ref_points,
                                                      ref_normals,
                                                      self.pose,
                                                      mask)(pose_params)
        weights = self.ls_scheme.weights(residuals.detach().abs(),
                                         target_points=target_points,
                                         ref_points=ref_points)
        return weights * residuals

    def loss(self, target_points: torch.Tensor, pose_params: torch.Tensor,
             ref_points: torch.Tensor, ref_normals: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs):
        """Returns the Point-to-Plane Loss"""
        return self.residuals(target_points, pose_params, ref_points, ref_normals, mask)
