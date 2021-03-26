from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn

# Hydra and OmegaConf
from hydra.conf import dataclass, MISSING, field

# Project Imports
from hydra.core.config_store import ConfigStore

from slam.common.geometry import compute_normal_map, projection_map_to_points
from slam.common.optimization import _LS_SCHEME, _WLSScheme, PointToPlaneCost
from slam.common.pose import Pose
from slam.common.projection import Projector
from slam.common.utils import assert_debug, check_sizes


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class LossConfig:
    """Abstract Loss Config for training PoseNet"""
    mode: str = MISSING


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class PointToPlaneLossConfig(LossConfig):
    """Unsupervised Point-to-Plane Loss Config"""
    mode: str = "unsupervised"

    least_square_scheme: Optional[Dict[str, Any]] = field(default_factory=lambda: dict(scheme="geman_mcclure",
                                                                                       sigma=0.5))


# ----------------------------------------------------------------------------------------------------------------------
# Point-To-Plane Loss Module for unsupervised training of PoseNet
class _PointToPlaneLossModule(nn.Module):
    """
    Point-to-Plane Loss Module
    """

    def __init__(self, config: PointToPlaneLossConfig, projector: Projector, pose: Pose):
        nn.Module.__init__(self)
        self.pose = pose
        self.projector = projector
        self.config = config
        self._ls_scheme = _LS_SCHEME.get(**self.config.least_square_scheme)

    def point_to_plane_loss(self,
                            vm_target,
                            vm_reference,
                            nm_reference,
                            pose_tensor,
                            data_dict: dict):
        """
        Computes the Point-to-Plane loss between a target vertex map and a reference

        Parameters
        ----------
        vm_target: torch.Tensor
            The vertex map tensor
        vm_reference: torch.Tensor
            The vertex map tensor
        nm_reference: torch.Tensor
            The normal map tensor
        pose_tensor: torch.Tensor
            The relative pose parameters or transform matrix to apply on the target point cloud
        data_dict:
            The dictionary to add tensor for logging

        Returns
        -------
        The point-to-plane loss between the reference and the target

        """
        b, _, h, w = vm_target.shape
        target_pc = vm_target.permute(0, 2, 3, 1).reshape(b, h * w, 3)

        mask_vm = (target_pc.norm(dim=2, keepdim=True) != 0.0)
        pc_transformed_target = self.pose.apply_transformation(target_pc, pose_tensor)

        # Mask out null points which will be transformed by the pose
        pc_transformed_target = pc_transformed_target * mask_vm
        vm_transformed = self.projector.build_projection_map(pc_transformed_target, height=h, width=w)

        # Transform to point clouds to compute the point-to-plane error
        pc_transformed = projection_map_to_points(vm_transformed)
        pc_reference = projection_map_to_points(vm_reference)
        normal_reference = projection_map_to_points(nm_reference)

        mask = ~(normal_reference.norm(dim=-1) == 0.0)
        mask *= ~(pc_reference.norm(dim=-1) == 0.0)
        mask *= ~(pc_transformed.norm(dim=-1) == 0.0)
        mask = mask.detach().to(torch.float32)

        residuals = mask * ((pc_reference - pc_transformed) * normal_reference).sum(dim=-1).abs()

        cost = self._ls_scheme.cost(residuals, target_points=pc_transformed, reference_points=pc_reference)
        loss_icp = ((cost * cost).sum(dim=1) / mask.sum(dim=1)).mean()

        return loss_icp

    def forward(self, data_dict: dict):
        vertex_map = data_dict["vertex_map"]
        if "normal_map" not in data_dict:
            check_sizes(vertex_map, [-1, 2, 3, -1, -1])
            b, seq, _, h, w = vertex_map.shape
            normal_map = compute_normal_map(vertex_map.view(b * seq, 3, h, w)).view(b, seq, 3, h, w)
            data_dict["normal_map"] = normal_map
        normal_map = data_dict["normal_map"]

        b, s, _, h, w = vertex_map.shape
        assert_debug(s == 2)

        tgt_vmap = vertex_map[:, 1]
        ref_vmap = vertex_map[:, 0]
        ref_nmap = normal_map[:, 0]

        tgt_to_ref = data_dict["pose_params"]
        if tgt_to_ref.size(-1) != 4:
            tgt_to_ref = self.pose.build_pose_matrix(tgt_to_ref)

        # Compute the 3D Point-to-Plane
        loss_icp = self.point_to_plane_loss(tgt_vmap, ref_vmap, ref_nmap, tgt_to_ref, data_dict).mean()
        loss = loss_icp

        return loss, data_dict


# ----------------------------------------------------------------------------------------------------------------------
# ExponentialWeights for Supervised Loss Module
class ExponentialWeights(nn.Module):
    """
    A Module which exponentially weights different losses during training

    It holds parameters weigh the different losses.
    The weights change during training, as they are concerned by the the gradient descent
    For n losses, the computed loss is :
    $$ loss = \\sum_{k=1}^n loss_i * e^{s_i} + s_i $$

    Parameters
    ----------
    num_losses : int
        The number of losses (and parameters)
    init_weights : list
        The initial weights for the parameters
    """

    def __init__(self, num_losses: int, init_weights: list):
        nn.Module.__init__(self)
        assert_debug(len(init_weights) == num_losses)

        self.s_param = torch.nn.Parameter(torch.tensor(init_weights), requires_grad=True)
        self.num_losses = num_losses

    def forward(self, list_losses: list) -> (torch.Tensor, list):
        """
        Computes the exponential weighing of the losses in list_losses

        Parameters
        ----------
        list_losses : list
            The losses to weigh. Expects a list of self.num_losses torch.Tensor scalars

        Returns
        -------
        tuple (torch.Tensor, list)
            The weighted loss, and the list of parameters
        """
        assert_debug(len(list_losses) == self.num_losses)

        s_params = []
        loss = 0.0

        for i in range(self.num_losses):
            loss_item = list_losses[i]
            s_param = self.s_param[i]
            exp_part_loss = loss_item * torch.exp(-s_param) + s_param
            loss += exp_part_loss
            s_params.append(s_param.detach())

        return loss, s_params


# ----------------------------------------------------------------------------------------------------------------------
# Config for Supervised Loss Module
@dataclass
class SupervisedLossConfig(LossConfig):
    """Config for the supervised loss module of PoseNet"""
    mode: str = "supervised"

    loss_degrees: bool = True  # Whether to express the rotation loss in degrees (True) or radians (False)

    # The weights of rotation and translation losses
    loss_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])

    # Exponential Weighting Params
    # Parameters for adaptive scaling of rotation and translation losses during training
    with_exp_weights: bool = False
    init_weights: List[float] = field(default_factory=lambda: [-3.0, -3.0])

    # Loss option (l1, l2)
    loss_option: str = "l2"


# ----------------------------------------------------------------------------------------------------------------------
# Supervised Loss Module
class _PoseSupervisionLossModule(nn.Module):
    """
    Supervised Loss Module
    """

    def __init__(self, config: SupervisedLossConfig, pose: Pose):
        super().__init__()
        self.config = config
        self.pose = pose
        self.euler_pose = Pose("euler")

        self.exp_weighting: Optional[ExponentialWeights] = None
        self.weights: Optional[list] = None
        self.degrees = self.config.loss_degrees
        if self.config.with_exp_weights:
            self.exp_weighting = ExponentialWeights(2, self.config.init_weights)
        else:
            self.weights = self.config.loss_weights
            assert_debug(len(self.weights) == 2)
        loss = self.config.loss_option

        assert_debug(loss in ["l1", "l2"])
        self.loss_config = loss

    def __l1(self, x, gt_x):
        return (x - gt_x).abs().sum(dim=1).mean()

    def __loss(self, x, gt_x):
        if self.loss_config == "l1":
            return self.__l1(x, gt_x)
        elif self.loss_config == "l2":
            return ((x - gt_x) * (x - gt_x)).sum(dim=1).mean()
        else:
            raise NotImplementedError("")

    def forward(self, data_dict: dict) -> (torch.Tensor, dict):
        pose_params = data_dict["pose_params"]
        ground_truth = data_dict["ground_truth"]

        if self.degrees:
            euler_pose_params = self.euler_pose.from_pose_matrix(self.pose.build_pose_matrix(pose_params))
            gt_params = self.euler_pose.from_pose_matrix(ground_truth)

            pred_degrees = (180.0 / np.pi) * euler_pose_params[:, 3:]
            gt_degrees = (180.0 / np.pi) * gt_params[:, 3:]
            loss_rot = self.__loss(pred_degrees, gt_degrees)

            data_dict["loss_rot_l1"] = self.__l1(pred_degrees, gt_degrees).detach()
        else:
            gt_params = self.pose.from_pose_matrix(ground_truth)
            loss_rot = self.__loss(gt_params[:, 3:], pose_params[:, 3:])
            data_dict["loss_rot_l1"] = self.__l1(gt_params[:, 3:], pose_params[:, 3:]).detach()

        loss_trans = self.__loss(pose_params[:, :3], gt_params[:, :3])
        loss_trans_l1 = self.__l1(pose_params[:, :3], gt_params[:, :3])

        data_dict["loss_rot"] = loss_rot
        data_dict["loss_trans"] = loss_trans
        data_dict["loss_trans_l1"] = loss_trans_l1

        loss = 0.0
        if self.exp_weighting:
            loss, s_param = self.exp_weighting([loss_trans, loss_rot])
            data_dict["s_rot"] = s_param[1]
            data_dict["s_trans"] = s_param[0]
        else:
            loss = loss_trans * self.weights[0] + loss_rot * self.weights[1]

        data_dict["loss"] = loss
        return loss, data_dict


# ------------------------------------------------------------
# Hydra -- Add the config group for the different Loss options
cs = ConfigStore.instance()
cs.store(group="training/loss", name="supervised", node=SupervisedLossConfig)
cs.store(group="training/loss", name="unsupervised", node=PointToPlaneLossConfig)
