import logging
import re
from abc import ABC
from enum import Enum
from typing import Optional, Union

import numpy as np

# Hydra and OmegaConf
from hydra.conf import dataclass, MISSING
from omegaconf import DictConfig

# Project Imports
from slam.common.modules import _with_g2o
from slam.common.utils import assert_debug, check_tensor, ObjectLoaderEnum
from slam.eval.eval_odometry import compute_relative_poses


# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class BackendConfig:
    """An abstract class for the Config for a Backend"""
    type: str = MISSING


class Backend(ABC):
    """An abstract class for the backend of a SLAM algorithm

    Given a set of trajectory constraints, the backend performs a global optimization
    Taking into account the different constraints and their uncertainties
    """

    def __init__(self, config: BackendConfig, **kwargs):
        self.config = config
        self._constraints: Optional[dict] = None
        self.need_to_update_pose: bool = False

    def need_to_synchronise_poses(self):
        if self.need_to_update_pose:
            self.need_to_update_pose = False
            return True

    def init(self):
        """Clears the current representation and prepare a new model"""
        self.clear()
        self._constraints = {
            "se3_odometry": [],
            "se3_loop_closure": [],
            "se3_absolute": []
        }

    def clear(self):
        """Clears the current data stored in the backend"""
        raise NotImplementedError("")

    def world_poses(self) -> np.ndarray:
        """Returns the poses expressed in the world frame"""
        raise NotImplementedError("")

    def absolute_poses(self) -> np.ndarray:
        """Returns the poses expressed in the first pose frame

        Note: Should differ from world_poses only when additional information on the trajectory are known
              e.g. If some GPS constraints are factored in
        """
        raise NotImplementedError("")

    def relative_odometry_poses(self):
        """Returns the relative poses between two consecutive frames corrected after global optimization"""
        raise NotImplementedError("")

    def next_frame(self, data_dict: dict):
        """Processes a next frame"""
        raise NotImplementedError("")

    @staticmethod
    def _regexes():
        """Regex for the search of relative constraints in the new frame dict"""
        return "^se3_odometry_constraint_([\\d]+)$", \
               "^se3_loop_closure_constraint_([\\d]+)_([\\d]+)$", \
               "^se3_absolute_constraint_([\\d]+)$"

    @staticmethod
    def se3_odometry_constraint(reference_idx: int):
        """Returns a key which defines a relative constraint between two consecutive poses

        Backends will search the `dict` of a new frame for keys following
        the pattern r"se3_odometry_constraint_([d]+)" to define relative trajectory constraints between
        Two Poses.

        Args:
            reference_idx: The index of the reference Pose
        """
        return f"se3_odometry_constraint_{int(reference_idx)}"

    @staticmethod
    def se3_loop_closure_constraint(reference_idx: int, tgt_idx: int):
        """Returns a key which defines a relative constraint between two poses"""
        return f"se3_loop_closure_constraint_{int(reference_idx)}_{int(tgt_idx)}"

    @staticmethod
    def se3_absolute_constraint(reference_idx: int):
        """Returns a key which defines an absolute constraint between poses"""
        return f"se3_absolute_constraint_{int(reference_idx)}"

    def search_constraints(self, data_dict: dict) -> dict:
        """Returns a set of constraints read from `data_dict`"""
        constraints = {
            "se3_odometry": [],
            "se3_loop_closure": [],
            "se3_absolute": []
        }
        for key in data_dict.keys():
            reg_odom, reg_loop, reg_abs = self._regexes()

            # Search se3 constraints
            m = re.search(reg_odom, key)
            if m is not None:
                id0 = int(m.group(1))
                matrix, quat_information = data_dict[key]
                assert_debug(isinstance(matrix, np.ndarray))
                constraints["se3_odometry"].append((id0, matrix, quat_information))

            m = re.search(reg_loop, key)
            if m is not None:
                id0 = int(m.group(1))
                id1 = int(m.group(2))
                matrix, quat_information = data_dict[key]
                assert_debug(isinstance(matrix, np.ndarray))
                constraints["se3_loop_closure"].append((id0, id1, matrix, quat_information))

            m = re.search(reg_abs, key)
            if m is not None:
                id0 = int(m.group(1))
                matrix, quat_information = data_dict[key]
                assert_debug(isinstance(matrix, np.ndarray))
                constraints["se3_absolute"].append((id0, matrix, quat_information))

        constraints["se3_odometry"] = sorted(constraints["se3_odometry"], key=lambda x: x[0])

        self._constraints["se3_odometry"] += constraints["se3_odometry"]
        self._constraints["se3_loop_closure"] += constraints["se3_loop_closure"]
        self._constraints["se3_absolute"] += constraints["se3_absolute"]

        return constraints

    def registered_loop_constraints(self):
        """Returns the registered loop constraints"""
        if self._constraints is None:
            return []
        return self._constraints["se3_loop_closure"]

    def registered_odometry_constraints(self):
        """Returns the registered odometry constraints"""
        if self._constraints is None:
            return []
        return self._constraints["se3_odometry"]

    def registered_absolute_constraints(self):
        """Returns the registered absolute constraints"""
        if self._constraints is None:
            return []
        return self._constraints["se3_absolute"]


# ----------------------------------------------------------------------------------------------------------------------

if _with_g2o:
    import g2o

    from slam.viz import _with_viz3d

    if _with_viz3d:
        from viz3d.window import OpenGLWindow


    @dataclass
    class GraphSLAMConfig(BackendConfig):
        type = "graph_slam"
        initialize_world_coordinates: bool = True
        fix_first_frame: bool = True
        max_optim_iterations: int = 100
        online_optimization: bool = True

        debug: bool = False


    class GraphSLAM(Backend):
        """A PoseGraph backend which maintains and performs global optimization

        Args:
            Whether to initialize the world coordinates at the first frame
        """

        def __init__(self, config: GraphSLAMConfig, **kwargs):
            super().__init__(config)
            self.optimizer: g2o.SparseOptimizer = None

            self.vertices: set = None  # The set of vertices indices
            self._num_poses: int = 0

            self.initialize_wc = config.initialize_world_coordinates
            self.fix_first_frame = config.fix_first_frame
            self.max_iterations = config.max_optim_iterations

            self.robust_kernel: g2o.RobustKernel = None  # g2o.RobustKernelGemanMcClure()

            self.odometry_poses: Optional[list] = None
            self.loop_closure_constraints: Optional[dict] = None
            self.absolute_pose_constraints: Optional[dict] = None

            self.window = None
            self.with_window = config.debug and _with_viz3d

        @property
        def robust_kernel(self):
            """A Robust Kernel for Least-Square minimization"""
            return self._robust_kernel

        @robust_kernel.setter
        def robust_kernel(self, robust_kernel: Optional[g2o.BaseRobustKernel]):
            assert_debug(robust_kernel is None or isinstance(robust_kernel, g2o.BaseRobustKernel))
            self._robust_kernel = robust_kernel

        def __del__(self):
            self.clear()

        def clear(self):
            self.optimizer = None
            self.vertices = None
            self.odometry_poses = None
            self._num_poses = 0
            if _with_viz3d:
                if self.window is not None:
                    self.window.close(True)
                    self.window = None

        def init(self):
            super().init()
            self.optimizer = g2o.SparseOptimizer()
            solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
            solver = g2o.OptimizationAlgorithmLevenberg(solver)
            self.optimizer.set_algorithm(solver)

            self.vertices: set = set()  # The set of vertices indices

            self.loop_closure_constraints = dict()
            self.absolute_pose_constraints = dict()
            self._num_poses = 0

            if self.initialize_wc:
                # Adds an initial vertex
                self.__add_vertex(self.param_vid(0), np.eye(4, dtype=np.float64),
                                  self.fix_first_frame)
                self.odometry_poses = [np.eye(4)]

            if self.with_window:
                self.window = OpenGLWindow()
                self.window.init()

        def __add_vertex(self, v_id, pose: np.ndarray, fixed: bool = False, gps_vertex: bool = False):
            assert_debug(v_id not in self.vertices)
            check_tensor(pose, [4, 4])
            v_se3 = g2o.VertexSE3()
            v_se3.set_id(v_id)
            v_se3.set_estimate(
                g2o.Isometry3d(pose))  # The pose representation is (x, y, z, qx, qy, qz)
            v_se3.set_fixed(fixed)
            self.optimizer.add_vertex(v_se3)
            self.vertices.add(v_id)
            if not gps_vertex:
                self._num_poses += 1

        def gps_vid(self, pose_id: int):
            """Returns the vertex index in the pose graph of the gps vertex corresponding to a pose index"""
            return 2 * pose_id

        def param_vid(self, pose_id: int):
            """Returns the vertex index in the pose graph of the param block given a pose index"""
            return 2 * pose_id + 1

        def _get_pose(self, v_id):
            assert_debug(v_id in self.vertices)
            pose = self.optimizer.vertex(v_id).estimate().matrix()  # [4, 4]
            return pose

        def next_frame(self, data_dict: dict):
            """Processes a next frame by adding all trajectory constraints to the graph slam"""
            assert isinstance(self.config, GraphSLAMConfig)
            constraints = self.search_constraints(data_dict)

            do_update: bool = False

            _relative_constraints = []

            # Add New vertices (from odometry constraints)
            for constraint in constraints["se3_odometry"]:
                i, mat, information = constraint

                i_pid = self.param_vid(i)
                i_1_pid = self.param_vid(i + 1)
                if i_1_pid not in self.vertices:
                    assert_debug(self.param_vid(i) in self.vertices)
                    # Add a pose between two updates
                    pose_i_p = self.odometry_poses[-1].dot(mat)
                    self.__add_vertex(i_1_pid, self._get_pose(i_pid).dot(mat).astype(np.float64))
                    self.odometry_poses.append(pose_i_p)

                _relative_constraints.append((i_pid, i_1_pid, mat, information))

            # Add Absolute poses Vertices from absolute constraints
            for constraint in constraints["se3_absolute"]:
                i, mat, information = constraint

                i_gps_id = self.gps_vid(i)
                i_pid = self.param_vid(i)
                assert_debug(i_pid in self.vertices)
                assert_debug(i_gps_id not in self.vertices)

                self.__add_vertex(i_gps_id, mat, True, gps_vertex=True)
                if information is None:
                    information = np.eye(6, dtype=np.float64)
                    information[:3, :3] = 1.0  # 1.0 m of error for pose obtained by the GPS
                    information[3:, 3:] = 0.001  # High Covariance for the orientation

                # GPS Constraints are at Identity
                identity = np.eye(4, dtype=np.float64)
                _relative_constraints.append((i_gps_id, i_pid, identity, information))

            # Add Loop Closure Constraints
            for constraint in constraints["se3_loop_closure"]:
                i, j, mat_j_to_i, information = constraint
                self.loop_closure_constraints[(i, j)] = (mat_j_to_i, information)

                i_pid = self.param_vid(i)
                j_pid = self.param_vid(j)

                assert_debug(i_pid in self.vertices and j_pid in self.vertices)
                _relative_constraints.append((i_pid, j_pid, mat_j_to_i, information))

            # Add Edge Constraints
            _constraints_indices = []
            for constraint in _relative_constraints:
                i, j, mat_j_to_i, information = constraint

                assert_debug(i < j)
                if information is None:
                    if abs(i - j) < 10:
                        # High confidence in Odometry
                        information = np.eye(6)
                        information[:3, :3] *= 2
                        information[3:, 3:] *= 5
                    else:
                        # Low confidence in Loop Closure
                        information = np.eye(6)
                        information[:3, :3] *= 0.1
                        information[3:, 3:] *= 0.5

                assert_debug(i in self.vertices)

                # Add Edge i<->j
                edge = g2o.EdgeSE3()
                edge.set_vertex(0, self.optimizer.vertex(i))
                edge.set_vertex(1, self.optimizer.vertex(j))

                iso_j_to_i = g2o.Isometry3d(mat_j_to_i)
                edge.set_measurement(g2o.Isometry3d(iso_j_to_i))
                edge.set_information(information.astype(np.float64))

                if self.robust_kernel is not None:
                    edge.set_robust_kernel(self.robust_kernel)

                self.optimizer.add_edge(edge)

                # Only update when a loop constraint is added
                # TODO : Find better update criterions
                if abs(i - j) > 2:
                    do_update = True

                _constraints_indices.append([i, j])

            if do_update:
                logging.info(f"Updating the Pose Graph for {self.config.max_optim_iterations} iterations.")
                self.optimize(self.config.max_optim_iterations)
                self.need_to_update_pose = True

                if self.with_window and self.window is not None:
                    self.window.set_cameras(0, self.absolute_poses().astype(np.float32))

        def optimize(self, max_num_epochs: int = 20):
            if not self.config.online_optimization:
                # Reset all poses to odometry poses
                for idx in range(1, len(self.odometry_poses)):
                    self.optimizer.vertex(self.param_vid(idx)).set_estimate(g2o.Isometry3d(self.odometry_poses[idx]))
            self.optimizer.initialize_optimization()
            self.optimizer.optimize(max_num_epochs)

        def world_poses(self) -> np.ndarray:
            return self.absolute_poses()

        def absolute_poses(self) -> np.ndarray:
            poses = np.zeros((self._num_poses, 4, 4), dtype=np.float64)
            for idx in range(self._num_poses):
                pid = self.param_vid(idx)
                assert_debug(pid in self.vertices)
                poses[idx] = self._get_pose(pid)
            return poses

        def relative_odometry_poses(self):
            return compute_relative_poses(self.absolute_poses())


# ----------------------------------------------------------------------------------------------------------------------
class BACKEND(ObjectLoaderEnum, Enum):
    if _with_g2o:
        graph_slam = (GraphSLAM, GraphSLAMConfig)

    none = (None, None)

    @classmethod
    def type_name(cls):
        return "type"
