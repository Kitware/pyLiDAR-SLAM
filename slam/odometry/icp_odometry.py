# Project Imports
from typing import Optional

from hydra.core.config_store import ConfigStore

from slam.common.utils import RuntimeDefaultDict
from slam.common.geometry import projection_map_to_points, mask_not_null
from slam.common.pose import Pose
from slam.common.projection import Projector
from slam.common.modules import _with_viz3d
from slam.dataset import DatasetLoader
from slam.common.utils import check_tensor, remove_nan, modify_nan_pmap
from slam.odometry.alignment import RigidAlignmentConfig, RIGID_ALIGNMENT, RigidAlignment
from slam.odometry.initialization import InitializationConfig, INITIALIZATION, Initialization
from slam.odometry.odometry import *
from slam.odometry.local_map import LOCAL_MAP, LocalMapConfig, LocalMap
from slam.preprocessing.preprocessing import PreprocessingConfig, Preprocessing
from slam.viz.color_map import *

if _with_viz3d:
    from viz3d.window import OpenGLWindow


# ----------------------------------------------------------------------------------------------------------------------
@RuntimeDefaultDict.runtime_defaults({"initialization": "slam/odometry/initialization/CV",
                                      "local_map": "slam/odometry/local_map/kdtree",
                                      "alignment": "slam/odometry/alignment/point_to_plane_GN"
                                      })
@dataclass
class ICPFrameToModelConfig(OdometryConfig, RuntimeDefaultDict):
    """
    The Configuration for the Point-To-Plane ICP based Iterative Least Square estimation of the pose
    """
    algorithm: str = "icp_F2M"
    device: str = "cpu"
    pose: str = "euler"
    max_num_alignments: int = 100

    # Config for the Initialization
    initialization: InitializationConfig = MISSING

    # Config for the Local Map
    local_map: LocalMapConfig = MISSING

    # Config for the Rigid Alignment
    alignment: RigidAlignmentConfig = MISSING

    threshold_delta_pose: float = 1.e-4
    threshold_trans: float = 0.1
    threshold_rot: float = 0.3
    sigma: float = 0.1

    # The data key which is used to search into the data dictionary for the pointcloud to register onto the new frame
    data_key: str = "vertex_map"

    viz_debug: bool = True  # Whether to display the FM in a window (if exists)

    # Visualization parameters
    viz_with_edl: bool = True
    viz_num_pcs: int = 50


cs = ConfigStore.instance()
cs.store(name="icp_odometry", group="slam/odometry", node=ICPFrameToModelConfig)


# ----------------------------------------------------------------------------------------------------------------------
class ICPFrameToModel(OdometryAlgorithm):
    """
    OdometryAlgorithm based on the ICP-registration
    """

    def __init__(self, config: ICPFrameToModelConfig,
                 projector: Projector = None, pose: Pose = Pose("euler"),
                 device: torch.device = torch.device("cpu"), **kwargs):
        if not isinstance(config, ICPFrameToModelConfig):
            config = ICPFrameToModelConfig(**config)
        config = config.completed()
        OdometryAlgorithm.__init__(self, config)

        assert_debug(projector is not None)
        self.pose = pose
        self.projector = projector
        self.device = device

        # --------------------------------
        # Loads Components from the Config

        self._motion_model: Initialization = INITIALIZATION.load(self.config.initialization,
                                                                 pose=self.pose, device=device)
        self.local_map: LocalMap = LOCAL_MAP.load(self.config.local_map,
                                                  pose=self.pose, projector=projector)

        assert isinstance(self.config, ICPFrameToModelConfig)
        self.config.alignment.pose = self.pose.pose_type
        self.rigid_alignment: RigidAlignment = RIGID_ALIGNMENT.load(self.config.alignment, pose=self.pose)

        # self._post_processing:

        # -----------------------
        # Optimization Parameters
        self.gn_max_iters = self.config.max_num_alignments
        self._sample_pointcloud: bool = False

        # ---------------------
        # Local state variables
        self.relative_poses: list = []
        self.absolute_poses: list = []  # Absolute poses (/!\ type: np.float64)
        self.gt_poses: Optional[np.ndarray] = None  # Ground Truth poses
        self._iter = 0
        self._tgt_vmap: torch.Tensor = None
        self._tgt_pc: torch.Tensor = None
        self._tgt_nmap: torch.Tensor = None
        self._delta_since_map_update = None  # delta pose since last estimate update
        self._register_threshold_trans = self.config.threshold_trans
        self._register_threshold_rot = self.config.threshold_rot

        self.viz3d_window: Optional[OpenGLWindow] = None
        self._has_window = config.viz_debug and _with_viz3d

    def __del__(self):
        if self._has_window:
            if self.viz3d_window is not None:
                self.viz3d_window.close(True)

    def init(self):
        """Initialize/ReInitialize the state of the Algorithm and its components"""
        super().init()
        self.relative_poses = []
        self.absolute_poses = []
        self.gt_poses = None

        self.local_map.init()
        self._motion_model.init()
        self._iter = 0
        self._delta_since_map_update = torch.eye(4, dtype=torch.float32, device=self.device).reshape(1, 4, 4)

        if self._has_window:
            if self.viz3d_window is not None:
                self.viz3d_window.close(True)
                self.viz3d_window = None
            self.viz3d_window = OpenGLWindow(
                engine_config={"with_edl": self.config.viz_with_edl, "edl_strength": 1000.0})
            self.viz3d_window.init()

    # ------------------------------------------------------------------------------------------------------------------
    def do_process_next_frame(self, data_dict: dict):
        """
        Processes a new frame

        Estimates the motion for the new frame, and update the states of the different components
        (Local Map, Initialization)

        Args:
            data_dict (dict): The input frame to be processed.
                              The key 'self.config.data_key' is required
        """
        # Reads the input frame
        self._read_input(data_dict)

        if self._iter == 0:
            # Initiate the map with the first frame
            relative_pose = torch.eye(4, dtype=torch.float32,
                                      device=self._tgt_vmap.device).unsqueeze(0)

            self.local_map.update(relative_pose,
                                  new_vertex_map=self._tgt_vmap)
            self.relative_poses.append(relative_pose.cpu().numpy())
            self.absolute_poses.append(relative_pose.cpu().to(torch.float64).numpy()[0])
            self._iter += 1
            return

        # Extract initial estimate
        initial_estimate = self._motion_model.next_initial_pose(data_dict)

        sample_points = self.sample_points()

        # Registers the new frame onto the map
        new_rpose_params, new_rpose, losses = self.register_new_frame(sample_points,
                                                                      initial_estimate,
                                                                      data_dict=data_dict)

        # Update initial estimate
        self.update_initialization(new_rpose, data_dict)
        self.__update_map(new_rpose, data_dict)

        # Update Previous pose
        np_new_rpose = new_rpose.cpu().numpy()
        self.relative_poses.append(np_new_rpose)

        latest_pose = self.absolute_poses[-1].dot(
            self.pose.build_pose_matrix(new_rpose_params.cpu().to(torch.float64).reshape(1, 6))[0].numpy())
        self.absolute_poses.append(latest_pose)

        tgt_np_pc = self._tgt_pc.cpu().numpy().reshape(-1, 3)

        if self._has_window:
            # Add Ground truth poses (mainly for visualization purposes)
            if DatasetLoader.absolute_gt_key() in data_dict:
                pose_gt = data_dict[DatasetLoader.absolute_gt_key()].reshape(1, 4, 4).cpu().numpy()
                self.gt_poses = pose_gt if self.gt_poses is None else np.concatenate(
                    [self.gt_poses, pose_gt], axis=0)

            # Apply absolute pose to the pointcloud
            world_points = np.einsum("ij,nj->ni", latest_pose[:3, :3].astype(np.float32), tgt_np_pc)
            world_points += latest_pose[:3, 3].reshape(1, 3).astype(np.float32)
            self.viz3d_window.set_pointcloud(self._iter % self.config.viz_num_pcs, world_points)
            # Follow Camera
            camera_pose = latest_pose.astype(np.float32).dot(np.array([[1.0, 0.0, 0.0, 0.0],
                                                                       [0.0, 1.0, 0.0, 0.0],
                                                                       [0.0, 0.0, 1.0, 60.0],
                                                                       [0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
            self.viz3d_window.update_camera(camera_pose)

            if self.gt_poses is not None and len(self.gt_poses) > 0:
                # Update Pose to the pointcloud
                self.viz3d_window.set_poses(-1, self.gt_poses.astype(np.float32))

        # Update Dictionary with pointcloud and pose
        data_dict[self.pointcloud_key()] = tgt_np_pc
        data_dict[self.relative_pose_key()] = np_new_rpose.reshape(4, 4)

        self._iter += 1

    def register_new_frame(self,
                           target_points: torch.Tensor,
                           initial_estimate: Optional[torch.Tensor] = None,
                           data_dict: Optional[dict] = None,
                           **kwargs) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Registers a new frame against the Local Map

        Args:
            target_points (torch.Tensor): The target Ver
            initial_estimate (Optional[torch.Tensor]): The initial motion estimate for the ICP
            data_dict (dict): The dictionary containing the data of the new frame

        Returns
            pose_matrix (torch.Tensor): The relative pose between the current frame and the map `(1, 4, 4)`

        """
        new_pose_matrix = initial_estimate
        new_pose_params = torch.zeros(self.pose.num_params(), device=target_points.device, dtype=target_points.dtype)
        if initial_estimate is None:
            new_pose_matrix = torch.eye(4, device=target_points.device,
                                        dtype=target_points.dtype).unsqueeze(0)

        losses = []

        old_target_points = target_points
        for _ in range(self.gn_max_iters):
            target_points = self.pose.apply_transformation(old_target_points.unsqueeze(0), new_pose_matrix)[0]

            # Compute the nearest neighbors for the selected points
            result: LocalMap.NeighborhoodResult = self.local_map.nearest_neighbor_search(target_points)
            neigh_pc = result.neighbor_points
            neigh_normals = result.neighbor_normals
            tgt_pc = result.new_target_points

            # Compute the rigid transform alignment
            delta_pose_matrix, delta_pose, residuals = self.rigid_alignment.align(neigh_pc,
                                                                                  tgt_pc,
                                                                                  neigh_normals,
                                                                                  **kwargs)

            loss = residuals.sum()
            losses.append(loss)

            if delta_pose.norm() < self.config.threshold_delta_pose:
                break

            # Manifold normalization to keep proper rotations
            new_pose_params = self.pose.from_pose_matrix(delta_pose_matrix @ new_pose_matrix)
            new_pose_matrix = self.pose.build_pose_matrix(new_pose_params)

        return new_pose_params, new_pose_matrix, losses

    def sample_points(self):
        """Returns the points sampled"""
        if not self._sample_pointcloud:
            target_points = projection_map_to_points(self._tgt_vmap[0], dim=0)
            target_points = target_points[target_points.norm(dim=-1) > 0.0]
        else:
            target_points = self._tgt_pc[0]
        return target_points

    def get_relative_poses(self) -> np.ndarray:
        """Returns the estimated relative poses for the current sequence"""
        if len(self.relative_poses) == 0:
            return None
        return np.concatenate(self.relative_poses, axis=0)

    def update_initialization(self, new_rpose, data_dict: dict):
        """Send the frame to the initialization after registration for its state update"""
        self._motion_model.register_motion(new_rpose, data_dict)

    # ------------------------------------------------------------------------------------------------------------------
    # `Private` methods

    def _read_input(self, data_dict: dict):
        """Reads and interprets the input from the data_dict"""
        assert_debug(self.config.data_key in data_dict,
                     f"Could not find the key `{self.config.data_key}` in the input dictionary.\n"
                     f"With keys : {data_dict.keys()}). Set the parameter `slam.odometry.data_key` to the desired key")
        data = data_dict[self.config.data_key]

        self._tgt_vmap = None
        self._tgt_pc = None
        if isinstance(data, np.ndarray):
            check_tensor(data, [-1, 3])
            self._sample_pointcloud = True
            pc_data = torch.from_numpy(data).to(self.device).unsqueeze(0)
            # Project into a spherical image
            vertex_map = self.projector.build_projection_map(pc_data)
        elif isinstance(data, torch.Tensor):
            if len(data.shape) == 3 or len(data.shape) == 4:
                # Cast the data tensor as a vertex map
                vertex_map = data.to(self.device)
                if len(data.shape) == 3:
                    vertex_map = vertex_map.unsqueeze(0)
                else:
                    assert_debug(data.shape[0] == 1, f"Unexpected batched data format.")
                check_tensor(vertex_map, [1, 3, -1, -1])
                pc_data = vertex_map.permute(0, 2, 3, 1).reshape(1, -1, 3)
                pc_data = pc_data[mask_not_null(pc_data, dim=-1)[:, :, 0]]

            else:
                assert_debug(len(data.shape) == 2)
                pc_data = data.to(self.device).unsqueeze(0)
                vertex_map = self.projector.build_projection_map(pc_data)
        else:
            raise RuntimeError(f"Could not interpret the data: {data} as a pointcloud tensor")

        self._tgt_vmap = vertex_map.to(torch.float32)  # [1, 3, -1, -1]
        self._tgt_pc = pc_data.to(torch.float32)

        self._tgt_vmap = modify_nan_pmap(self._tgt_vmap, 0.0)
        _tgt_pc, _ = remove_nan(self._tgt_pc[0])
        self._tgt_pc = _tgt_pc.unsqueeze(0)

    def __update_map(self, new_rpose: torch.Tensor, data_dict: dict):
        # Updates the map if the motion since last registration is large enough
        new_delta = self._delta_since_map_update @ new_rpose
        delta_params = self.pose.from_pose_matrix(new_delta)

        if delta_params[0, :3].norm() > self._register_threshold_trans or \
                delta_params[0, 3:].norm() * 180 / np.pi > self._register_threshold_rot:

            new_mask = mask_not_null(self._tgt_vmap)
            new_nmap = None
            if "normal_map" in data_dict:
                new_nmap = data_dict["normal_map"]
            self.local_map.update(new_rpose,
                                  new_vertex_map=self._tgt_vmap,
                                  new_pc_data=self._tgt_pc,
                                  normal_map=new_nmap,
                                  mask=new_mask)
            self._delta_since_map_update = torch.eye(4, dtype=torch.float32, device=self.device)
        else:
            self.local_map.update(new_rpose)
            self._delta_since_map_update = new_delta
    # ------------------------------------------------------------------------------------------------------------------
