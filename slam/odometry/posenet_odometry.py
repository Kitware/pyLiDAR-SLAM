from pathlib import Path
from typing import Dict, Union, Any

import numpy as np

# Hydra and OmegaConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig
from hydra.conf import dataclass

# Project Imports
from slam.common.pose import Pose
from slam.common.timer import *
from slam.odometry import *
from slam.odometry.odometry import OdometryAlgorithm, OdometryConfig
from slam.training.prediction_modules import _PoseNetPredictionModule


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class PoseNetOdometryConfig(OdometryConfig):
    """
    The Configuration for the Point-To-Plane ICP based Iterative Least Square estimation of the pose
    """
    debug: bool = False
    viz_mode: str = "aggregated"
    algorithm: str = "posenet"

    train_dir: str = MISSING  # The directory where the posenet_config and checkpoint file should be searched in
    train_config_file: str = "config.yaml"  # Default value set by ATrainer
    checkpoint_file: str = "checkpoint.ckp"  # Default value set by ATrainer

    device: str = MISSING
    pose: str = MISSING
    posenet_config: Dict[str, Any] = MISSING


# Hydra -- Add a PoseNetOdometryCfonig
cs = ConfigStore.instance()
cs.store(name="poseresnet18", node=PoseNetOdometryConfig(posenet_config={"type": "poseresnet",
                                                                         "model": 18}),
         package="odometry.posenet_config")


# ----------------------------------------------------------------------------------------------------------------------
class PoseNetOdometry(OdometryAlgorithm):
    """Deep Odometry"""

    def __init__(self, config: Union[PoseNetOdometryConfig, DictConfig],
                 pose: Pose = Pose("euler"),
                 device: torch.device = torch.device("cpu"),
                 **kwargs):
        OdometryAlgorithm.__init__(self, config)

        # Set variables needed by the module
        self.device = device
        self.pose = pose

        # Loads the train config from the disk
        train_dir = Path(config.train_dir)
        assert_debug(train_dir.exists())
        train_config_path = train_dir / config.train_config_file
        checkpoint_path = train_dir / config.checkpoint_file
        assert_debug(train_config_path.exists() and checkpoint_path.exists())
        self.checkpoint_path = str(checkpoint_path)

        # Reads the prediction config from the dict
        with open(str(train_config_path), "r") as stream:
            train_config = OmegaConf.load(stream)
        prediction_config: DictConfig = train_config["training"]["prediction"]

        # Construct the Prediction module from the config read from disk
        self.prediction_module = _PoseNetPredictionModule(prediction_config,
                                                          pose=self.pose)
        self.prediction_module = self.prediction_module.to(self.device)

        # ----------------------
        # Local variable
        self.previous_vertex_map = None
        self._iter = 0
        self.relative_poses = []

    def init(self):
        """
        Initializes the Odometry algorithm

        Clears the persisted relative poses, reset the _iter to 0
        And loads the module parameters from disk
        """
        super().init()
        self.relative_poses = []
        self._iter = 0

        # Load the parameters of the model from the config
        state_dict = torch.load(self.checkpoint_path)
        self.prediction_module.load_state_dict(state_dict["prediction_module"])

    def do_process_next_frame(self, data_dict: dict):
        """
        Registers the new frame
        """
        vertex_map = data_dict["vertex_map"]
        if self._iter == 0:
            self.previous_vertex_map = vertex_map.unsqueeze(0)
            self._iter += 1
            self.relative_poses.append(np.eye(4, dtype=np.float32).reshape(1, 4, 4))
            return

        pair_vmap = torch.cat([self.previous_vertex_map, vertex_map.unsqueeze(0)], dim=1)

        with torch.no_grad():
            output_dict = self.prediction_module(dict(vertex_map=pair_vmap))
        pose_params = output_dict["pose_params"]
        new_rpose = self.pose.build_pose_matrix(pose_params)

        # Update the state of the odometry
        self.previous_vertex_map = vertex_map.unsqueeze(0)
        self.relative_poses.append(new_rpose.cpu().numpy())
        self._iter += 1

    def get_relative_poses(self) -> np.ndarray:
        return np.concatenate(self.relative_poses, axis=0)
