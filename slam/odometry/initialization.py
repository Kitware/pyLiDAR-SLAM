from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Any

# Hydra and omegaconf
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, MISSING
from hydra.conf import field, dataclass

# Project imports
from slam.common.pose import Pose
from slam.common.timer import *
from slam.odometry import *
from slam.common.utils import assert_debug
from slam.training.prediction_modules import _PoseNetPredictionModule
from slam.viz import _with_cv2

if _with_cv2:
    import cv2

    from slam.common.registration import ElevationImageRegistration


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class InitializationConfig:
    """The Initialization Config for registration based SLAM"""
    type: str = MISSING


# ----------------------------------------------------------------------------------------------------------------------
class Initialization(ABC):
    """The initialization model which gives a first estimate for the next frame"""

    def __init__(self, config: InitializationConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def init(self):
        """Initializes the Algorithm ()"""
        raise NotImplementedError("")

    @abstractmethod
    def next_initial_pose(self, data_dict: Optional[dict] = None):
        """Initializes the Algorithm ()"""
        raise NotImplementedError("")

    @abstractmethod
    def register_motion(self, new_pose: torch.Tensor, data_dict: dict):
        """Registers the new motion into the algorithm"""
        raise NotImplementedError("")


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class NIConfig(InitializationConfig):
    """The configuration without initialization"""
    type: str = "ni"


# ----------------------------------------------------------------------------------------------------------------------
class NoInitialization(Initialization):
    """Initialize motion with identity"""

    def __init__(self, config: InitializationConfig, pose: Pose, device: torch.device = torch.device("cpu"), **kwargs):
        super().__init__(config)
        self.device = device
        self.next_estimate = None

    def init(self):
        """Sets the predicted motion as the identity systematically"""
        torch.eye(4, device=self.device).reshape(1, 4, 4)

    def next_initial_pose(self, data_dict: Optional[dict] = None):
        """Returns the identity"""
        return self.next_estimate

    def register_motion(self, relative_pose: torch.Tensor, data_dict: dict):
        """No actions required"""
        pass


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class CVConfig(InitializationConfig):
    """The configuration without initialization"""
    type: str = "cv"


# ----------------------------------------------------------------------------------------------------------------------
class ConstantVelocityInitialization(Initialization):
    """A Constant Velocity model for initialization (returns the last registered relative pose at each time step)"""

    def __init__(self, config: CVConfig, pose: Pose, device: torch.device = torch.device("cpu"), **kwargs):
        super().__init__(config)
        self.pose = pose
        self.device = device
        self.initial_estimate = None

    def init(self):
        self.initial_estimate = torch.eye(4, dtype=torch.float32, device=self.device).reshape(1, 4, 4)

    def next_initial_pose(self, data_dict: Optional[dict] = None):
        return self.initial_estimate

    def register_motion(self, relative_pose: torch.Tensor, data_dict: dict):
        self.initial_estimate = relative_pose


# ----------------------------------------------------------------------------------------------------------------------
if _with_cv2:

    @dataclass
    class EIConfig(InitializationConfig):
        """Config for Elevation Image feature based 2D alignment"""
        type: str = "ei"
        debug: bool = False
        ni_if_failure: bool = False
        registration_config: DictConfig = field(default_factory=lambda: DictConfig({}))


    class ElevationImageInitialization(Initialization):
        """Initialize motion by resolving a planar motion registration"""

        def __init__(self, ei_config: EIConfig, pose: Pose, device: torch.device = torch.device("cpu")):
            super().__init__(ei_config)
            self.pose = pose
            self.device = device

            self.next_estimate = None

            # Local variables
            self._previous_kpts = None
            self._previous_desc = None
            self._previous_image = None

            self.algorithm = ElevationImageRegistration(DictConfig(ei_config.registration_config))
            self.debug = ei_config.debug
            self.ni_if_failure = ei_config.ni_if_failure

            if self.debug:
                self._previous_pc = None
                cv2.namedWindow("matches", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        def __del__(self):
            if hasattr(self, "debug") and self.debug:
                cv2.destroyWindow("matches")

        def init(self):
            self.next_estimate = torch.eye(4, dtype=torch.float32, device=self.device).reshape(1, 4, 4)
            # Local variables
            self._previous_kpts = None
            self._previous_desc = None
            self._previous_image = None

        def next_initial_pose(self, data_dict: Optional[dict] = None):
            assert_debug(data_dict is not None and "numpy_pc_0" in data_dict)
            next_estimate = self.next_estimate

            # Convert new vmap to numpy
            pc_numpy = data_dict["numpy_pc_0"]

            # Build elevation image
            image, kpts, desc = self.algorithm.compute_features(pc_numpy)

            # Extract KeyPoints and descriptors
            result = None
            if self._previous_image is not None and len(kpts) > 50:
                result, inliers, inliers_matches = self.algorithm.align_2d(self._previous_kpts, self._previous_desc,
                                                                           kpts, desc,
                                                                           self._previous_image, image)
                if result is not None:
                    np_transform = result
                    next_estimate = torch.from_numpy(np_transform).to(self.device).reshape(1, 4, 4)

                if self.debug:
                    matches_image = cv2.drawMatches(self._previous_image, self._previous_kpts,
                                                    image, kpts, inliers_matches, None)
                    cv2.imshow("matches", matches_image)
                    cv2.waitKey(5)

            if desc is not None:
                self._previous_kpts = kpts
                self._previous_desc = desc
                self._previous_image = image

            return next_estimate

        def register_motion(self, relative_pose: torch.Tensor, data_dict: dict):
            if not self.ni_if_failure:
                self.next_estimate = relative_pose


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class PNConfig(InitializationConfig):
    """
    The Initialization config for PoseNet

    TODO : Refactor to avoid repetition with PoseNet for Odometry
    """
    type: str = "posenet"

    train_dir: str = MISSING
    checkpoint_file: str = "checkpoint.ckp"  # Default checkpoint file generated by trainer
    train_config_file: str = "config.yaml"  # Default config file generated by trainer

    prediction: Dict[str, Any] = MISSING


class PoseNetInitialization(Initialization):
    """Initialization using a PoseNet for LiDAR odometry"""

    def __init__(self, config: PNConfig, pose: Pose, device: torch.device = torch.device("cpu")):
        super().__init__(config)
        self.device = device
        self.pose = pose

        # Loads the train config from the disk
        # TODO refactor
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

        checkpoint_file = config.checkpoint_file
        self.train_dir = Path(config.train_dir)
        self.checkpoint_file = self.train_dir / checkpoint_file
        assert_debug(self.checkpoint_file.exists())

        # ----------------------
        # Local variable
        self.previous_vertex_map = None
        self._iter = 0
        self.relative_poses = []

    def init(self):
        self.relative_poses = []
        self._iter = 0

        # Load PoseNet params from disk
        state_dict = torch.load(str(self.checkpoint_file))
        self.prediction_module.load_state_dict(state_dict["prediction_module"])

    def next_initial_pose(self, data_dict: Optional[dict] = None):
        vertex_map = data_dict["vertex_map"]
        if self.previous_vertex_map is None:
            estimate = torch.eye(4, dtype=torch.float32, device=self.device).reshape(1, 4, 4)
        else:
            with torch.no_grad():
                input_ = torch.cat([self.previous_vertex_map, vertex_map], dim=0).unsqueeze(0)
                estimate_params = self.prediction_module(dict(vertex_map=input_))["pose_params"]
                estimate = self.pose.build_pose_matrix(estimate_params)

        self.previous_vertex_map = vertex_map
        return estimate

    def register_motion(self, new_pose: torch.Tensor, data_dict: dict):
        pass


# ----------------------------------------------------------------------------------------------------------------------

# Hydra Config Store : for the group odometry/initialization
cs = ConfigStore.instance()
cs.store(group="slam/odometry/initialization", name="CV", node=CVConfig)
cs.store(group="slam/odometry/initialization", name="PoseNet", node=PNConfig)
cs.store(group="slam/odometry/initialization", name="NI", node=NIConfig())

if _with_cv2:
    cs.store(group="odometry/initialization", name="EI", node=EIConfig)


# ----------------------------------------------------------------------------------------------------------------------
class INITIALIZATION(Enum):
    """A Convenient enum to load the Algorithm from a config dictionary"""
    ni = (NoInitialization, NIConfig)
    cv = (ConstantVelocityInitialization, CVConfig)
    posenet = (PoseNetInitialization, PNConfig)

    if _with_cv2:
        ei = (ElevationImageInitialization, EIConfig)

    @staticmethod
    def load(config: InitializationConfig, **kwargs) -> Initialization:
        _type = config.type
        assert_debug(_type in INITIALIZATION.__members__)
        _algo, _config = INITIALIZATION.__members__[_type].value

        return _algo(_config(**config), **kwargs)
