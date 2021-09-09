from hydra.core.config_store import ConfigStore

from slam.common.modules import _with_ct_icp
from slam.common.pose import transform_pointcloud
from slam.common.utils import remove_nan
from slam.eval.eval_odometry import compute_relative_poses

if _with_ct_icp:
    import pyct_icp as pct
    # Project Imports
    from typing import Optional
    from slam.odometry.odometry import *
    from hydra.conf import field
    from slam.viz.color_map import *
    from slam.common.modules import _with_viz3d

    if _with_viz3d:
        from viz3d.window import OpenGLWindow, field


    def add_pct_annotations(pct_cls):
        """A Decorator which defines the fields of a Dataclass from the attributes of a pyct_icp object

        Note: It should be added before the decorator @dataclass. See CTICPOptionsWrapper for example usage
        """

        default_pct_options = pct_cls()

        def wrap(cls):
            if not hasattr(cls, "__annotations__"):
                setattr(cls, "__annotations__", {})
            # Add fields from pct.OdometryOptions as annotations
            for key, value in pct_cls.__dict__.items():
                if isinstance(value, property):
                    default_value = getattr(default_pct_options, key)
                    key_type = type(default_value)

                    if key_type in [str, int, float, bool]:
                        cls.__annotations__[key] = key_type
                        setattr(cls, key, default_value)
                    elif key_type in [pct.ICP_DISTANCE, pct.LEAST_SQUARES, pct.CT_ICP_DATASET, pct.MOTION_COMPENSATION,
                                      pct.CT_ICP_SOLVER, pct.INITIALIZATION]:
                        # Replace pyct_icp enums by string
                        cls.__annotations__[key] = str
                        value_name = default_value.name
                        setattr(cls, key, value_name)
                    elif key_type == pct.CTICPOptions:
                        cls.__annotations__[key] = CTICPOptionsWrapper
                        setattr(cls, key, CTICPOptionsWrapper.build_from_pct(default_value))
                        print(f"Type {key_type} is not recognised")

            return cls

        return wrap


    @dataclass
    @add_pct_annotations(pct.CTICPOptions)
    class CTICPOptionsWrapper:

        @staticmethod
        def _enums():
            # Return the enums used in the options of pyct_icp
            # They need to be processed differently to insure compatibility with hydra
            return ["distance", "loss_function", "solver"]

        @staticmethod
        def build_from_pct(pct_options: pct.CTICPOptions):
            from dataclasses import _FIELDS
            assert_debug(isinstance(pct_options, pct.CTICPOptions))

            wrapped = CTICPOptionsWrapper()
            for _field in getattr(wrapped, _FIELDS):
                if _field in CTICPOptionsWrapper._enums():
                    value_ = getattr(pct_options, _field)
                    setattr(wrapped, _field, value_.name)
                else:
                    setattr(wrapped, _field, getattr(pct_options, _field))

            return wrapped

        def to_pct_object(self):
            options = pct.CTICPOptions()

            for field_name in self.__dict__:
                if field_name in CTICPOptionsWrapper._enums():
                    field_value = getattr(self, field_name)
                    if field_name == "distance":
                        field_value = getattr(pct.ICP_DISTANCE, field_value)
                    elif field_name == "loss_function":
                        field_value = getattr(pct.LEAST_SQUARES, field_value)
                    elif field_name == "solver":
                        field_value = getattr(pct.CT_ICP_SOLVER, field_value)
                    else:
                        raise NotImplementedError(f"The field name {field_value} is not recognised")
                    setattr(options, field_name, field_value)
                else:
                    field_value = getattr(self, field_name)
                    assert_debug(field_value != MISSING)
                    setattr(options, field_name, field_value)

            return options


    @dataclass
    @add_pct_annotations(pct.OdometryOptions)  # /!\ Generate the properties from the attributes of pcd.OdometryOptions
    class OdometryOptionsWrapper:

        @staticmethod
        def _enums():
            # Return the enums used in the options of pyct_icp
            # They need to be processed differently to insure compatibility with hydra
            return ["motion_compensation", "initialization"]

        @staticmethod
        def return_fieldnames():
            from dataclasses import _FIELDS
            return getattr(OdometryOptionsWrapper, _FIELDS, [])

        def to_pct_object(self):
            options = pct.OdometryOptions()

            for field_name in self.__dict__:
                if field_name == "ct_icp_options":
                    field_value = CTICPOptionsWrapper(**getattr(self, field_name))
                    setattr(options, field_name, field_value.to_pct_object())
                elif field_name == "motion_compensation":
                    field_value = getattr(self, field_name)
                    field_value = getattr(pct.MOTION_COMPENSATION, field_value)
                    setattr(options, field_name, field_value)
                elif field_name == "initialization":
                    field_value = getattr(self, field_name)
                    field_value = getattr(pct.INITIALIZATION, field_value)
                    setattr(options, field_name, field_value)
                else:
                    field_value = getattr(self, field_name)
                    assert_debug(field_value != MISSING)
                    setattr(options, field_name, field_value)

            return options

        @staticmethod
        def build_from_pct(pct_options: pct.OdometryOptions):
            from dataclasses import _FIELDS
            assert_debug(isinstance(pct_options, pct.OdometryOptions))

            wrapped = OdometryOptionsWrapper()
            for _field in getattr(wrapped, _FIELDS):
                if _field == "ct_icp_options":
                    value_ = getattr(pct_options, _field)
                    assert_debug(isinstance(value_, pct.CTICPOptions))
                    value_ = CTICPOptionsWrapper.build_from_pct(value_)
                    setattr(wrapped, _field, value_)
                elif _field in OdometryOptionsWrapper._enums():
                    value_ = getattr(pct_options, _field)
                    setattr(wrapped, _field, value_.name)
                else:
                    setattr(wrapped, _field, getattr(pct_options, _field))

            return wrapped


    @dataclass
    class CT_ICPOdometryConfig(OdometryConfig):
        """
        The Configuration for the Point-To-Plane ICP based Iterative Least Square estimation of the pose
        """
        algorithm: str = "ct_icp"
        debug_viz: bool = False

        numpy_pc_key: str = "numpy_pc"
        timestamps_key: str = "numpy_pc_timestamps"

        pose_type: str = "mid_pose"  # The relative pose to return pose (in mid_pose, begin_pose, end_pose)

        options: OdometryOptionsWrapper = field(default_factory=lambda: OdometryOptionsWrapper())


    def default_drive_config() -> CT_ICPOdometryConfig:
        default_config = CT_ICPOdometryConfig()
        default_pct_options = pct.DefaultDrivingProfile()
        default_config.options = OdometryOptionsWrapper.build_from_pct(default_pct_options)
        return default_config


    def robust_drive_config() -> CT_ICPOdometryConfig:
        default_config = CT_ICPOdometryConfig()
        default_pct_options = pct.RobustDrivingProfile()
        default_config.options = OdometryOptionsWrapper.build_from_pct(default_pct_options)
        return default_config


    def default_small_motion_config() -> CT_ICPOdometryConfig:
        default_config = CT_ICPOdometryConfig()
        default_pct_options = pct.DefaultRobustOutdoorLowInertia()
        default_config.options = OdometryOptionsWrapper.build_from_pct(default_pct_options)
        return default_config


    # Store ct_icp in the group slam/odometry
    cs = ConfigStore.instance()
    cs.store(name="ct_icp", group="slam/odometry", node=CT_ICPOdometryConfig())
    cs.store(name="ct_icp_drive", group="slam/odometry", node=default_drive_config())
    cs.store(name="ct_icp_robust_drive", group="slam/odometry", node=robust_drive_config())
    cs.store(name="ct_icp_slow_outdoor", group="slam/odometry", node=default_small_motion_config())


    class CT_ICPOdometry(OdometryAlgorithm):
        """
        An Odometry Algorithm which updates the poses by taking into account the distortion of the frame

        The algorithm uses the wrapping python of CT_ICP (for Continuous-Time ICP)
        (see https://github.com/jedeschaud/ct_icp for more details)
        """

        @staticmethod
        def lidar_frame_key():
            return "ct_icp_lidar_frame"

        def get_relative_poses(self) -> np.ndarray:
            return compute_relative_poses(np.array(self.absolute_poses))

        def __init__(self, config: CT_ICPOdometryConfig, **kwargs):
            OdometryAlgorithm.__init__(self, config)

            self.options: Optional[pct.OdometryOptions] = None
            self.ct_icp_odometry: Optional[pct.Odometry] = None

            self._has_window = config.debug_viz and _with_viz3d
            self.viz3d_window = None
            self._frame_index = 0
            self.absolute_poses = []
            self.gt_poses = []

        def __del__(self):
            if self._has_window:
                if self.viz3d_window is not None:
                    self.viz3d_window.close(True)

        def init(self):
            """Initialize/ReInitialize the state of the Algorithm and its components"""
            super().init()
            import logging
            logging.basicConfig(level=logging.WARNING)

            self.options = OdometryOptionsWrapper(**self.config.options).to_pct_object()
            self.ct_icp_odometry = pct.Odometry(self.options)

            self._frame_index = 0
            self.absolute_poses.clear()
            self.gt_poses.clear()

            if self._has_window:
                if self.viz3d_window is not None:
                    self.viz3d_window.close(True)
                    self.viz3d_window = None
                self.viz3d_window = OpenGLWindow(
                    engine_config={"with_edl": True, "edl_strength": 10000.0})
                self.viz3d_window.init()

        # ------------------------------------------------------------------------------------------------------------------
        def do_process_next_frame(self, data_dict: dict):
            """
            Registers a new frame to the Map

            Note: `CT_ICP` requires frames with timestamps.
                  If no timestamps are found in the frame dict, the option `slam.odometry.options.ct_icp_options`
                  Must be set to the string POINT_TO_PLANE
            """
            assert isinstance(self.config, CT_ICPOdometryConfig)

            # Search for or build the lidar frame
            lidar_frame = None
            if self.lidar_frame_key() in data_dict:
                lidar_frame = data_dict[self.lidar_frame_key()]
                assert_debug(isinstance(lidar_frame, pct.LiDARFrame))
            else:
                lidar_frame = pct.LiDARFrame()
                assert_debug(self.config.numpy_pc_key in data_dict)
                numpy_pc = data_dict[self.config.numpy_pc_key].astype(np.float64)

                new_points, __filter = remove_nan(numpy_pc)

                if self.options.ct_icp_options.distance == pct.CT_POINT_TO_PLANE:
                    # CT_ICP requires timestamps
                    assert_debug(self.config.timestamps_key in data_dict,
                                 f"[CT_ICP] The timestamps dict {self.config.timestamps_key} is not in the dict containing keys={data_dict.keys()}.\n"
                                 f"Set the parameter slam.odometry.odometry_options.ct_icp_options.distance=POINT_TO_PLANE to run the standard Point-to-plane algorithm")
                    timestamps = data_dict[self.config.timestamps_key].astype(np.float64)[__filter]
                else:
                    timestamps = np.ones((new_points.shape[0],), dtype=np.float64)

                min_timestamp = timestamps.min()
                max_timestamp = timestamps.max()
                if min_timestamp != max_timestamp:
                    alpha_timestamp = (timestamps - min_timestamp) / (max_timestamp - min_timestamp)
                else:
                    alpha_timestamp = timestamps

                frame_index = np.ones((new_points.shape[0],), dtype=np.int32) * self._frame_index

                frame_points = np.rec.fromarrays([new_points, new_points,
                                                  alpha_timestamp, timestamps, frame_index],
                                                 dtype=[("raw_point", "3f8"),
                                                        ("point", "3f8"),
                                                        ("alpha_timestamp", "1f8"),
                                                        ("timestamp", "1f8"),
                                                        ("frame_index", "1i4")])
                lidar_frame.SetFrame(frame_points)

            # Add the frame to the odometry
            result: pct.RegistrationSummary = self.ct_icp_odometry.RegisterFrame(lidar_frame)
            assert_debug(result.success, f"[CT_ICP]The registration of frame {self._frame_index} has failed")

            new_pose = np.eye(4, dtype=np.float64)
            if self.config.pose_type == "mid_pose":
                new_pose = result.frame.MidPose()  # [4, 4] the mid pose of the new frame
            elif self.config.pose_type == "end_pose":
                new_pose[:3, :3] = result.frame.end_R
                new_pose[:3, 3] = result.frame.end_t
            elif self.config.pose_type == "begin_pose":
                new_pose[:3, :3] = result.frame.begin_R
                new_pose[:3, 3] = result.frame.begin_t
            else:
                raise ValueError(f"Unrecognised `slam.odometry.pose_type` option {self.config.pose_type}")

            # Compute the new relative pose
            if len(self.absolute_poses) == 0:
                relative_pose = new_pose
            else:
                relative_pose = np.linalg.inv(self.absolute_poses[-1]).dot(new_pose)

            self.absolute_poses.append(new_pose)
            data_dict[self.relative_pose_key()] = relative_pose
            world_points = result.points.GetStructuredArrayRef()["pt"]
            corrected_frame_points = transform_pointcloud(world_points, np.linalg.inv(new_pose))
            data_dict[self.pointcloud_key()] = corrected_frame_points

            if "absolute_pose_gt" in data_dict:
                gt_pose = data_dict["absolute_pose_gt"]
                if isinstance(gt_pose, torch.Tensor):
                    gt_pose = gt_pose.cpu().numpy().reshape(4, 4)
                self.gt_poses.append(gt_pose)

            if self._has_window:
                wpoints = world_points.astype(np.float32)
                self.viz3d_window.set_pointcloud(self._frame_index % 100, wpoints)
                self.viz3d_window.update_camera(new_pose.astype(np.float32))
                if self._frame_index % 1 == 0:
                    if len(self.gt_poses) > 0:
                        self.viz3d_window.set_poses(-1, np.array(self.gt_poses).astype(np.float32))

            self._frame_index += 1
