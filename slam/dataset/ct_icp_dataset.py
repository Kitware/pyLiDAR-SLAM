from typing import Optional, Union

from slam.common.modules import _with_ct_icp

if _with_ct_icp:
    import pyct_icp as pct

    import logging
    from pathlib import Path

    import numpy as np
    from torch.utils.data import Dataset, IterableDataset

    # Hydra and OmegaConf
    from hydra.conf import dataclass, field
    from hydra.core.config_store import ConfigStore
    from omegaconf import MISSING

    # Project Imports
    from slam.eval.eval_odometry import compute_relative_poses
    from slam.dataset import DatasetConfig, DatasetLoader
    from slam.common.projection import SphericalProjector
    from slam.common.utils import assert_debug
    from slam.odometry.ct_icp_odometry import add_pct_annotations, CT_ICPOdometry


    @dataclass
    @add_pct_annotations(pct.DatasetOptions)
    class CT_ICPDatasetOptionsWrapper:
        """A dataclass wrapper for a pct.DatasetOptions

        The fields of the dataclass are programmatically defined from the attributes of a pct.DatasetOptions
        """

        @staticmethod
        def build_from_pct(pct_options: pct.DatasetOptions):
            from dataclasses import _FIELDS
            assert_debug(isinstance(pct_options, pct.DatasetOptions))

            wrapped = CT_ICPDatasetOptionsWrapper()
            for _field in getattr(wrapped, _FIELDS):
                if _field == "dataset":
                    value_ = getattr(pct_options, _field)
                    setattr(wrapped, _field, value_.name)
                else:
                    setattr(wrapped, _field, getattr(pct_options, _field))

            return wrapped

        def to_pct_object(self):
            options = pct.DatasetOptions()
            for field_name in self.__dict__:
                if field_name == "dataset":
                    field_value = getattr(self, field_name)
                    field_value = getattr(pct.CT_ICP_DATASET, field_value)
                    setattr(options, field_name, field_value)
                else:
                    field_value = getattr(self, field_name)
                    assert_debug(field_value != MISSING)
                    setattr(options, field_name, field_value)

            return options


    @dataclass
    class CT_ICPDatasetConfig(DatasetConfig):
        """A configuration object read from a yaml conf"""
        # -------------------
        # Required Parameters
        dataset: str = "ct_icp"

        options: CT_ICPDatasetOptionsWrapper = field(default_factory=lambda: CT_ICPDatasetOptionsWrapper())

        # ------------------------------
        # Parameters with default values
        lidar_key: str = "vertex_map"
        lidar_height: int = 64
        lidar_width: int = 1024
        up_fov: int = 3
        down_fov: int = -24
        all_sequence: list = field(default_factory=lambda: [f"{i:02}" for i in range(11) if i != 3] +
                                                           [f"Town{1 + i:02}" for i in range(7)])
        train_sequences: list = field(default_factory=lambda: [f"{i:02}" for i in range(11) if i != 3] +
                                                              [f"Town{1 + i:02}" for i in range(7)])
        test_sequences: list = field(default_factory=lambda: [f"{i:02}" for i in range(22) if i != 3])
        eval_sequences: list = field(default_factory=lambda: ["09", "10"])


    # Hydra -- stores a CT_ICPDatasetConfig `ct_icp` in the `dataset` group
    cs = ConfigStore.instance()
    cs.store(group="dataset", name="ct_icp", node=CT_ICPDatasetConfig)


    class CT_ICPDatasetSequence:
        """
        Dataset for a Sequence defined in CT_ICP Datasets
        See https://github.com/jedeschaud/ct_icp for more details

        Attributes:
            options (CT_ICPDatasetOptionsWrapper): the ct_icp options to load the dataset
            sequence_id (str): id of the sequence
        """

        def __init__(self,
                     options: Union[pct.DatasetOptions, CT_ICPDatasetOptionsWrapper],
                     sequence_id: int,
                     gt_pose_channel: str = "absolute_pose_gt",
                     numpy_pc_channel: str = "numpy_pc"):
            assert isinstance(options, pct.DatasetOptions) or isinstance(options, CT_ICPDatasetOptionsWrapper)
            self.options: pct.DatasetOptions = options if isinstance(options,
                                                                     pct.DatasetOptions) else options.to_pct_object()
            # assert_debug(self.options.dataset != pct.NCLT, "The NCLT Dataset is not available in Random Access")
            self.dataset_sequences = pct.get_dataset_sequence(self.options, sequence_id)
            self.sequence_id = sequence_id
            self.gt = None
            self.is_initialized = False
            if pct.has_ground_truth(self.options, sequence_id):
                self.gt = np.array(pct.load_sensor_ground_truth(self.options, sequence_id), np.float64)
            self.numpy_pc_channel = numpy_pc_channel
            self.gt_pose_channel = gt_pose_channel

        def __reduce__(self):
            # Make the dataset pickable
            return CT_ICPDatasetSequence, (CT_ICPDatasetOptionsWrapper.build_from_pct(self.options), self.sequence_id,
                                           self.gt_pose_channel, self.numpy_pc_channel)

        def process_frame(self, lidar_frame: pct.LiDARFrame):
            data_dict = dict()

            # Add numpy pc values
            lidar_frame_ref = lidar_frame.GetStructuredArrayRef()
            numpy_pc = lidar_frame_ref["raw_point"].copy()
            timestamps = lidar_frame_ref["timestamp"].copy()

            data_dict[self.numpy_pc_channel] = numpy_pc.astype(np.float32)
            data_dict[f"{self.numpy_pc_channel}_timestamps"] = timestamps

            if self.gt is not None:
                data_dict[f"{self.gt_pose_channel}"] = self.gt[idx]

            return data_dict


    class IterableCT_ICPDataset(CT_ICPDatasetSequence, IterableDataset):
        def __init__(self,
                     options: Union[pct.DatasetOptions, CT_ICPDatasetOptionsWrapper],
                     sequence_id: int,
                     gt_pose_channel: str = "absolute_pose_gt",
                     numpy_pc_channel: str = "numpy_pc"):
            super().__init__(options, sequence_id, gt_pose_channel, numpy_pc_channel)

            assert isinstance(options, pct.DatasetOptions) or isinstance(options, CT_ICPDatasetOptionsWrapper)
            self._idx = 0

        def __len__(self):
            return 10000

        def __iter__(self):
            if self._idx > 0:
                assert_debug("Error, cannot iter twice over IterableDataset")
            return self

        def __next__(self):
            if not self.dataset_sequences.HasNext():
                raise StopIteration
            lidar_frame = self.dataset_sequences.Next()
            return super().process_frame(lidar_frame)


    class TorchCT_ICPDataset(CT_ICPDatasetSequence, Dataset):

        def __getitem__(self, idx):
            assert_debug(0 <= idx < len(self), "Index Error")
            lidar_frame = self.dataset_sequences.Frame(idx)
            return super().process_frame(lidar_frame)

        def __len__(self):
            return self.dataset_sequences.NumFrames()


    class CT_ICPDatasetLoader(DatasetLoader):
        """
        Configuration for a dataset proposed in CT_ICP
        """

        __KITTI_SEQUENCE = [f"{i:02}" for i in range(22) if i != 3]
        __KITTI_CARLA_SEQUENCE = [f"Town{1 + i:02}" for i in range(7)]

        @staticmethod
        def sequence_to_dataset(seq_name):
            if "_vel" in seq_name:
                return pct.NCLT
            if seq_name in CT_ICPDatasetLoader.__KITTI_CARLA_SEQUENCE:
                return pct.KITTI_CARLA
            if seq_name in CT_ICPDatasetLoader.__KITTI_SEQUENCE:
                return pct.KITTI_raw
            else:
                assert_debug(False, f"Sequence name {seq_name} does not match expected datasets")

        @staticmethod
        def is_iterable_dataset(dataset: pct.CT_ICP_DATASET):
            return dataset == pct.NCLT

        @staticmethod
        def have_sequence(seq_name):
            return "_vel" in seq_name or \
                   seq_name in CT_ICPDatasetLoader.__KITTI_SEQUENCE or \
                   seq_name in CT_ICPDatasetLoader.__KITTI_CARLA_SEQUENCE

        def __init__(self, config: CT_ICPDatasetConfig):
            super().__init__(config)
            self.options: CT_ICPDatasetOptionsWrapper = CT_ICPDatasetOptionsWrapper(**config.options).to_pct_object()

            root_path = Path(self.options.root_path)
            assert_debug(root_path.exists(), f"The root path of the dataset {str(root_path)} does not exist on disk")

            # Build the dictionary sequence_name -> sequence_id
            self.map_seqname_seqid = dict()
            all_sequences_id_size = pct.get_sequences(self.options)
            for seq_info in all_sequences_id_size:
                seq_id = seq_info.sequence_id
                seq_size = seq_info.sequence_size
                seq_name = seq_info.sequence_name
                assert_debug(seq_name in self.__KITTI_SEQUENCE or
                             seq_name in self.__KITTI_CARLA_SEQUENCE or
                             "_vel" in seq_name)
                self.map_seqname_seqid[seq_name] = seq_id

        def projector(self) -> SphericalProjector:
            """Default SphericalProjetor for KITTI (projection of a pointcloud into a Vertex Map)"""
            assert isinstance(self.config, CT_ICPDatasetConfig)
            lidar_height = self.config.lidar_height
            lidar_with = self.config.lidar_width
            up_fov = self.config.up_fov
            down_fov = self.config.down_fov
            # Vertex map projector
            projector = SphericalProjector(lidar_height, lidar_with, 3, up_fov, down_fov)
            return projector

        def get_ground_truth(self, sequence_name):
            """Returns the ground truth poses associated to a sequence of KITTI's odometry benchmark"""
            assert_debug(sequence_name in self.map_seqname_seqid)
            seq_id = self.map_seqname_seqid[sequence_name]

            if self.sequence_to_dataset(sequence_name) == pct.NCLT:
                return None

            ground_truth = pct.load_sensor_ground_truth(self.options, seq_id)
            absolute_poses = np.array(ground_truth).astype(np.float64)
            return compute_relative_poses(absolute_poses)

        def sequences(self):
            """
            Returns
            -------
            (train_dataset, eval_dataset, test_dataset, transform) : tuple
            train_dataset : (list, list)
                A list of dataset_config (one for each sequence of KITTI's Dataset),
                And the list of sequences used to build them
            eval_dataset : (list, list)
                idem
            test_dataset : (list, list)
                idem
            transform : callable
                A transform to be applied on the dataset_config
            """
            assert isinstance(self.config, CT_ICPDatasetConfig)

            # Sets the path of the kitti benchmark
            train_sequence_ids = self.config.train_sequences
            eval_sequence_ids = self.config.eval_sequences
            test_sequence_ids = self.config.test_sequences

            list_seq_info = pct.get_sequences(self.options)
            seqname_to_seqid = {seq_info.sequence_name: seq_info.sequence_id for seq_info in list_seq_info}

            _options = self.options

            def __get_datasets(sequences: list):
                if sequences is None or len(sequences) == 0:
                    return None

                datasets = []
                sequence_names = []
                for seq_name in sequences:
                    if not self.have_sequence(seq_name) or seq_name not in seqname_to_seqid:
                        logging.warning(
                            f"The dataset located at {_options.root_path} does not have the sequence named {seq_name}")
                        continue
                    seq_id = seqname_to_seqid[seq_name]
                    datasets.append(
                        TorchCT_ICPDataset(_options, seq_id) if not self.is_iterable_dataset(_options.dataset)
                        else IterableCT_ICPDataset(_options, seq_id))
                    sequence_names.append(seq_name)

                return datasets, sequence_names

            return __get_datasets(train_sequence_ids), \
                   __get_datasets(eval_sequence_ids), \
                   __get_datasets(test_sequence_ids), lambda x: x
