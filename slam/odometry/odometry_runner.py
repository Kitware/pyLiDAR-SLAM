from pathlib import Path
from typing import Optional
import time

import os
import torch

from abc import ABC
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Hydra and OmegaConf imports
from hydra.core.config_store import ConfigStore

# Project Imports
from slam.common.pose import Pose
from slam.common.torch_utils import collate_fun
from slam.common.utils import check_sizes, assert_debug
from slam.dataset import DatasetLoader, DATASET
from slam.eval.eval_odometry import OdometryResults
from slam.odometry import ODOMETRY
from slam.odometry.odometry import OdometryAlgorithm
from slam.dataset.configuration import DatasetConfig
from hydra.conf import dataclass, MISSING, field

from slam.odometry import OdometryConfig


@dataclass
class SLAMConfig:
    """The configuration dataclass"""

    # --------------------------------
    # OdometryConfig
    odometry: OdometryConfig = MISSING
    dataset: DatasetConfig = MISSING

    # ------------------
    # Default parameters
    log_dir: str = field(default_factory=os.getcwd)
    num_workers: int = 2
    pin_memory: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pose: str = "euler"

    # ----------------
    # Debug parameters
    viz_num_pointclouds: int = 200
    debug: bool = True


# -------------
# HYDRA Feature
# Automatically casts the config as a SLAMConfig object, and raises errors if it cannot do so
cs = ConfigStore.instance()
cs.store(name="slam_config", node=SLAMConfig)


class SLAMRunner(ABC):
    """
    A SLAMRunner runs a LiDAR SLAM algorithm on a set of pytorch datasets,
    And if the ground truth is present, it evaluates the performance of the algorithm and saved the results to disk
    """

    def __init__(self, config: SLAMConfig):
        super().__init__()

        self.config: SLAMConfig = config

        # Pytorch parameters extracted
        self.num_workers = self.config.num_workers
        self.batch_size = 1
        self.pin_memory = self.config.pin_memory
        self.log_dir = self.config.log_dir
        self.device = torch.device(self.config.device)

        self.pose = Pose(self.config.pose)
        self.viz_num_pointclouds = self.config.viz_num_pointclouds

        # Dataset config
        dataset_config: DatasetConfig = self.config.dataset
        self.dataset_loader: DatasetLoader = DATASET.load(dataset_config)

        # Odometry algorithm config
        self.slam_config: OdometryConfig = self.config.odometry

    def run_odometry(self):
        """Runs the LiDAR Odometry algorithm on the different datasets"""
        # Load the Datasets
        datasets: list = self.load_datasets()
        # Load the Slam algorithm
        slam = self.load_slam_algorithm()

        for sequence_name, dataset in datasets:
            # Build dataloader
            dataloader = DataLoader(dataset,
                                    collate_fn=collate_fun,
                                    pin_memory=self.pin_memory,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers)

            # Init the SLAM
            slam.init()

            elapsed = 0.0
            relative_ground_truth = self.ground_truth(sequence_name)
            for b_idx, data_dict in self._progress_bar(dataloader, desc=f"Sequence {sequence_name}"):
                data_dict = self._send_to_device(data_dict)
                try:
                    start = time.time()

                    # Process next frame
                    slam.process_next_frame(data_dict)

                    # Measure the time spent on the processing of the next frame
                    elapsed_sec = time.time() - start
                    elapsed += elapsed_sec

                except RuntimeError as e:
                    relative_poses = slam.get_relative_poses()
                    if relative_poses is not None and len(relative_poses) > 0:
                        self.save_and_evaluate(sequence_name, relative_poses, None)
                    print("[ERRROR] running SLAM : the estimated trajectory was dumped")
                    raise e

            # Evaluate the SLAM if it has a ground truth
            relative_poses = slam.get_relative_poses()
            check_sizes(relative_poses, [-1, 4, 4])
            if relative_ground_truth is not None:
                check_sizes(relative_ground_truth, [relative_poses.shape[0], 4, 4])

            self.save_and_evaluate(sequence_name, relative_poses, relative_ground_truth, elapsed=elapsed)

    def save_and_evaluate(self, sequence_name: str,
                          trajectory: np.ndarray,
                          ground_truth: Optional[np.ndarray],
                          elapsed: Optional[float] = None):
        """Saves metrics and trajectory in a folder on disk"""

        odo_results = OdometryResults(str(Path(self.log_dir) / sequence_name))
        odo_results.add_sequence(sequence_name,
                                 trajectory,
                                 ground_truth,
                                 elapsed)
        odo_results.close()

    @staticmethod
    def _progress_bar(dataloader: DataLoader, desc: str = ""):
        return tqdm(enumerate(dataloader, 0),
                    desc=desc,
                    total=len(dataloader),
                    ncols=120, ascii=True)

    def _send_to_device(self, data_dict: dict):
        output_dict: dict = {}
        for key, item in data_dict.items():
            if isinstance(item, torch.Tensor):
                output_dict[key] = item.to(device=self.device)
            else:
                output_dict[key] = item
        return output_dict

    def load_datasets(self) -> list:
        """
        Loads the Datasets for which the odometry is evaluated

        Returns
        -------
        A list of pairs (sequence_name :str, dataset_config :Dataset)
        Where :
            sequence_name is the name of a sequence which will be constructed
        """
        train_dataset, _, _, _ = self.dataset_loader.sequences()
        assert_debug(train_dataset is not None)
        pairs = [(train_dataset[1][idx], train_dataset[0][idx]) for idx in range(len(train_dataset[0]))]
        return pairs

    def load_slam_algorithm(self) -> OdometryAlgorithm:
        """
        Returns the SLAM algorithm which will be run
        """
        return ODOMETRY.load(self.config.odometry,
                             projector=self.dataset_loader.projector(),
                             pose=self.pose,
                             device=self.device,
                             viz_num_pointclouds=self.viz_num_pointclouds)

    def ground_truth(self, sequence_name: str) -> Optional[np.ndarray]:
        """
        Returns the ground truth associated with the sequence
        """
        return self.dataset_loader.get_ground_truth(sequence_name)
