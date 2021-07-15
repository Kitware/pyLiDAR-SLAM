from typing import Optional, Dict

from torch import nn as nn
from torch.utils.data import Dataset

# Hydra and OmegaConf
import hydra
from hydra.conf import dataclass, MISSING
from omegaconf import DictConfig, OmegaConf

# Project Imports
from slam.common.pose import Pose
from slam.common.projection import SphericalProjector
from slam.common.utils import assert_debug
from slam.dataset import DATASET, DatasetLoader, DatasetConfig
from slam.training.loss_modules import _PointToPlaneLossModule, _PoseSupervisionLossModule, LossConfig
from slam.training.prediction_modules import _PoseNetPredictionModule, PredictionConfig
from slam.training.trainer import ATrainer, ATrainerConfig


# ----------------------------------------------------------------------------------------------------------------------
# Training Config

@dataclass
class TrainingConfig:
    """A Config for training of a PoseNet module"""
    loss: LossConfig = MISSING
    prediction: PredictionConfig = MISSING


@dataclass
class PoseNetTrainingConfig(ATrainerConfig):
    """A Config for a PoseNetTrainer"""
    pose: str = "euler"
    ei_config: Optional[Dict] = None
    sequence_len: int = 2
    num_input_channels: int = 3

    dataset: DatasetConfig = MISSING
    training: TrainingConfig = MISSING


# ----------------------------------------------------------------------------------------------------------------------
# Trainer for PoseNet
class PoseNetTrainer(ATrainer):
    """Unsupervised / Supervised Trainer for the PoseNet prediction module"""

    def __init__(self, config: PoseNetTrainingConfig):
        super().__init__(config)
        self.pose = Pose(self.config.pose)

        self.dataset_config: DatasetLoader = DATASET.load(config.dataset)
        self.projector: SphericalProjector = self.dataset_config.projector()

        # Share root parameters to Prediction Node
        self.config.training.prediction.sequence_len = self.config.sequence_len
        self.config.training.prediction.num_input_channels = self.config.num_input_channels

    def __transform(self, data_dict: dict):
        return data_dict

    def prediction_module(self) -> nn.Module:
        """Returns the PoseNet Prediction Module"""
        return _PoseNetPredictionModule(OmegaConf.create(self.config.training.prediction), self.pose)

    def loss_module(self) -> nn.Module:
        """Return the loss module used to train the model"""
        loss_config = self.config.training.loss
        mode = loss_config.mode
        assert_debug(mode in ["unsupervised", "supervised"])
        if mode == "supervised":
            return _PoseSupervisionLossModule(loss_config, self.pose)
        else:
            return _PointToPlaneLossModule(loss_config, self.projector, self.pose)

    def load_datasets(self) -> (Optional[Dataset], Optional[Dataset], Optional[Dataset]):
        """Loads the Datasets"""
        train_dataset, eval_dataset, test_dataset = self.dataset_config.get_sequence_dataset()
        train_dataset.sequence_transforms = self.__transform
        if test_dataset is not None:
            test_dataset.sequence_transforms = self.__transform
        if eval_dataset is not None:
            eval_dataset.sequence_transforms = self.__transform
        return train_dataset, eval_dataset, test_dataset

    def test(self):
        pass


@hydra.main(config_name="train_posenet", config_path="config")
def run(cfg: PoseNetTrainingConfig):
    trainer = PoseNetTrainer(PoseNetTrainingConfig(**cfg))
    # Initialize the trainer (Optimizer, Cuda context, etc...)
    trainer.init()

    if trainer.config.do_train:
        trainer.train(trainer.config.num_epochs)
    if trainer.config.do_test:
        trainer.test()


if __name__ == "__main__":
    run()
