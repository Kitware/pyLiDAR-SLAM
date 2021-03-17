from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List
import os

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

# Hydra and OmegaConf
from hydra.conf import dataclass, field, MISSING
from omegaconf import OmegaConf

# Project Imports
from slam.common.torch_utils import send_to_device, collate_fun
from slam.common.utils import assert_debug, get_git_hash
from slam.viz.visualizer import ImageVisualizer
from slam.viz.color_map import tensor_to_image


class AverageMeter(object):
    """
    An util object which progressively computes the mean over logged values
    """

    def __init__(self):
        self.average = 0.0
        self.count = 1

    def update(self, loss: float):
        """Adds a new item to the meter"""
        if isinstance(loss, torch.Tensor):
            loss = loss.cpu().detach().item()
        self.average = (self.average * self.count + loss) / (self.count + 1)
        self.count += 1


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class ATrainerConfig:
    """The configuration dataclass for a Trainer"""
    train_dir: str = MISSING  # The train directory

    # Configuration for the current run
    do_train: bool = True
    do_test: bool = True
    num_epochs: int = 100  # Number of epochs for the current run

    # Standard training parameters
    device: str = "cpu"
    num_workers: int = 0
    shuffle: bool = True
    batch_size: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 5

    # IO : for saving/loading state dicts
    out_checkpoint_file: str = "checkpoint.ckp"
    in_checkpoint_file: Optional[str] = "checkpoint.ckp"

    # Optimizer params
    optimizer_type: str = "adam"
    optimizer_momentum: float = 0.9
    optimizer_beta: float = 0.999
    optimizer_weight_decay: float = 0.001  # /!\ Important for PoseNet training stability
    optimizer_learning_rate: float = 0.0001
    optimizer_sgd_dampening: float = 0.0
    optimizer_sgd_nesterov: float = False
    # Scheduler params
    optimizer_scheduler_decay: float = 0.5
    optimizer_scheduler_milestones: List[int] = field(default_factory=lambda: [20 * i for i in range(10)])

    # Logging and visualization fields
    visualize: bool = False
    test_frequency: int = 20  # The number of epochs before launching testing
    visualize_frequency: int = 20
    scalar_log_frequency: int = 20
    tensor_log_frequency: int = 200
    average_meter_frequency: int = 50
    # Keys of tensors to log to tensorboard and visualize
    log_image_keys: List[str] = field(default_factory=list)  # The keys in the data_dict to add to tensorboard as images
    log_scalar_keys: List[str] = field(
        default_factory=list)  # The keys in the data_dict to add to tensorboard as scalar
    log_histo_keys: list = field(default_factory=list)
    viz_image_keys: list = field(default_factory=list)


# ----------------------------------------------------------------------------------------------------------------------
class ATrainer(ABC):
    """
    An abstract Trainer class is the backbone for training deep learning modules

    Each ATrainer child classes defines a prediction Module and a loss Module, and how the data should be loaded

    # TODO Rewrite to allow training on multiple GPUs
    """

    def __init__(self,
                 config: ATrainerConfig):

        self.config = config

        # Training state variables
        self.num_epochs = 0
        self.train_iter: int = 0
        self.eval_iter: int = 0
        self.eval_: bool = False
        self.train_: bool = False
        self.test_: bool = False
        self.test_frequency = 10

        # Loggers and Visualizer
        self.logger: Optional[SummaryWriter] = None
        self.do_visualize: bool = False
        self.image_visualizer: Optional[ImageVisualizer] = None
        self.average_meter: Optional[AverageMeter] = None

        # -- Averages of the last training epochs
        self._average_train_loss: Optional[float] = None
        self._average_eval_loss: Optional[float] = None

        # Load the params from the registered params
        self.__load_params()

        # -- Optimizer variables
        self._optimizer: Optional[Optimizer] = None
        self._scheduler: Optional[MultiStepLR] = None

        # -- Parameters that should be defined by child classes
        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.loss_module_: Optional[nn.Module] = None
        self.prediction_module_: Optional[nn.Module] = None

    def __load_params(self):
        # ATrainer Params

        # -- Load the directories / file paths, etc...
        train_dir = self.config.train_dir
        train_dir_path = Path(train_dir)
        assert_debug(train_dir is not None, "The key 'train_dir' must be specified in the training params")
        if not train_dir_path.exists():
            assert_debug(Path(train_dir_path.parent).exists(),
                         f"Both the train directory {train_dir} and its parent do not exist")
            train_dir_path.mkdir()
        assert_debug(train_dir_path.is_dir(), f"The directory 'train_dir':{train_dir} does not exist")

        in_checkpoint_file = self.config.in_checkpoint_file
        out_checkpoint_file = self.config.out_checkpoint_file
        assert_debug(out_checkpoint_file is not None)
        if in_checkpoint_file is not None:
            self.input_checkpoint_file = str(train_dir_path / in_checkpoint_file)
        self.output_checkpoint_file = str(train_dir_path / out_checkpoint_file)
        self.log_dir = train_dir

        # -- Load the training params
        self.device = torch.device(self.config.device)

    def init(self):
        """
        Initializes the ATrainer

        It will in order :
            - load the training, validation and test dataset_config
            - load the prediction module and loss module
            - send the modules to the chosen device
            - load the optimizer
            - reload the input_checkpoints (if it was specified)

        """
        # Loads the datasets
        self.train_dataset, self.eval_dataset, self.test_dataset = self.load_datasets()

        self.loss_module_ = self.loss_module()
        self.prediction_module_ = self.prediction_module()

        self.loss_module_ = self.loss_module_.to(self.device)
        self.prediction_module_ = self.prediction_module_.to(self.device)

        self._optimizer = self._load_optimizer()

        # Load checkpoint file
        if self.input_checkpoint_file:
            self.load_checkpoint()

        if self._optimizer:
            self._scheduler = MultiStepLR(self._optimizer,
                                          milestones=self.config.optimizer_scheduler_milestones,
                                          gamma=self.config.optimizer_scheduler_decay,
                                          last_epoch=self.num_epochs - 1)

        # -- Copy the configuration file in the train dir
        train_dir_path = Path(self.config.train_dir)
        with open(str(train_dir_path / "config.yaml"), "w") as config_file:
            # Add the git hash to improve tracking of modifications
            config_dict = self.config.__dict__

            git_hash = get_git_hash()
            if git_hash is not None:
                config_dict["git_hash"] = git_hash
            config_dict["_working_dir"] = os.getcwd()
            config_file.write(OmegaConf.to_yaml(config_dict))

    def _load_optimizer(self) -> Optimizer:
        loss_module_params = {"params": self.loss_module_.parameters(),
                              "lr": self.config.optimizer_learning_rate}
        prediction_module_params = {
            "params": self.prediction_module_.parameters(),
            "lr": self.config.optimizer_learning_rate
        }
        optimizer_type = self.config.optimizer_type
        assert_debug(optimizer_type in ["adam", "adamw", "rmsprop", "sgd"])
        if optimizer_type == "adam":
            return torch.optim.Adam([prediction_module_params, loss_module_params],
                                    betas=(self.config.optimizer_beta, self.config.optimizer_momentum),
                                    weight_decay=self.config.optimizer_weight_decay)
        elif optimizer_type == "adamw":
            return AdamW([prediction_module_params, loss_module_params],
                         betas=(self.config.optimizer_beta, self.config.optimizer_momentum),
                         weight_decay=self.config.optimizer_weight_decay)
        elif optimizer_type == "sgd":
            return torch.optim.SGD([prediction_module_params,
                                    loss_module_params],
                                   lr=self.optimizer_learning_rate,
                                   momentum=self.config.optimizer_momentum,
                                   weight_decay=self.config.optimizer_weight_decay,
                                   dampening=self.config.optimizer_sgd_nesterov)
        elif optimizer_type == "rmsprop":
            return torch.optim.RMSprop([prediction_module_params,
                                        loss_module_params],
                                       lr=self.optimizer_learning_rate,
                                       weight_decay=self.config.optimizer_weight_decay,
                                       momentum=self.config.optimizer_momentum)
        else:
            raise NotImplementedError("")

    def _init_logger(self):
        if self.log_dir is None:
            return None
        return SummaryWriter(log_dir=self.log_dir)

    def _init_visualizer(self) -> Optional[ImageVisualizer]:
        return ImageVisualizer(self.config.viz_image_keys, update_frequency=self.config.visualize_frequency)

    def train(self, num_epochs: int = 10):
        """
        Launches num_epochs of training

        Parameters
        ----------
        num_epochs : int
            The number of epochs to launch
        """
        # progress_bar = tqdm(range(num_epochs), desc=f"Training {num_epochs} new epochs", ncols=max(num_epochs, 100))
        for i in range(num_epochs):
            self.train_epoch()
            self.evaluate_epoch()
            self.save_checkpoint()
            if i > 0 and i % self.test_frequency == 0:
                self.test()
            if self._scheduler:
                self._scheduler.step()

                lr = self._scheduler.get_last_lr()
                if lr != self.config.optimizer_learning_rate:
                    print(f"Last learning rate : {lr}")
                    self.config.optimizer_learning_rate = lr

    @staticmethod
    def progress_bar(dataloader: DataLoader, desc: str = ""):
        return tqdm(enumerate(dataloader, 0),
                    desc=desc,
                    total=len(dataloader),
                    ncols=120, ascii=True)

    def train_epoch(self):
        """
        Launches the training for an epoch
        """
        assert_debug(self.loss_module_ is not None)
        assert_debug(self.prediction_module_ is not None)

        self.prediction_module_.train()
        self.loss_module_.train()
        self.train_ = True
        if self.config.num_workers == 0:
            dataloader = DataLoader(
                self.train_dataset,
                pin_memory=self.config.pin_memory,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                collate_fn=collate_fun,
                shuffle=self.config.shuffle)

        else:
            dataloader = DataLoader(
                self.train_dataset,
                pin_memory=self.config.pin_memory,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                prefetch_factor=self.config.prefetch_factor,
                collate_fn=collate_fun,
                shuffle=self.config.shuffle)

        loss_meter = AverageMeter()
        progress_bar = self.progress_bar(dataloader, desc=f"Training epoch n°{self.num_epochs}")
        for batch_idx, batch in progress_bar:
            # Reinitialize the optimizer
            self._optimizer.zero_grad()
            # send the data to the GPU
            batch = self.send_to_device(batch)

            # Prediction step
            prediction_dict = self.prediction_module_(batch)
            # Loss step
            loss, log_dict = self.loss_module_(prediction_dict)
            if loss is not None:
                if torch.any(torch.isnan(loss)):
                    raise RuntimeError("[ERROR]Loss is NaN.")
                # Optimizer step
                try:
                    loss.backward()
                    self._optimizer.step()
                except RuntimeError as e:
                    print("[ERROR]NaN During back progation... Good luck.")
                    raise e

                if batch_idx % self.config.average_meter_frequency == 0:
                    loss_meter.update(loss.detach().cpu())

            self.log_dict(log_dict)
            self.train_iter += 1
        # Save module to checkpoint
        self.num_epochs += 1
        self.train_ = False
        if loss_meter.count > 0:
            print(f"Train average loss : {loss_meter.average}")
            self._average_train_loss = loss_meter.average

    def evaluate_epoch(self):
        """
        Launches the evaluation for an epoch
        """
        if self.eval_dataset is None:
            return

        assert_debug(self.loss_module_ is not None)
        assert_debug(self.prediction_module_ is not None)

        self.eval_ = True
        self.prediction_module_.eval()
        self.loss_module_.eval()

        dataloader = DataLoader(
            self.eval_dataset,
            pin_memory=self.config.pin_memory,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=collate_fun)

        loss_meter = AverageMeter()
        progress_bar = progress_bar = self.progress_bar(dataloader, "Eval epoch")
        for batch_idx, batch in progress_bar:
            # send the data to the GPU
            batch = self.send_to_device(batch)

            # Prediction step
            prediction_dict = self.prediction_module_(batch)
            # Loss step
            loss, log_dict = self.loss_module_(prediction_dict)
            # Log the log_dict
            self.log_dict(log_dict)
            if loss:
                loss_meter.update(loss.detach())

            self.eval_iter += 1
        self.eval_ = False
        if loss_meter.count > 0:
            print(f"Eval  average loss : {loss_meter.average}")
            self._average_eval_loss = loss_meter.average

    def load_checkpoint(self, fail_if_absent: bool = False):
        """
        Loads a checkpoint file saved during training

        The checkpoint file is a python dictionary saved,
        The dictionary contains the parameters of the optimizer and
        The submodules of the loss module and the .train module

        To load the models, the variable input_checkpoint_file must be defined,
        and pointing to an existing checkpoint file

        Parameters
        ----------
        fail_if_absent : bool
            Whether to fail if the checkpoint file does not exist
        """
        if not self.input_checkpoint_file:
            return
        checkpoint_path = Path(self.input_checkpoint_file)
        if fail_if_absent:
            assert_debug(checkpoint_path.exists() and checkpoint_path.is_file(), "The checkpoint file does not exist")
        else:
            if not checkpoint_path.exists():
                return
        state_dict = torch.load(str(checkpoint_path))

        # Load the optimizer from the state dict
        self._optimizer.load_state_dict(state_dict["optimizer"])
        self.loss_module_.load_state_dict(state_dict["loss_module"])
        self.prediction_module_.load_state_dict(state_dict["prediction_module"])
        self.num_epochs = state_dict["num_train_epochs"]
        self.eval_iter = state_dict["eval_iter"]
        self.train_iter = state_dict["train_iter"]
        if "average_eval_loss" in state_dict:
            self._average_eval_loss = state_dict["average_eval_loss"]
        if "average_train_loss" in state_dict:
            self._average_train_loss = state_dict["average_train_loss"]
        if "last_lr" in state_dict:
            self.config.optimizer_learning_rate = state_dict["last_lr"]

    def save_checkpoint(self):
        """
        Saves the modules and optimizer parameters in a checkpoint file
        """
        if not self.output_checkpoint_file:
            return

        state_dict = {
            "optimizer": self._optimizer.state_dict(),
            "loss_module": self.loss_module_.state_dict(),
            "prediction_module": self.prediction_module_.state_dict(),
            "num_train_epochs": self.num_epochs,
            "train_iter": self.train_iter,
            "eval_iter": self.eval_iter
        }
        if self._average_eval_loss is not None:
            state_dict["average_eval_loss"] = self._average_eval_loss
        if self._average_train_loss is not None:
            state_dict["average_train_loss"] = self._average_train_loss
        if self._scheduler:
            state_dict["last_lr"] = self._scheduler.get_last_lr()

        torch.save(state_dict, self.output_checkpoint_file)

    def send_to_device(self, data_dict: dict) -> dict:
        """
        Default method to send a dictionary to a device

        By default, only tensors are sent to the GPU

        Parameters
        ----------
        data_dict : dict
            A dictionary of objects to send to the device
        """
        return send_to_device(data_dict, self.device)

    def log_dict(self, log_dict: dict):
        """
        Logs the output of the

        Parameters
        ----------
        log_dict : dict
            The dictionary of items to be logged
        """
        if self.logger is None:
            # Init the logger
            self.logger = self._init_logger()

        # Init the visualizer
        if self.image_visualizer is None and self.do_visualize:
            self.image_visualizer = self._init_visualizer()
            if self.image_visualizer is None:
                self.do_visualize = False

        if self.logger is None:
            return

        assert_debug(self.train_ or self.eval_)
        if self.train_:
            _iter = self.train_iter
            tag_prefix = ".train/"
        else:
            _iter = self.eval_iter
            tag_prefix = ".eval/"

        if _iter % self.config.scalar_log_frequency == 0:
            # Log scalars
            for scalar_key in self.config.log_scalar_keys:
                assert_debug(scalar_key in log_dict, f"scalar key {scalar_key} not in log_dict")
                item = log_dict[scalar_key]
                if isinstance(item, torch.Tensor):
                    item = item.item()
                self.logger.add_scalar(f"{tag_prefix}{scalar_key}", item, _iter)

        if _iter % self.config.tensor_log_frequency == 0:
            # Log histograms
            for histo_key in self.config.log_histo_keys:
                assert_debug(histo_key in log_dict)
                self.logger.add_histogram(f"{tag_prefix}{histo_key}", log_dict[histo_key], _iter)

            # Log images
            for image_key in self.config.log_image_keys:
                assert_debug(image_key in log_dict)
                image = tensor_to_image(log_dict[image_key])
                self.logger.add_image(f"{tag_prefix}{image_key}", image, _iter)

        if self.image_visualizer is not None:
            self.image_visualizer.visualize(log_dict, _iter)

    @abstractmethod
    def prediction_module(self) -> nn.Module:
        """
        Returns the prediction module for the specific trainer

        Returns
        -------
        nn.Module
            The evaluation module which extracts data for the evaluation
            The module takes the dict with the data for each iter,
            And returns a dict with all the predictions and data expected by the loss Module or the evaluation

        """
        raise NotImplementedError()

    @abstractmethod
    def loss_module(self) -> nn.Module:
        """
        Returns the loss module for the specific trainer

        Returns
        -------
        nn.Module
            The loss module computes the loss on which will be applied the gradient descent
            The module should expect the dict returned from the prediction module
            And returns a tuple, with the first item the loss (as a torch.Tensor)
            And the second item a dictionary with all the data to be logged
        """

        raise NotImplementedError()

    @abstractmethod
    def load_datasets(self) -> (Optional[Dataset], Optional[Dataset], Optional[Dataset]):
        """
        Returns the .train, validation and test datasets as options
        """
        raise NotImplementedError()

    @abstractmethod
    def test(self):
        raise NotImplementedError()
