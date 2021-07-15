from abc import abstractmethod, ABC
from typing import Tuple, Optional

from torch.utils.data import Dataset, ConcatDataset

# Hydra and OmegaConf
from hydra.conf import dataclass
from omegaconf import MISSING

# Project Imports
from slam.common.projection import SphericalProjector
from slam.dataset.sequence_dataset import DatasetOfSequences


@dataclass
class DatasetConfig:
    """A DatasetConfig contains the configuration values used to define a DatasetConfiguration"""
    dataset: str = MISSING

    # The length of the sequence returned
    sequence_len: int = 2

    # ----------------------------------
    # Default item keys in the data_dict
    vertex_map_key: str = "vertex_map"
    numpy_pc_key: str = "numpy_pc"
    absolute_gt_key: str = "absolute_pose_gt"
    with_numpy_pc: bool = True  # Whether to add the numpy pc to the data_dict


class DatasetLoader(ABC):
    """
    A DatasetConfiguration is the configuration for the construction of pytorch Datasets
    """

    def __init__(self, config: DatasetConfig):
        self.config = config

    @abstractmethod
    def projector(self) -> SphericalProjector:
        """
        Returns the Default Spherical Image projector associated to the dataset_config
        """
        raise NotImplementedError("")

    @abstractmethod
    def sequences(self):
        """
        Returns the train, eval and test datasets and the corresponding sequence name

        Returns: (train, eval, test, transform)
            train (Optional[List[Dataset], List]): Is an Optional pair of a list of datasets
                                                   and the corresponding sequences
            eval (Optional[List[Dataset], List]): Idem
            test (Optional[List[Dataset], List]): Idem
            transform (callable): The function applied on the data from the given datasets

        """
        raise NotImplementedError("")

    def get_dataset(self) -> Tuple[Dataset, Dataset, Dataset, callable]:
        """
        Returns:
        (train_dataset, eval_dataset, test_dataset)
            A tuple of `DatasetOfSequences` consisting of concatenated datasets
        """
        train_dataset, eval_datasets, test_datasets, transform = self.sequences()

        def __swap(dataset):
            if dataset[0] is not None:
                return ConcatDataset(dataset[0])
            return None

        train_dataset = __swap(train_dataset)
        eval_datasets = __swap(eval_datasets)
        test_datasets = __swap(test_datasets)

        return train_dataset, eval_datasets, test_datasets, transform

    def get_sequence_dataset(self) -> Tuple[Optional[DatasetOfSequences],
                                            Optional[DatasetOfSequences],
                                            Optional[DatasetOfSequences]]:
        """
        Returns:
            (train_dataset, eval_dataset, test_dataset) : A tuple of `DatasetOfSequences`
        """
        sequence_len = self.config.sequence_len
        train_dataset, eval_datasets, test_datasets, transform = self.sequences()

        def __to_sequence_dataset(dataset_pair):
            if dataset_pair is None or dataset_pair[0] is None:
                return None
            return DatasetOfSequences(sequence_len, dataset_pair[0], dataset_pair[1], transform=transform)

        return tuple(map(__to_sequence_dataset, [train_dataset, eval_datasets, test_datasets]))

    @abstractmethod
    def get_ground_truth(self, sequence_name):
        """Returns the ground truth for the dataset_config for a given sequence"""
        return None
