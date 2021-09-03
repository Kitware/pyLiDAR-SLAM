from typing import Optional
from torch.utils.data import Dataset
import collections
from torch.utils.data._utils.collate import default_collate
import numpy as np


class DatasetOfSequences(Dataset):
    """
    A Dataset which concatenates data into for a small window of frames

    Takes a list of Datasets, each corresponding to a sequence.
    The dataset_config created returns a Dataset of windows.
    Each window takes consecutive frames of a given sequence and
    concatenates them

    Parameters
        sequence_len (int): The length of a window of frames
        dataset_list (list): The list of dataset_config
        sequence_ids (list): The list ids of the dataset_config list
        sequence_transforms (callable): A Transform to be applied
    """

    def __init__(self,
                 sequence_len: int,
                 dataset_list: list,
                 sequence_ids: list = None,
                 transform: Optional[callable] = None,
                 sequence_transforms: Optional[callable] = None,
                 stride: int = 1):
        self.datasets: list = dataset_list
        self.dataset_sizes: list = [0]
        self.sequence_len: int = sequence_len
        self.transform: Optional[callable] = transform
        self.sequence_transforms: Optional[callable] = sequence_transforms
        self.stride = stride

        for i in range(len(dataset_list)):
            num_sequences_in_dataset = len(dataset_list[i]) - self.sequence_len * self.stride + 1
            self.dataset_sizes.append(self.dataset_sizes[i] + num_sequences_in_dataset)
        self.size = self.dataset_sizes[-1]
        self.sequence_ids = sequence_ids

    def find_dataset_with_idx(self, idx):
        assert idx < self.size, "INVALID ID"
        dataset_idx, sizes = list(x for x in enumerate(self.dataset_sizes) if x[1] <= idx)[-1]
        return self.datasets[dataset_idx], idx - sizes, self.sequence_ids[dataset_idx]

    def load_sequence(self, dataset, indices):
        sequence = []
        for seq_index in indices:
            data_dict = dataset[seq_index]
            if self.transform is not None:
                data_dict = self.transform(data_dict)
            sequence.append(data_dict)
        return sequence

    def transform_sequence(self, elem):
        if self.sequence_transforms:
            elem = self.sequence_transforms(elem)
        return elem

    def __getitem__(self, idx):
        dataset, start_idx_in_dataset, seq_id = self.find_dataset_with_idx(idx)
        indices = [start_idx_in_dataset + i * self.stride for i in range(self.sequence_len)]

        sequence = self.load_sequence(dataset, indices)
        sequence_item = self.__sequence_collate(sequence)

        sequence_item = self.transform_sequence(sequence_item)
        return sequence_item

    def __len__(self):
        return self.size

    @staticmethod
    def __sequence_collate(batch):
        """
        Agglomerate window data for a sequence

        Args:
            batch (List): A list of elements which are to be aggregated into a batch of elements
        """
        elem = batch[0]
        if elem is None:
            return batch
        if isinstance(elem, collections.Mapping):
            result = dict()
            for key in elem:
                if "numpy" in key:
                    for idx, d in enumerate(batch):
                        result[f"{key}_{idx}"] = d[key]
                else:
                    result[key] = DatasetOfSequences.__sequence_collate([d[key] for d in batch])
            return result
        elif isinstance(elem, np.ndarray):
            return np.concatenate([np.expand_dims(e, axis=0) for e in batch], axis=0)
        else:
            return default_collate(batch)

    @staticmethod
    def _params_type() -> str:
        return DatasetOfSequences.__name__
