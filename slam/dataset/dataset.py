from typing import Optional

from torch.utils.data import Dataset

from slam.common.utils import assert_debug


class WrapperDataset(Dataset):
    """
    A Wrapper Dataset which applies a transform on the items returned by a torch.Dataset
    """

    def __init__(self, dataset: Dataset, transform: Optional[callable] = None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        transformed = self.transform(data)
        return transformed


class WindowDataset(Dataset):
    """
    A Window datasets wraps a dataset_config, and limit it to a defined window
    """

    def __init__(self, dataset: Dataset, start: int = 0, length: int = -1):
        self.dataset = dataset
        assert_debug(start < len(dataset))
        assert_debug(start + length <= len(dataset))

        self.length = length
        self.start = start

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.dataset[item + self.start]
