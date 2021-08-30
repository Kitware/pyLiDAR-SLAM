from pathlib import Path

import numpy as np
import pandas as pd

# Project Imports
from slam.common.utils import assert_debug, check_tensor


def delimiter():
    """
    The column delimiter in pandas csv
    """
    return ","


def write_poses_to_disk(file_path: str, poses: np.ndarray):
    """
    Writes an array of poses to disk

    Parameters
    ----------
    file_path : str
    poses : np.ndarray [N, 4, 4]
    """
    check_tensor(poses, [-1, 4, 4])
    path = Path(file_path)
    assert_debug(path.parent.exists())
    poses_to_df(poses).to_csv(file_path, sep=delimiter(), index=False)


def read_poses_from_disk(file_path: str, _delimiter: str = delimiter()) -> np.ndarray:
    """
    Reads an array of poses from disk

    Returns
    ----------
    poses : np.ndarray [N, 4, 4]
    """
    path = Path(file_path)
    assert_debug(path.exists() and path.is_file())
    return df_to_poses(pd.read_csv(path, sep=_delimiter, index_col=None))


def df_to_poses(df: pd.DataFrame) -> np.ndarray:
    """
    Converts a pd.DataFrame to a [N, 4, 4] array of poses read from a dataframe

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame of size [N, 12] with the 12 values of the first 3 rows of the pose matrix
    """
    array = df.to_numpy(dtype=np.float32)
    assert_debug(array.shape[1] == 12)
    num_rows = array.shape[0]
    poses = array.reshape([num_rows, 3, 4])
    last_row = np.concatenate((np.zeros((num_rows, 3)), np.ones((num_rows, 1))), axis=1)
    last_row = np.expand_dims(last_row, axis=1)
    poses = np.concatenate((poses, last_row), axis=1)

    return poses


def poses_to_df(poses_array: np.ndarray):
    """
    Converts an array of pose matrices [N, 4, 4] to a DataFrame [N, 12]
    """
    shape = poses_array.shape
    assert_debug(len(shape) == 3)
    assert_debug(shape[1] == 4 and shape[2] == 4)
    nrows = shape[0]
    reshaped = poses_array[:, :3, :].reshape([nrows, 12])
    df = pd.DataFrame(reshaped)

    return df
