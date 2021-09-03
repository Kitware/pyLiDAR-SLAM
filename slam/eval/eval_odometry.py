from typing import Optional, List, Union, Tuple

import matplotlib
import numpy as np
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from slam.common.utils import assert_debug, check_tensor
from slam.common.io import poses_to_df, delimiter


def draw_trajectory_files(xs: list, ys: list,
                          output_file: str, labels: list = None,
                          figsize: Optional[Tuple] = None, font_size: int = 20,
                          palette: Optional[list] = None):
    """
    Draws multiple 2D trajectories in matplotlib plots and saves the plots as png images

    Parameters
    ----------
    xs : list of arrays
        The list of xs arrays
    ys : list of arrays
        The list of ys arrays
    output_file :
        The output file
    labels : Optional[list]
        An optional list of labels to be displayed in the trajectory
    figsize : Optional[Tuple]
    font_size : int
        The font size of the legend

    """
    sns.set_theme(style="darkgrid")
    fig = plt.figure(figsize=figsize if figsize is not None else (10., 10.), dpi=100, clear=True)

    matplotlib.rcParams.update({'font.size': font_size})
    plt.rc('font', size=font_size)
    plt.rc('axes', labelsize=font_size)
    plt.rc('axes', titlesize=font_size)
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    plt.rc('xtick', labelsize=font_size)
    plt.rc('legend', fontsize=font_size)
    plt.rc('legend', title_fontsize=font_size)

    _labels = []
    _xs = []
    _ys = []

    for i, (x, y) in enumerate(zip(xs, ys)):
        _xs.append(x.reshape(-1, 1))
        _ys.append(y.reshape(-1, 1))
        s = pd.Series([i if labels is None else labels[i]]).repeat(x.shape[0])
        _labels.append(s)
    _xs = np.concatenate(_xs, axis=0)
    _ys = np.concatenate(_ys, axis=0)
    df = pd.DataFrame(np.concatenate([_xs, _ys], axis=1), columns=["x[m]", "y[m]"])
    _labels = pd.concat(_labels, ignore_index=True)
    df["Trajectory"] = _labels

    axes = plt.gca()
    sns.lineplot(x="x[m]", y="y[m]", sort=False, data=df, hue="Trajectory", lw=4,
                 palette="tab10" if palette is None else palette, axes=axes)
    plt.axis("equal")

    leg = axes.legend()
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    fig.set_dpi(100)
    plt.savefig(output_file)
    plt.close(fig)


def list_poses_to_poses_array(poses_list: list):
    return np.concatenate([np.expand_dims(pose, axis=0) for pose in poses_list], axis=0)


def shift_poses(poses: np.ndarray):
    shifted = poses[:-1, :4, :4]
    shifted = np.concatenate([np.expand_dims(np.eye(4), axis=0), shifted], axis=0)
    return shifted


def compute_relative_poses(poses: np.ndarray):
    shifted = shift_poses(poses)
    relative_poses = np.linalg.inv(shifted) @ poses
    return relative_poses


def compute_absolute_poses_(relative_poses: np.ndarray, absolute_poses: np.ndarray):
    for i in range(absolute_poses.shape[0] - 1):
        pose_i = absolute_poses[i, :, :].copy()
        r_pose_ip = relative_poses[i + 1, :, :].copy()
        absolute_poses[i + 1, :] = np.dot(pose_i, r_pose_ip)


def compute_absolute_poses(relative_poses: np.ndarray):
    absolute_poses = relative_poses.copy()
    compute_absolute_poses_(relative_poses, absolute_poses)
    return absolute_poses


def compute_cumulative_trajectory_length(trajectory: np.ndarray):
    shifted = shift_poses(trajectory)
    lengths = np.linalg.norm(shifted[:, :3, 3] - trajectory[:, :3, 3], axis=1)
    lengths = np.cumsum(lengths)
    return lengths


def rotation_error(pose_err: np.ndarray):
    _slice = []
    if len(pose_err.shape) == 3:
        _slice.append(slice(pose_err.shape[0]))

    a = pose_err[tuple(_slice + [slice(0, 1), slice(0, 1)])]
    b = pose_err[tuple(_slice + [slice(1, 2), slice(1, 2)])]
    c = pose_err[tuple(_slice + [slice(2, 3), slice(2, 3)])]
    d = 0.5 * (a + b + c - 1.0)
    error = np.arccos(np.maximum(np.minimum(d, 1.0), -1.0))
    error = error.reshape(error.shape[0])
    return error


def translation_error(pose_err: np.ndarray):
    _slice = []
    axis = 0
    if len(pose_err.shape) == 3:
        _slice.append(slice(pose_err.shape[0]))
        axis = 1

    return np.linalg.norm(pose_err[tuple(_slice + [slice(3), slice(3, 4)])], axis=axis)


def lastFrameFromSegmentLength(dist: list, first_frame: int, segment: float) -> int:
    for i in range(first_frame, len(dist)):
        if dist[i] > dist[first_frame] + segment:
            return i
    return -1


__default_segments = [100, 200, 300, 400, 500, 600, 700, 800]


def calcSequenceErrors(trajectory, ground_truth, all_segments=__default_segments, step_size: int = 10) -> list:
    dist = compute_cumulative_trajectory_length(ground_truth)
    n_poses = ground_truth.shape[0]

    errors = []
    for first_frame in range(0, n_poses, step_size):
        for segment_len in all_segments:
            last_frame = lastFrameFromSegmentLength(dist, first_frame, segment_len)

            if last_frame == -1:
                continue

            pose_delta_gt = np.linalg.inv(ground_truth[first_frame]).dot(ground_truth[last_frame])
            pose_delta_traj = np.linalg.inv(trajectory[first_frame]).dot(trajectory[last_frame])
            pose_err = np.linalg.inv(pose_delta_traj).dot(pose_delta_gt)

            r_err = rotation_error(pose_err)
            t_err = translation_error(pose_err)

            num_frames = last_frame - first_frame + 1
            speed = segment_len / (0.1 * num_frames)

            errors.append({"tr_err": t_err / segment_len,
                           "r_err": r_err / segment_len,
                           "segment": segment_len,
                           "speed": speed,
                           "first_frame": first_frame,
                           "last_frame": last_frame})

    return errors


def compute_kitti_metrics(trajectory, ground_truth) -> tuple:
    errors = calcSequenceErrors(trajectory, ground_truth)

    if len(errors) > 0:
        # Compute averaged errors
        tr_err = sum([error["tr_err"] for error in errors])[0]
        rot_err = sum([error["r_err"] for error in errors])[0]
        tr_err /= len(errors)
        rot_err /= len(errors)
        return tr_err, rot_err, errors
    return None, None


def compute_ate(relative_predicted, relative_ground_truth) -> (float, float):
    pred_xyz = relative_predicted[:, :3, 3]
    gt_xyz = relative_ground_truth[:, :3, 3]

    tr_err = np.linalg.norm(pred_xyz - gt_xyz, axis=1)
    ate = tr_err.mean()
    std_dev = np.sqrt(np.power(tr_err - ate, 2).mean())

    return ate, std_dev


def compute_are(relative_trajectory, relative_ground_truth) -> (float, float):
    diff = np.linalg.inv(relative_ground_truth[:, :3, :3]) @ relative_trajectory[:, :3, :3] - np.eye(3)
    r_err = np.linalg.norm(diff, axis=(1, 2))
    are = r_err.mean()
    std_dev = np.sqrt(np.power(r_err - are, 2).mean())
    return are, std_dev


def rescale_prediction(sequence_pred: np.ndarray, sequence_gt: np.ndarray) -> np.ndarray:
    check_tensor(sequence_pred, [-1, 4, 4])
    check_tensor(sequence_gt, [-1, 4, 4])
    rescaled_pred = []
    for i in range(len(sequence_pred)):
        poses_pred = sequence_pred[i]
        poses_gt = sequence_gt[i]
        norm_pred = np.linalg.norm(poses_pred[:3, -1])
        norm_gt = np.linalg.norm(poses_gt[:3, -1])
        scale = 1.0
        if norm_pred > 1e-6:
            scale = np.linalg.norm(norm_gt) / norm_pred
        new_poses = poses_pred.copy()
        new_poses[:3, -1] *= scale
        rescaled_pred.append(new_poses)

    return list_poses_to_poses_array(rescaled_pred)


class OdometryResults(object):
    """
    An object which aggregrates the results of an Odometry benchmark
    """

    def __init__(self, log_dir: str):
        self.log_dir_path = Path(log_dir)
        if not self.log_dir_path.exists():
            self.log_dir_path.mkdir()

        self.metrics = {}

    def add_sequence(self, sequence_id: str,
                     relative_prediction: Union[np.ndarray, List],
                     relative_ground_truth: Optional[Union[np.ndarray, List]],
                     elapsed: Optional[float] = None,
                     mode: str = "normal",
                     additional_metrics_filename: Optional[str] = None):
        """
        Computes the odometry metrics ATE, ARE, tr_err, rot_err for the sequence sequence_id,
        Saves the result in the log_dir, the trajectories and the projected images

        Parameters
        ----------
        sequence_id : str
            The id of the sequence, will be used as key / prefix for the metrics computed
        relative_prediction : Union[list, np.ndarray]
            The prediction of the relative poses
        relative_ground_truth : Optional[Union[list, np.ndarray]]
            The ground truth of the relative poses
        elapsed : Optional[float]
            The optional number of seconds elapsed for the acquisition of the sequence
        mode : str
            The mode of evaluation accepted modes are :
            normal : the poses are evaluated against the ground truth
            rescale_simple : the poses are rescaled with respect to the ground truth using a 5 frame snippet
            eval_rotation : the translation are set to the ground truth, the rotations are set by the gt
            eval_translation : the rotation are set to the ground truth, the translations by the gt

        additional_metrics_filename: Optional[str] = None
            An optional path to a metrics file to which the metrics should be appended
        """
        with_ground_truth = relative_ground_truth is not None
        if isinstance(relative_prediction, list):
            relative_prediction = list_poses_to_poses_array(relative_prediction)
        if with_ground_truth:
            if isinstance(relative_ground_truth, list):
                relative_ground_truth = list_poses_to_poses_array(relative_ground_truth)

            if mode == "rescale_simple":
                relative_prediction = rescale_prediction(relative_prediction, relative_ground_truth)
            elif mode == "eval_rotation":
                relative_prediction[:, :3, 3] = relative_ground_truth[:, :3, 3]
            elif mode == "eval_translation":
                relative_prediction[:, :3, :3] = relative_ground_truth[:, :3, :3]
            assert_debug(list(relative_ground_truth.shape) == list(relative_prediction.shape))

        absolute_pred = compute_absolute_poses(relative_prediction)
        df_relative_pred = poses_to_df(absolute_pred)
        df_relative_pred.to_csv(str(self.log_dir_path / f"{sequence_id}.poses.txt"), sep=delimiter(), index=False)
        draw_trajectory_files([absolute_pred[:, 0, 3]],
                              [absolute_pred[:, 1, 3]],
                              output_file=str(self.log_dir_path / f"trajectory_{sequence_id}.png"),
                              labels=["prediction"])

        if with_ground_truth:
            absolute_gt = compute_absolute_poses(relative_ground_truth)
            # Save the absolute poses (along with the ground truth)

            df_absolute_gt = poses_to_df(absolute_gt)
            df_absolute_gt.to_csv(str(self.log_dir_path / f"{sequence_id}_gt.poses.txt"), sep=delimiter(), index=False)

            # Save the metrics dict
            tr_err, rot_err, errors = compute_kitti_metrics(absolute_pred, absolute_gt)
            if tr_err and rot_err:
                ate, std_ate = compute_ate(relative_prediction, relative_ground_truth)
                are, std_are = compute_are(relative_prediction, relative_ground_truth)

                self.metrics[sequence_id] = {
                    "tr_err": float(tr_err),
                    "rot_err": float(rot_err),
                    "ATE": float(ate),
                    "STD_ATE": float(std_ate),
                    "ARE": float(are),
                    "STD_ARE": float(std_are),
                }
                if elapsed is not None:
                    self.metrics[sequence_id]["nsecs_per_frame"] = float(elapsed / absolute_gt.shape[0])
                self.save_metrics()

            if additional_metrics_filename is not None:
                self.save_metrics(additional_metrics_filename)

            # TODO Add Average translation error as simple metric over all sequences (to have one number)

            # Save the files
            draw_trajectory_files([absolute_pred[:, 0, 3], absolute_gt[:, 0, 3]],
                                  [absolute_pred[:, 1, 3], absolute_gt[:, 1, 3]],
                                  output_file=str(self.log_dir_path / f"trajectory_{sequence_id}_with_gt.png"),
                                  labels=["prediction", "GT"])

    def __add_mean_metrics(self):
        avg_metrics = {
            "tr_err": 0.0,
            "rot_err": 0.0,
            "ATE": 0.0,
            "STD_ATE": 0.0,
            "ARE": 0.0,
            "STD_ARE": 0.0,
            "nsecs_per_frame": 0.0
        }
        count = 0
        for seq_id, metrics_dict in self.metrics.items():
            if seq_id != "AVG":
                for key, metric in metrics_dict.items():
                    avg_metrics[key] += metric
                count += 1
        if count > 0:
            for key, metric in avg_metrics.items():
                avg_metrics[key] = metric / count
            self.metrics["AVG"] = avg_metrics

    def save_metrics(self, filename: str = "metrics.yaml"):
        """
        Saves the metrics dictionary saved as a yaml file
        """
        assert_debug(self.log_dir_path.exists() and self.log_dir_path.is_dir())
        open_file_mode = "w"
        file_path: Path = self.log_dir_path / filename

        with open(str(file_path), open_file_mode) as metrics_file:
            yaml.safe_dump(self.metrics, metrics_file)

    def close(self):
        """
        Close the metrics file
        """
        self.__add_mean_metrics()
        self.save_metrics()

    def __del__(self):
        self.close()
