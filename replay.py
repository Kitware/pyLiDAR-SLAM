# Hydra and OmegaConf
from dataclasses import MISSING, dataclass, field
import numpy as np
from pathlib import Path
from typing import Optional
from slam.common.modules import _with_viz3d

if _with_viz3d:
    from viz3d.window import OpenGLWindow

from omegaconf import DictConfig, OmegaConf

# Project Imports
from tqdm import tqdm

from slam.dataset.dataset import WindowDataset
from slam.odometry.odometry_runner import SLAMRunner
from argparse import ArgumentParser


@dataclass
class ReplayArguments:
    config_path: str = ""  # The path to the SLAMRunner Config
    sequence: str = ""  # The name of sequence to replay
    root_dir: Path = field(default_factory=lambda: Path())
    sequence_dir: Path = field(default_factory=lambda: Path())
    start_index: int = 0
    num_frames: int = -1
    show_information: bool = True  # Whether to print information about the to experiment be replayed
    overrides_path: Optional[str] = None  # The path to the yaml containing the overrides


def parse_arguments() -> ReplayArguments:
    parser = ArgumentParser()
    parser.add_argument("--root_dir", type=str, help="Path to the root of the execution", required=True)
    parser.add_argument("--start_index", type=int, help="The index at which the SLAM should start", required=False)
    parser.add_argument("--seq", type=str, help="The name of the sequence to replay", required=True)
    parser.add_argument("--info", action="store_true",
                        help="Whether to display information of the sequence prior to the replay")
    parser.add_argument("--overrides", type=str,
                        help="The path (optional) to the overrides")

    args, _ = parser.parse_known_args()
    options = ReplayArguments()
    root_dir = Path(args.root_dir)
    assert root_dir.exists(), f"The root dir {root_dir} for the execution does not exist"
    options.root_dir = root_dir
    options.sequence_dir = root_dir / args.seq
    assert options.sequence_dir.exists(), f"The sequence dir {options.sequence_dir} does not exist"
    config_path = root_dir / "config.yaml"
    assert config_path.exists(), f"The config path {config_path} does not exist"
    options.config_path = str(config_path)
    options.start_index = args.start_index
    options.sequence = args.seq
    options.show_information = args.info
    options.overrides_path = args.overrides

    return options


def replay_slam(options: ReplayArguments) -> None:
    """The main entry point to the script running the SLAM"""
    # Load the config
    from slam.common.io import read_poses_from_disk
    import time

    # Display information about the previous execution
    poses: Optional[np.ndarray] = None
    poses_file_path = options.sequence_dir / f"{options.sequence}.poses.txt"
    gt_file_path = options.sequence_dir / f"{options.sequence}_gt.poses.txt"
    if poses_file_path.exists():
        poses = read_poses_from_disk(str(poses_file_path))

    if options.show_information:
        print("*" * 80)
        if poses is not None:
            print(f"[INFO]Found Pose Estimate file {poses_file_path}.")
        if gt_file_path.exists():
            print(f"[INFO]Found GT Pose Estimate file {poses_file_path}. The algorithm run to completion.")
        else:
            if poses is not None:
                print(f"[INFO]The execution stopped after {poses.shape[0]} frames.")
        print("*" * 80)

    # Run the algorithm again on the same data
    config: DictConfig = OmegaConf.load(options.config_path)

    if options.overrides_path is not None:
        overrides_conf = OmegaConf.load(options.overrides_path)
        # Merge the two config
        config.merge_with(overrides_conf)

    config.dataset.train_sequences = [options.sequence]
    config.debug = True
    config.log_dir = f"/tmp/{time.time()}"
    Path(config.log_dir).mkdir()

    runner = SLAMRunner(config)

    # Load the Datasets
    datasets: list = runner.load_datasets()
    for sequence_name, dataset in datasets:

        window = None
        try:
            # Build dataloader
            num_frames = options.num_frames if options.num_frames > 0 else len(dataset) - options.start_index
            dataset = WindowDataset(dataset, options.start_index, num_frames)
            slam = runner.load_slam_algorithm()
            slam.init()

            elapsed = 0.0
            relative_ground_truth = runner.ground_truth(sequence_name)
            if _with_viz3d:
                window = OpenGLWindow()
                if poses is not None:
                    window.init()
                    saved_poses = poses[options.start_index:]
                    saved_poses = np.einsum("ij,njk->nik", np.linalg.inv(saved_poses[0]), saved_poses)
                    window.set_poses(0, saved_poses.astype(np.float32))

            for data_dict in tqdm(dataset, desc=f"Sequence {sequence_name}", ncols=100, total=num_frames):
                start = time.time()

                # Process next frame
                slam.process_next_frame(data_dict)

                # Measure the time spent on the processing of the next frame
                elapsed_sec = time.time() - start
                elapsed += elapsed_sec

        except KeyboardInterrupt:
            if _with_viz3d and window is not None:
                window.close(True)

    del slam


if __name__ == "__main__":
    options: ReplayArguments = parse_arguments()
    replay_slam(options)
