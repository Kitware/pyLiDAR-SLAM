import sys
import logging
from argparse import ArgumentParser

from pathlib import Path

from slam.dataset import DATASET
from slam.dataset.urban_loco_dataset import UrbanLocoConfig, UrbanLocoDatasetLoader

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def parse_args():
    """Parses arguments from the command line"""
    parser = ArgumentParser()
    parser.add_argument("--root_dir", type=str,
                        help="The path to the directory containing UrbanLoco's RosBags", required=True)
    args, _ = parser.parse_known_args()

    return args


def main():
    """Main function: builds iteratively the UrbanLoco ground truth files"""
    args = parse_args()

    root_dir = Path(args.root_dir)
    assert root_dir.exists(), f"The UrbanLoco root dir {root_dir} does not exist on disk"

    config = UrbanLocoConfig()

    config.root_dir = args.root_dir
    dataloader: UrbanLocoDatasetLoader = DATASET.load(config)
    dataloader.generate_ground_truth(config.train_sequences)


if __name__ == "__main__":
    main()
