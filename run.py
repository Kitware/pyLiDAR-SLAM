# Hydra and OmegaConf
import hydra
from omegaconf import DictConfig

# Project Imports
from slam.odometry.odometry_runner import SLAMRunner, SLAMConfig


@hydra.main(config_path="config", config_name="icp_odometry")
def run_odometry(cfg: DictConfig) -> None:
    """The main entry point to the script running the odometry"""
    _odometry_runner = SLAMRunner(SLAMConfig(**cfg))
    _odometry_runner.run_odometry()


if __name__ == "__main__":
    run_odometry()
