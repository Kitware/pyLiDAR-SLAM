# Hydra and OmegaConf
import hydra
from omegaconf import DictConfig

# Project Imports
from slam.odometry.odometry_runner import SLAMRunner, SLAMRunnerConfig


@hydra.main(config_path="config", config_name="slam")
def run_slam(cfg: DictConfig) -> None:
    """The main entry point to the script running the SLAM"""
    _odometry_runner = SLAMRunner(SLAMRunnerConfig(**cfg))
    _odometry_runner.run_odometry()


if __name__ == "__main__":
    run_slam()
