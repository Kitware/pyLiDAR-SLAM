# pyLiDAR-SLAM

This codebase proposes modular light python and pytorch implementations of several LiDAR Odometry methods, 
which can easily be evaluated and compared on a set of public Datasets.

It heavily relies on [omegaconf](https://omegaconf.readthedocs.io/en/2.0_branch/) and [hydra](https://hydra.cc/), 
which allows us to easily test the different modules and parameters with few but structured configuration files.


This is a research project provided "as-is" without garanties, 
use at your own risk. It is actively used for **[Kitware Vision team](https://www.kitware.fr/equipe-vision-par-odinateur/)** internal research thus is likely to be heavily extended, 
rewritten (and hopefully improved) in a near future.



## Overview

![KITTI Sequence 00 with pyLiDAR-SLAM](docs/data/video_file.webm)

*pyLIDAR-SLAM* is designed to be modular, multiple components are implemented at each stage of the pipeline.

The motivation is to easily test and compare multiple SLAM algorithms in the same conditions, and a variety of datasets.

The image above is an overview of the different method currently implemented in this project. For more details on each item see [toolbox.md](docs/toolbox.md).

The goal for the future is to gradually add functionalities to pyLIDAR-SLAM (Loop Closure, Motion Segmentation, Multi-Sensors, etc...).

## Project structure

```bash
├─ config                  # Configuration files schemas 
├─ slam 
    ├─ common              # Common components 
    ├─ dataset             # Code to load Datasets 
    ├─ eval                # Code to evaluate the quality of SLAM algorithms 
    ├─ models              # PoseNet model code 
    ├─ odometry            # Odometry modules 
    ├─ training            # Modules for training PoseNet models 
    ├─ backend             # Work in Progress
    ├─ loop_closure        # Work in Progress
    └─ viz                 # Tools for visualization 

├─ tests
├─ run.py                  # Main script to run a LiDAR SLAM on sequences of a Dataset
└─ train.py                # Main script to launch a training
```
## Installation 

Clone the project:
```bash 
git clone https://gitlab.kitware.com/keu-computervision/pylidar-slam
```
Install [pytorch](https://pytorch.org/get-started/previous-versions/) and [hydra](https://hydra.cc/docs/intro).

Install the required packages:
```bash
pip install -r requirements.txt
```

## *Understanding the configuration*:

Using **Hydra**, one can easily change any element of the configuration by adding an argument to the launch script.

The configuration is organised into groups, within each group, multiple base configurations are defined by a simple string.

Selecting within a group a configurations is done as follows:
```bash
python run.py <config-path-to-group>/<group>=<config_name>
```


The configuration hierarchy in this project follows the hierarchy of folder `config`, consists of the following groups:

>- **Group** [`dataset`](slam/dataset/dataset.py):
>   - configurations: [`kitti`](slam/dataset/kitti_dataset.py), [`nclt`](slam/dataset/nclt_dataset.py), [`ford_campus`](slam/dataset/ford_campus.py)
>- **Group** [`odometry/initialization`](slam/odometry/initialization.py) (Initialization module for the Frame To Model):
>   - configurations: [`EI`](slam/odometry/initialization.py), [`CV`](slam/odometry/initialization.py), [`PoseNet`](slam/odometry/initialization.py)
>- **Group** [`odometry/local_map`](slam/odometry/local_map.py) (The local map Implementation):
>   - configurations [`projective`](slam/odometry/local_map.py), [`kdtree`](slam/odometry/local_map.py), 

```bash
├─ config
    ├─ hydra                        # Hydra configuration files
    ├─ dataset                      # Dataset Group 
        ├─ kitti.yaml                   # KITTI default configuration
        ├─ nclt.yaml                    # NCLT default configuration
        └─ ford_campus.yaml             # FORD CAMPUS default configuration
    ├─ slam 
        ├─ odometry         
            ├─ alignment                # Alignment Group (will be expended in the future)
                └─ point_to_plane_GN.yaml   # Point-to-Plane alignment for the Frame-to-Model
            ├─ initialization           # Initialization Group
                ├─ CV.yaml                  # Configuration for the constant velocity model
                ├─ PoseNet.yaml             # Configuration for using PoseNet as initialization
                └─ EI.yaml                  # Elevation Image 2D registration configuration
            └─ local_map                # The Local Map Group
                ├─ projective               # The projective Frame-to-Model proposed
                └─ kdtree                   # The kdtree based Frame-to-Model alignemnt 
        ├─ loop_closure                 # Loop Closure Group
        └─ backend                      # Backend Group

    ├─ training                     # Training Group
        ├─ supervised.yaml              # The configuration for   supervised training of PoseNet
        ├─ unsupervised.yaml            # The configuration for unsupervised training of PoseNet 
    ├─ deep_odometry.yaml           # Configuration file for full Deep LiDAR odometry
    ├─ icp_odometry.yaml            # Configuration file for ICP Frame-to-Model odometry
    └─ train_posenet.yaml           # Configuration file to traing PoseNet
```

To change a named variable within a specific configuration, is done as follows:  
```hydra
python run.py <config-path-to-variable>.<var_name>=<value>
```
For root variables, which are not defined in the .yaml files (but in the Structured Config dataclasses), you might also need to append a  `+`
to the variable declaration (Hydra will complain if this is the case):
```hydra
python train.py +num_workers=4 +device=cuda:0
```

Hydra generates a lot of outputs (at each run).
We use the following environment variables to structure the output folder of an execution:

```bash
JOB_NAME= The name of the job being run
DATASET= The name of the dataset the job is run on
TRAIN_DIR= The path to the dataset where training files (checkpoint and tensorboard logs) for PoseNet should be saved
```
Other, dataset specific environment variables are defined below.


Hydra will raise Exceptions on invalid configurations. 
See [hydra](https://hydra.cc/)  documentation for more details.

## Using *pyLIDAR-SLAM* (with *hydra*):

> *pyLIDAR-SLAM* proposes two scripts `run.py` and `train.py` which run with hydra's configuration generation (notably hydra's configuration verification).

> Hydra comes with many perks designed for a research workflow (multiple runs with grids of arguments, automatic setting of output directories, etc..). 
> But requires an effort to get into: hydra's enforced rigidity often leads to many configuration errors. *Read carefully hydra's error messages which give clues to the configuration errors*

### Running the SLAM 

The script `run.py` executes the SLAM algorithm defined by the configuration on the datasets defined in the configuration.
More specifically, it will:
 - Load all sequences of the given dataset
 - Sequentially launch the SLAM on each sequence
 - Output the estimated trajectory in the allocated folder
 - And if the sequence has defined ground truth poses, computes and saves the trajectory error.

The following example runs a **Projective Frame-to-Model Odometry** on the sequences of KITTI dataset, using the device `cuda:0`: 
```bash
# Set up the following required environment variables
export JOB_NAME=kitti_F2M                                          # The folder to log hydra output files
export DATASET=KITTI                                               # Name of the Dataset to construct the destination folder 
export KITTI_ODOM_ROOT=<path-to-kitti-odometry-root-directory>     # The path to KITTI odometry benchmark files

# Run the script
python run.py dataset=kitti num_workers=4 device=cuda:0 slam/odometry/local_map=projective \
       slam/odometry/initialization=CV slam.odometry.local_map.num_neighbors_normals=15
```

The output files (configuration files, logs, and optionally metrics on the trajectory) will be saved by default at location : 
```bash
.outputs/slam/${DATASET}/${JOB_NAME}/<date_time>/.
```

> Note: The output directory can be changed, using arguments `hydra.run.dir=<path-to-output-dir>`.

### Training PoseNet:

The script `train.py` launches a training of *PoseNet* on a specified dataset.
The script will:
 - Load all train, test and eval sequences from the Dataset 
 - For each epoch, update the running PoseNet model in memory
 - Save the model at a specified given location

The following example launches a training of Posenet on Ford Campus Dataset, for 100 epochs:
```bash
export DATASET=kitti
export JOB_NAME=train_posenet
export KITTI_ODOM_ROOT=<path-to-kitti-odometry-root-directory>     # The path to KITTI odometry benchmark files
export TRAIN_DIR=<absolute-path-to-the-desired-train-dir>          # Path to the output models 

# Launches the Training of PoseNet
python train.py +device=cuda:0 +num_workers=4 +num_epochs=100 dataset=kitti
```

> /!\ The working directory of the script is controlled by hydra, so beware of relative paths!

The output files are saved by default at: 
```bash
.outputs/training/posenet_kitti_unsupervised/.          ===> Training files
.outputs/.training/<JOB_NAME>/<date_time>/.             ===> Hydra logs and config files for the current run 
```


### DATASETS

*pyLIDAR-SLAM* incorporates different datasets, see [datasets.md](docs/datasets.md) for installation and setup instructions for each of these datasets.
Only the datasets implemented in *pyLIDAR-SLAM* are compatible with hydra's mode and the scripts `run.py` and `train.py`. 

But you can define your own datasets by extending the class [`DatasetLoader`](slam/dataset/dataset.py).


## Benchmarks

One of the motivation of *pyLIDAR-SLAM* is to be able to compare the performances of its different modules on different datasets.
In [benchmark.md](docs/benchmark.md) we present the results of *pyLIDAR-SLAM* on the most popular open-source datasets. 

### System Tested

| OS            | CUDA   | pytorch  |
| --- | --- | --- |
| Ubuntu 18.04  | 10.2   | 1.7.1    |

### Author
This is a work realised in the context of Pierre Dellenbach PhD thesis under supervision of [Bastien Jacquet](https://www.linkedin.com/in/bastienjacquet/?originalSubdomain=fr) ([Kitware](https://www.kitware.com/computer-vision/)), 
Jean-Emmanuel Deschaud & François Goulette (Mines ParisTech).

### Cite

If you use this work for your research, consider citing:

```
@misc{dellenbach2021s,
      title={What's in My LiDAR Odometry Toolbox?},
      author={Pierre Dellenbach, 
      Jean-Emmanuel Deschaud, 
      Bastien Jacquet,
      François Goulette},
      year={2021},
}
```
