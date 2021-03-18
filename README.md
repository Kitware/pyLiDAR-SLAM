# pyLiDAR-SLAM

This codebase proposes modular light python and pytorch implementations of several LiDAR Odometry methods, 
which can easily be evaluated and compared on a set of public Datasets.

It heavily relies on [omegaconf](https://omegaconf.readthedocs.io/en/2.0_branch/) and [hydra](https://hydra.cc/), 
which allows us to easily test the different modules and parameters with few but structured configuration files.


This is a research project provided "as-is" without garanties, 
use at your own risk. It is actively used for internal research thus is likely to be heavily extended, 
rewritten (and hopefully improved) in a near future.



## Overview of methods currently implemented

![Main Architecture](docs/overview.png)


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
    └─ viz                 # Tools for visualization 
├─ tests
├─ run.py                  # Main script to run a LiDAR SLAM on sequences of a Dataset
└─ train.py                # Main script to launch a training
```
## Installing & Running

#### *Installing*:

Clone the project:
```bash 
git clone https://gitlab.kitware.com/keu-computervision/pylidar-slam
```
Install [pytorch](https://pytorch.org/get-started/previous-versions/) and [hydra](https://hydra.cc/docs/intro).

Install the required packages:
```bash
pip install -r requirements.txt
```

#### *Understanding the configuration*:

Using **Hydra**, one can easily change any element of the configuration by adding an argument to the launch script.

The configuration is organised into groups, within each group, multiple base configurations are defined by a simple string.

Selecting within a group a configurations is done as follows:
```bash
python run.py <config-path-to-group>/<group>=<config_name>
```


The Configuration hierarchy in this projects which can be seen in the `config`, is as follows:

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
To the variable declaration (Hydra will complain if this is the case):
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

#### *Running the Odometry:*
The following example runs a Projective Frame-to-Model odometry on KITTI sequences, using the device `cuda:0`: 
```bash
# Set up the following required environment variables
JOB_NAME=kitti_F2M                                          # The folder to log hydra output files
DATASET=KITTI                                               # Name of the Dataset to construct the destination folder 
KITTI_ODOM_ROOT=<path-to-kitti-odometry-root-directory>     # The path to KITTI odometry benchmark files

# Run the script
python run.py dataset=kitti num_workers=4 device=cuda:0 odometry/local_map=projective \
       odometry/initialization=CV odometry.local_map.num_neighbors_normals=15
```

The output files (configuration files, logs, and optionally metrics on the trajectory) will be saved by default at location : 
```bash
.outputs/slam/${DATASET}/${JOB_NAME}/<date_time>/.
```
Note: The output directory can be changed. 

#### *Training PoseNet:*

The following example launches a training of Posenet on Ford Campus Dataset, for 100 epochs:
```bash
DATASET=kitti
JOB_NAME=train_posenet
FORD_CAMPUS_ROOT=<path-to-ford_campus-root>
TRAIN_DIR=.outputs/training/posenet_kitti_unsupervised

# Launches the Training of PoseNet
python train.py +device=cuda:0 +num_workers=4 +num_epochs=100 dataset=ford_campus
```

The output files are saved by default at: 
```bash
.outputs/training/posenet_ford_campus_unsupervised/.  ===> Training files
.outputs/.training/JOB_NAME/<date_time>/.             ===> Hydra logs and config files for the current run 
```


## DATASETS

#### *KITTI*

LiDAR data and ground truth from [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) odometry benchmark.
The expected layout of KITTI data on disk is as follows:
```bash
├─ <KITTI_ODOM_ROOT>    # Root of the Odometry benchmark data
        ├─ poses        # The ground truth files
            ├─ 00.txt
            ├─ 01.txt
                ...
            └─ 10.txt
        └─ sequences
            ├─ 00 
                ├─ calib.txt
                └─ velodyne
                    ├─ 000000.bin
                    ├─ 000001.bin
                        ...
            ├─ 01
                ├─ calib.txt
                └─ velodyne
                    ├─ 000000.bin
                        ...
            ...
```

For example, to run the ICP odometry on KITTI:
```bash
# Set up the following required environment variables
JOB_NAME=kitti_F2M                                          
DATASET=KITTI                                              
KITTI_ODOM_ROOT=<path-to-kitti-odometry-root-directory>     

# Run the script
python run.py dataset=kitti
```

#### *Ford Campus*
LiDAR data and ground truth from [Ford Campus](http://robots.engin.umich.edu/SoftwareData/Ford) dataset. 
The expected layout of Ford Campus data on disk is as follows:
```bash
├─ <FORD_ROOT>    # Root of the data
        ├─ IJRR-Dataset-1        
            └─ SCANS                # The pointclouds are read from the Scan files 
                ├─ Scan0075.mat 
                ├─ Scan0076.mat
                    ...
        └─ IJRR-Dataset-2
            ...
```

Example running the ICP odometry on Ford Campus:
```bash
# Set up the following required environment variables
JOB_NAME=ford_campus_F2M                                          
DATASET=FORD_CAMPUS                                              
FORD_ROOT=<path-to-ford-campus-root-directory>     

# Run the script
python run.py dataset=ford_campus
```

#### *NCLT*
LiDAR data and ground truth from [NCLT](http://robots.engin.umich.edu/SoftwareData/Ford) dataset. 
The expected layout of NCLT data on disk is as follows:
```bash
├─ <NCLT_ROOT>    # Root of the data
        ├─ 2012-04-29               # Date of acquisition of the sequence        
            └─ velodyne_sync                # The folder containing the velodyne aggregated pointclouds
                ├─ 1335704127712909.bin
                ├─ 1335704127712912.bin
                    ...
            └─ groundtruth_2012-04-29.csv   # Corresponding ground truth file
         ... # Idem for other sequences
```

Example running the ICP odometry on NCLT:
```bash
# Set up the following required environment variables
JOB_NAME=nclt_F2M                                          
DATASET=NCLT                                              
NCLT_ROOT=<path-to-ford-campus-root-directory>     

# Run the script
python run.py dataset=nclt
```

### System Tested

| OS            | CUDA   | pytorch  |
| --- | --- | --- |
| Ubuntu 18.04  | 10.2   | 1.7.1    |

### Author
This is a work realised in the context of Pierre Dellenbach PhD thesis under supervision of Bastien Jacquet ([Kitware](https://www.kitware.fr/equipe-vision-par-odinateur/)), 
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
