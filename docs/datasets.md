## DATASETS

>PyLIDAR-SLAM integrates the following LiDAR Datasets:
> - [KITTI](#kitti)
> - [Ford-Campus](#ford_campus)
> - [NCLT](#nclt)
>
>The following describes for each dataset the installation instructions, how to adapt the configurations, and how to run
the SLAM on these Datasets. 

#### Hydra Configurations

> For each dataset a default YAML config file is given in the folder [dataset](../config/dataset). 
> These configs are loaded by the scripts `run.py` and `train.py` while loading the config. To select the appropriate dataset, use the command argument `dataset=<dataset_name>` where `<dataset_name>` is the name of the associated hydra config.
> For example `python run.py dataset=kitti` will launch the default SLAM on the KITTI benchmark.

> The hydra dataset config files are:
  - [ford_campus.yaml](../config/dataset/ford_campus.yaml)
  - [kitti.yaml](../config/dataset/kitti.yaml)
  - [nclt.yaml](../config/dataset/nclt.yaml)

> To modify fields of a `dataset` config in the command line, pass it as argument using hydra's mechanism, (e.g. `python run.py dataset=kitti dataset.lidar_height=64 dataset.lidar_width=840` will project the dataset pointclouds in *vertex maps* of dimension 3x64x840)

> /!\ Note that the dataset config files will require some environment variables to be set which points to the root folder of each datasets.
> If error arise, make sure that the environment variables (mentioned below) have been properly set.


#### Programmatically
> The configurations can also be defined programmatically (*ie* not using the scripts `run.py` not `train.py`).
>Each dataset defines a [`DatasetConfig`](../slam/dataset/dataset.py) is the (abstract) configuration dataclass matching a hydra's config file. 
>And a [`DatasetLoader`](../slam/dataset/dataset.py) which can load each sequence defined in the dataset as a `torch.utils.data.Dataset`.
> 

##  <a name="kitti">KITTI's Odometry Benchmark</a>

The odometry benchmark of KITTI's dataset (follow [link](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)) consists of 
21 sequences, 11 of which have the ground truth poses (for more than 20k lidar frames). 
The LiDAR sensor capturing the environment is a rotating 64 LiDAR Channel.

> /!\ The frame are corrected using pose obtained from a Visual-SLAM, and is thus not the raw data obtained from the sensor. 

### Installation Instructions

 - Download the lidar frames of the odometry benchmark (at this [download link](http://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip) (~80GB))
 - Download the ground truth poses of the odometry benchmark (at this [download link](http://www.cvlibs.net/download.php?file=data_odometry_poses.zip))
 - Extract both archives in the same root directory
 
The layout of KITTI's data on disk must be as follows:
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
#### Configuration setup

Before calling `run.py` or `train.py`, the following environment variables must be set:
> `KITTI_ODOM_ROOT`: points to the root dataset as defined above.

For example, to run the ICP odometry on KITTI:
```bash
# Set up the following required environment variables
JOB_NAME=kitti_F2M                                          
DATASET=KITTI                                              
KITTI_ODOM_ROOT=<path-to-kitti-odometry-root-directory>     

# Run the script
python run.py dataset=kitti
```

#### See [kitti_dataset.py](../slam/dataset/kitti_dataset.py) for more details.


## <a name="ford_campus">Ford Campus</a>

The Ford Campus Dataset consists of 2 large sequences of LiDAR data acquired with a HDL-64 with GPS ground truth (see the page [Ford Campus](http://robots.engin.umich.edu/SoftwareData/Ford) for more details). 

### Installation Instructions

 - Download the two sequences of the dataset: [dataset-1](http://robots.engin.umich.edu/uploads/SoftwareData/Ford/dataset-1.tar.gz) ~78GB, [dataset-2](http://robots.engin.umich.edu/uploads/SoftwareData/Ford/dataset-2.tar.gz) ~119GB
 - Alternatively download the sampled data of the [dataset-1](http://robots.engin.umich.edu/uploads/SoftwareData/Ford/dataset-1-subset.tgz) ~6GB
 - Extract both archives in the same root directory

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

#### Configuration setup
Before calling `run.py` or `train.py`, the following environment variables must be set:
- `FORD_ROOT`: points to the root dataset as defined above.

Example running the ICP odometry on Ford Campus:

```bash
# Set up the following required environment variables
JOB_NAME=ford_campus_F2M   # required by run.py to define the output path                                       
DATASET=FORD_CAMPUS        # required by run.py to define the output path                                     
FORD_ROOT=<path-to-ford-campus-root-directory>     

# Run the script
python run.py dataset=ford_campus
```
#### See [ford_dataset.py](../slam/dataset/ford_dataset.py) for more details.


## <a name="nclt">NCLT</a>

NCLT contains multiple sequences of the same environment captured with multiple sensors including a HDL-32 mounted on a segway and accurate GPS groundtruth.

#### Installation instructions
The data for each sequences can be downloaded on the dataset main page [NCLT](http://robots.engin.umich.edu/nclt/).

 - Download the sequences you are interested in (column `Velodyne`)
 - For each sequence downloaded, download the associated ground truth (column `Ground Truth Pose`)
 - Extract from the archives of each sequence the files with the following layout:
 
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

#### Configuration setup

Before calling `run.py` or `train.py`, the following environment variables must be set:
- `NCLT_ROOT`: points to the root dataset as defined above.

Example running the ICP odometry on NCLT:
```bash
# Set up the following required environment variables
JOB_NAME=nclt_F2M                                          
DATASET=NCLT                                              
NCLT_ROOT=<path-to-ford-campus-root-directory>     

# Run the script
python run.py dataset=nclt
```
#### See [nclt_dataset.py](../slam/dataset/nclt_dataset.py) for more details.