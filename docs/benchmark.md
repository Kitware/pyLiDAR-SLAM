# Results on KITTI Dataset 

| **Ids** | 00   |  01  | 02 | 03 | 04  | 05  | 06 | 07 | 08 | 09 | 10 | AVG | Average Time (ms) |
| --- | ---: | ---: |---:|---:| ---: | ---: |---:|---:|---:|---:|---:|---:|---:|
| [2021-05-06_12-31-06](/home/pdell/dev/pylidar-slam/.outputs/slam/KITTI/slam_loop_closure/2021-05-06_12-31-06) | 0.56   |  0.32  | 02 | 03 | 04   |  05  | 06 | 07 | 08 | 09 | 10 | 1 | 

### Command lines

| **Ids** | **Command Line** |
| --- | --- |
| [2021-05-06_12-31-06](/home/pdell/dev/pylidar-slam/.outputs/slam/KITTI/slam_loop_closure/2021-05-06_12-31-06) |  ```python run.py slam/odometry/local_map=kdtree slam/odometry/initialization=CV  slam/odometry/alignment=point_to_plane_GN dataset=kitti +dataset.train_sequences=[00,01,02,03,04,05,06,'07','08','09','10'] device=cpu  slam.odometry.data_key=numpy_pc slam.odometry.local_map.local_map_size=30``` |
