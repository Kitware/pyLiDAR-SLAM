## KITTI Benchmark:


#### Sorted trajectory error on all sequences:
| **Sequence Folder**|00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10  |  AVG  | AVG Time (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [CV+KdF2M](/home/pdell/dev/pylidar-slam/docs/results/KITTI/CV+KdF2M) | 0.6003 | 0.8131 | 0.6365 | 0.7261 | 0.6152 | 0.3826 | 0.3352 | 0.4012 | 0.8244 | 0.5792 | 0.5252 | 0.6182 | 357.587 |
| [CV+PF2M](/home/pdell/dev/pylidar-slam/docs/results/KITTI/CV+PF2M) | 0.8758 | 1.1384 | 0.9884 | 1.4319 | 0.8869 | 0.7648 | 0.5922 | 0.6415 | 1.5061 | 1.1210 | 1.6449 | 1.0541 | 92.755 |
#### Command Lines for each entry
| **Sequence Folder** | Command Line | git hash |
| ---: | ---: |  ---: |
| [CV+KdF2M](/home/pdell/dev/pylidar-slam/docs/results/KITTI/CV+KdF2M) |  `python run.py slam/odometry/local_map=kdtree slam/odometry/initialization=CV slam/odometry/alignment=point_to_plane_GN dataset=kitti +dataset.train_sequences=[00,01,02,03,04,05,06,'07','08','09','10'] device=cpu slam.odometry.data_key=numpy_pc slam.odometry.local_map.local_map_size=30 slam.odometry.max_num_alignments=100 +slam.odometry.viz_num_pcs=20 slam/odometry/preprocessing/filters=voxelization slam.odometry.data_key=input_data`   | 66471ae|
| [CV+PF2M](/home/pdell/dev/pylidar-slam/docs/results/KITTI/CV+PF2M) |  `python run.py slam/odometry/local_map=projective slam/odometry/initialization=CV slam/odometry/alignment=point_to_plane_GN dataset=kitti +dataset.train_sequences=[00,01,02,03,04,05,06,'07','08','09','10'] device=cuda:0 slam.odometry.data_key=vertex_map slam.odometry.local_map.local_map_size=10 slam.odometry.max_num_alignments=10 slam.odometry.alignment.gauss_newton_config.scheme=geman_mcclure slam.odometry.alignment.gauss_newton_config.sigma=0.1`   | d933c61|
