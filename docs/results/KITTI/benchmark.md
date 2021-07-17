## KITTI Benchmark:




#### Sorted trajectory error on all sequences:
| **Sequence Folder**|00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10  |  AVG  | AVG Time (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [EI+KdF2M](/home/pdell/dev/pylidar-slam/docs/results/KITTI/.results/EI+KdF2M) | 0.5286 | 0.7978 | 0.5236 | 0.6940 | 0.4491 | 0.3367 | 0.3074 | 0.3544 | 0.7977 | 0.5441 | 0.5055 | 0.5578 | 418.414 |
| [CV+KdF2M](/home/pdell/dev/pylidar-slam/docs/results/KITTI/.results/CV+KdF2M) | 0.5407 | 0.7453 | 0.5479 | 0.7108 | 0.4139 | 0.3289 | 0.3125 | 0.3462 | 0.8036 | 0.5224 | 0.5368 | 0.5637 | 634.990 |
| [CV+PF2M](/home/pdell/dev/pylidar-slam/docs/results/KITTI/.results/CV+PF2M) | 0.8758 | 1.1384 | 0.9884 | 1.4319 | 0.8869 | 0.7648 | 0.5922 | 0.6415 | 1.5061 | 1.1210 | 1.6449 | 1.0541 | 92.755 |


#### Command Lines for each entry
| **Sequence Folder** | Command Line | git hash |
| ---: | ---: |  ---: |
| [EI+KdF2M](/home/pdell/dev/pylidar-slam/docs/results/KITTI/.results/EI+KdF2M) |  `python run.py slam/odometry/local_map=kdtree slam/odometry/initialization=EI slam/odometry/alignment=point_to_plane_GN dataset=kitti +dataset.train_sequences=[00,01,02,03,04,05,06,'07','08','09','10'] device=cpu slam.odometry.local_map.local_map_size=30 slam.odometry.max_num_alignments=20 slam.odometry.alignment.gauss_newton_config.scheme=geman_mcclure slam.odometry.alignment.gauss_newton_config.sigma=0.1 num_workers=2 slam/odometry/preprocessing/filters=voxelization slam.odometry.preprocessing.filters.1.voxel_size=0.4 slam.odometry.data_key=input_data +slam.odometry.viz_debug=False`   | fbf9933|
| [CV+KdF2M](/home/pdell/dev/pylidar-slam/docs/results/KITTI/.results/CV+KdF2M) |  `python run.py slam/odometry/local_map=kdtree slam/odometry/initialization=CV slam/odometry/alignment=point_to_plane_GN dataset=kitti +dataset.train_sequences=[00,01,02,03,04,05,06,'07','08','09','10'] device=cpu slam.odometry.local_map.local_map_size=30 slam.odometry.max_num_alignments=20 slam.odometry.alignment.gauss_newton_config.scheme=geman_mcclure slam.odometry.alignment.gauss_newton_config.sigma=0.1 num_workers=2 slam/odometry/preprocessing/filters=voxelization slam.odometry.preprocessing.filters.1.voxel_size=0.2 slam.odometry.data_key=input_data`   | d933c61|
| [CV+PF2M](/home/pdell/dev/pylidar-slam/docs/results/KITTI/.results/CV+PF2M) |  `python run.py slam/odometry/local_map=projective slam/odometry/initialization=CV slam/odometry/alignment=point_to_plane_GN dataset=kitti +dataset.train_sequences=[00,01,02,03,04,05,06,'07','08','09','10'] device=cuda:0 slam.odometry.data_key=vertex_map slam.odometry.local_map.local_map_size=10 slam.odometry.max_num_alignments=10 slam.odometry.alignment.gauss_newton_config.scheme=geman_mcclure slam.odometry.alignment.gauss_newton_config.sigma=0.1`   | d933c61|
