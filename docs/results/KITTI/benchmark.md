## KITTI Benchmark:




#### Sorted trajectory error on all sequences:
| **Sequence Folder**|00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10  |  AVG  | AVG Time (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [CV+KdF2M](/home/pdell/dev/pylidar-slam/docs/results/KITTI/.results/CV+KdF2M) | 0.5111 | 0.7924 | 0.5052 | 0.6422 | 0.3634 | 0.2874 | 0.2873 | 0.3214 | 0.7782 | 0.4569 | 0.5712 | 0.5333 | 174.792 |
| [EI+KdF2M](/home/pdell/dev/pylidar-slam/docs/results/KITTI/.results/EI+KdF2M) | 0.5286 | 0.7978 | 0.5236 | 0.6940 | 0.4491 | 0.3367 | 0.3074 | 0.3544 | 0.7977 | 0.5441 | 0.5055 | 0.5578 | 418.414 |
| [EI+PF2M](/home/pdell/dev/pylidar-slam/docs/results/KITTI/.results/EI+PF2M) | 0.5731 | 0.6793 | 0.6297 | 0.8308 | 0.5058 | 0.3726 | 0.3637 | 0.3243 | 0.9304 | 0.6049 | 1.0158 | 0.6412 | 136.496 |
| [CV+PF2M](/home/pdell/dev/pylidar-slam/docs/results/KITTI/.results/CV+PF2M) | 0.5722 | 0.7104 | 0.6292 | 0.8203 | 1.1266 | 0.3675 | 0.3638 | 0.3254 | 0.9275 | 0.6055 | 1.0132 | 0.6428 | 116.620 |


#### Command Lines for each entry
| **Sequence Folder** | Command Line | git hash |
| ---: | ---: |  ---: |
| [CV+KdF2M](/home/pdell/dev/pylidar-slam/docs/results/KITTI/.results/CV+KdF2M) |  `python run.py slam/odometry/local_map=kdtree slam/odometry/initialization=CV slam/odometry/alignment=point_to_plane_GN dataset=kitti +dataset.train_sequences=[00,01,02,03,04,05,06,'07','08','09','10'] device=cpu slam.odometry.local_map.local_map_size=30 slam.odometry.max_num_alignments=20 slam.odometry.alignment.gauss_newton_config.scheme=neighborhood slam.odometry.alignment.gauss_newton_config.sigma=0.2 num_workers=1 slam/odometry/preprocessing=grid_sample slam.odometry.preprocessing.filters.1.voxel_size=0.4 slam.odometry.data_key=input_data +slam.odometry.viz_debug=False`   | d02c7b3|
| [EI+KdF2M](/home/pdell/dev/pylidar-slam/docs/results/KITTI/.results/EI+KdF2M) |  `python run.py slam/odometry/local_map=kdtree slam/odometry/initialization=EI slam/odometry/alignment=point_to_plane_GN dataset=kitti +dataset.train_sequences=[00,01,02,03,04,05,06,'07','08','09','10'] device=cpu slam.odometry.local_map.local_map_size=30 slam.odometry.max_num_alignments=20 slam.odometry.alignment.gauss_newton_config.scheme=geman_mcclure slam.odometry.alignment.gauss_newton_config.sigma=0.1 num_workers=2 slam/odometry/preprocessing/filters=voxelization slam.odometry.preprocessing.filters.1.voxel_size=0.4 slam.odometry.data_key=input_data +slam.odometry.viz_debug=False`   | fbf9933|
| [EI+PF2M](/home/pdell/dev/pylidar-slam/docs/results/KITTI/.results/EI+PF2M) |  `python run.py slam/odometry/local_map=projective slam/odometry/initialization=EI slam/odometry/alignment=point_to_plane_GN dataset=kitti +dataset.train_sequences=[04,00,01,02,03,05,06,'07','08','09','10'] device=cuda:0 num_workers=4 slam.odometry.data_key=vertex_map slam.odometry.local_map.local_map_size=20 slam.odometry.max_num_alignments=15 slam.odometry.alignment.gauss_newton_config.scheme=neighborhood slam.odometry.alignment.gauss_newton_config.sigma=0.2 +slam.odometry.viz_debug=False`   | 64181ae|
| [CV+PF2M](/home/pdell/dev/pylidar-slam/docs/results/KITTI/.results/CV+PF2M) |  `python run.py slam/odometry/local_map=projective slam/odometry/initialization=CV slam/odometry/alignment=point_to_plane_GN dataset=kitti +dataset.train_sequences=[00,01,02,03,04,05,06,'07','08','09','10'] device=cuda:0 slam.odometry.data_key=vertex_map slam.odometry.local_map.local_map_size=20 slam.odometry.max_num_alignments=15 slam.odometry.alignment.gauss_newton_config.scheme=neighborhood slam.odometry.alignment.gauss_newton_config.sigma=0.2 +slam.odometry.viz_debug=False`   | d02c7b3|
