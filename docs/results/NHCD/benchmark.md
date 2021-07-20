## Newer Handheld College Dataset Benchmark:




#### Sorted trajectory error on all sequences:
| **Sequence Folder**|01_short_experiment | 02_long_experiment  |  AVG  | AVG Time (ms) |
| ---: | ---: | ---: | ---: | ---: |
| [EI-KdF2M](/home/pdell/dev/pylidar-slam/docs/results/NHCD/.results/EI-KdF2M) | 1.5249 |  |  | -1.000 |


#### Command Lines for each entry
| **Sequence Folder** | Command Line | git hash |
| ---: | ---: |  ---: |
| [EI-KdF2M](/home/pdell/dev/pylidar-slam/docs/results/NHCD/.results/EI-KdF2M) |  `python run.py slam/odometry/local_map=kdtree slam/odometry/initialization=EI slam/odometry/alignment=point_to_plane_GN dataset=nhcd +dataset.train_sequences=[01_short_experiment] device=cpu slam.odometry.local_map.local_map_size=30 slam.odometry.max_num_alignments=20 slam.odometry.alignment.gauss_newton_config.scheme=neighborhood slam.odometry.alignment.gauss_newton_config.sigma=0.2 num_workers=1 slam/odometry/preprocessing=grid_sample slam.odometry.preprocessing.filters.1.voxel_size=0.4 slam.odometry.data_key=input_data +slam.odometry.viz_debug=False`   | da0e571|
