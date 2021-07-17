## Towards reproducible research...

> pyLIDAR-SLAM is a research project. One of its goal is to compare and benchmark different algorithms on a variety of datasets. 

> In `build_benchmark_md.py` we provide a tool to aggregate the results obtained after different runs of the algorithm on a given dataset (*For now only KITTI is supported, this will change in the future*).
> Hydra creates a lot of directories, and it is rather cumbersome to search for results by hand. The script thus searches recursively for all directories having both ground truth and estimated poses, starting at a given root.
> For each found entry, it computes the metrics (which can take a while depending on the number of results to aggregate), and generates a `benchmark.md` file similar to [benchmark.md](results/KITTI/benchmark.md)

> For reproducibility benchmark.md will save the *git_hash* and the *command_line* used to generate the results.
> Note that the command line might not be valid at any given commit, but using the *git_hash* you should be able to retrieve the same execution without (too much) complications. 

> You should be able to reproduce the results without too much difficulty. Note that *hydra* provides useful tool to this end, notably the *multi-run* mechanism.
> To launch multiple runs (with different set of parameters simply add `-m` and the different parameters desired). For example:
```bash
export JOB_NAME=kitti_F2M                                          # The folder to log hydra output files
export DATASET=KITTI                                               # Name of the Dataset to construct the destination folder 
export KITTI_ODOM_ROOT=<path-to-kitti-odometry-root-directory>     # The path to KITTI odometry benchmark files

# Run the script
python run.py -m dataset=kitti num_workers=4 device=cuda:0 slam/odometry/local_map=projective \
    slam/odometry/local_map=projective \
    slam/odometry/initialization=CV \
    slam/odometry/alignment=point_to_plane_GN \
    dataset=kitti \
    +dataset.train_sequences=["00","01","02","03","04","05","06",'07','08','09','10'] \
    device=cuda:0 \
    slam.odometry.data_key=vertex_map \
    slam.odometry.local_map.local_map_size=10,20,30 \ # Different local map sizes
    slam.odometry.max_num_alignments=10,15,20 \ # Different number of alignments
    slam.odometry.alignment.gauss_newton_config.scheme=geman_mcclure,neighborhood,cauchy \ # Different loss functions
    slam.odometry.alignment.gauss_newton_config.sigma=0.1,0.2,0.5 \   # Range of loss function arguments
    +slam.odometry.viz_debug=False 
```

> The above command will iteratively launch the 3 * 3 * 3 * 3 = 81 different executions on the grid of parameters defined.
> This will generate a lot of files and directories. So you can use `build_benchmark_md.py` to aggregate results.
### KITTI

> A benchmark of the methods implemented in **pyLIDAR-SLAM** evaluated on *KITTI* can be found at [benchmark.md](results/KITTI/benchmark.md).


#### TODOs
- [ ] NCLT
- [ ] FORD CAMPUS
- [ ] New Handheld College Dataset