**This folder contains all code used to generate the synthetic proprioceptive dataset.**  

The folder contains a modified version of [Sandbrink et al.](https://elifesciences.org/articles/81499)

* To explore, please start from the `Figure_dataset_generation` notebook in the paper_figure folder which contains a walk-through of the entire data generation process, starting from pen-tip trajectories up to spindle firing rates.
* `uci_tracjectorydata.py` contains code to reformat the [ dataset by Williams et al.](https://archive.ics.uci.edu/ml/datasets/Character+Trajectories) into a suitable format for our processing. References are contained in this website and our paper.
* `1_generate_start_points.sh` is a bash script to generate reachable starting points. It uses `generate_startpoints_5.py` that computes the reachable starting points in the workspace of a simple 2-link arm model, defined in `pcr_data_utils.py`. Note: it uses a parallel implementation with joblib.
* `2_run_dataset_seed.sh` is a bash script to generate datapoints of all labels (horizontal and vertical) with random seed. In the bash file, you can change the start and end index of the folder name in which trajectories (from `0.p` to `40.p`) will be stored. It runs `generate_data_hor_seed.py` and `generate_data_ver_seed.py` which are scripts that generate datapoints of a given character label in a given plane (horizontal/vertical).
* `3_run_create_dataset_final_all.sh` is a bash script to generate the synthetic spindle dataset for a specific monkey which is then used for training TCNs. It uses `create_dataset_final_all.py` that uses unprocessed_data and split the data in train, val and test set. Trajectories are padded to a specific time length and shuffled.

In order to run the bash scripts and `Figure_dataset_generation` notebook, please make sure that you use the provided docker container, since it has opensim python API installed on it and that you adjusted the corresponding path in code/path_utils.py