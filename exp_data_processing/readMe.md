**This folder contains code to process the behavioral data.**

Code to simulate muscle spindle inputs from behavioral markers (It has to be run in the docker container):
* `1_run_ik_from_markers.sh` bash script to perform the inverse kinematics from the markers data and to retrieve joint angle information, it uses `ik_behavioral_exp.py`. 
* `2_extract_muscle_from_joints.sh` bash script to extract muscle length and muscle velocity from the joint angle position, it uses `get_muscle_info_exp_pool.py`.

Code to generate dataset to train data-driven neural networks (It needs the DeepProprio conda environment):
* `3_generate_padded_datadriven_dataset.sh` bash script to generate the train/val/test output dataset (muscle spindles and firing rate) per each NHP, it uses `generate_monkey_dataset_spikes.py`. It also saves the index to split train/test/val for predictions and it saves the index of trials to be removed.

Code to generate padded dataset to extract activations from trained neural networks using behavioral data (It needs the DeepProprio conda environment):
* `4_generate_padded_datasets_monkey.sh` bash script to generate the muscle kinematic and neural spike analysis datasets used throughout the analysis per each NHP, it uses `generate_padded_datasets_monkey.py`.