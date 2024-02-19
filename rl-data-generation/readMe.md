**This folder contains code to convert RL pybullet data to muscle spindle.**

Code to simulate muscle spindle inputs from behavioral markers:
* `1_generate_spindle_from_rl.sh` is the bash script to convert joint angles to muscles and it uses `get_muscle_info_from_rl.py`.
* `2_create_rl_dataset.sh` is the bash script to generate the train/val/test dataset for the joint torque regression task and it uses `generate_torque_dataset_pool.py`.