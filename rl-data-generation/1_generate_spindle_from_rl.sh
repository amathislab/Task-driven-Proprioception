#!/bin/bash

##### Bash script to generate synthetic spindle inputs from joint angles #######

## The final dataset to train neural networks is already generated and it is present in rl_data
## If you want to use your data, you need 4 joint angles (shoulder_rotation_z, shoulder_rotation_x, shoulder_rotation_y, and elbow_rotation)

echo "Converting rl joint angles to muscle spindles"

python3.6 get_muscle_info_from_rl.py --batch_num 1
# python3.6 get_muscle_info_from_rl.py --batch_num 2
# python3.6 get_muscle_info_from_rl.py --batch_num 3
# python3.6 get_muscle_info_from_rl.py --batch_num 4
# python3.6 get_muscle_info_from_rl.py --batch_num 5