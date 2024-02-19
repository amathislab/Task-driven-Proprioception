#!/bin/bash

##### Bash script to generate the synthetic muscle spindle dataset from rl #######

## The final dataset to train neural networks is already generated and it is present in rl_data
## Once you already generated the batch of converted data, you can create the dataset for training neural networks models on the torque task

## To run this, you need to have already converted all the rl unprocessed data

echo "Generating rl-dataset from muscle spindle inputs"

python generate_torque_dataset_pool.py