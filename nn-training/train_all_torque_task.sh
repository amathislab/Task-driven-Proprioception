#!/bin/bash

echo "Training spatial temporal"

#exp id 8015
python3.6 train_all_torque_task.py --type conv --arch_type spatial_temporal --exp_id 415 --proj_size 128 --n_layersfc 1 --start_id 0 --end_id 50 
python3.6 train_all_torque_task.py --type conv --arch_type temporal_spatial --exp_id 415 --proj_size 128 --n_layersfc 1 --start_id 0 --end_id 50 
python3.6 train_all_torque_task.py --type conv --arch_type spatiotemporal --exp_id 415 --proj_size 128 --n_layersfc 1 --start_id 0 --end_id 50 


#### CONV_NEW

echo "Training spatial temporal"

#exp id 8030
python3.6 train_all_torque_task.py --type conv_new --arch_type spatial_temporal --exp_id 430 --proj_size 128 --n_layersfc 1 --start_id 0 --end_id 50 
python3.6 train_all_torque_task.py --type conv_new --arch_type temporal_spatial --exp_id 430 --proj_size 128 --n_layersfc 1 --start_id 0 --end_id 50 
python3.6 train_all_torque_task.py --type conv_new --arch_type spatiotemporal --exp_id 430 --proj_size 128 --n_layersfc 1 --start_id 0 --end_id 50 