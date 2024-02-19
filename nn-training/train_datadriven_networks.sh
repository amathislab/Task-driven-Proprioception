#!/bin/bash

echo "Training spikes conv"

OLDIFS=$IFS; IFS=',';
for data in Snap,20190829,13515 Butter,20180326,13615 Lando,20170917,13715 Han,20171122,13815 Chips,20170913,13915 S1Lando,20170917,13415; do set -- $data;

  echo "-- Current monkey/session data: "$1 and $2;
  python3.6 train_datadriven_networks.py --lr 0.001 --monkey $1 --session $2 --type conv --arch_type spatial_temporal --exp_id $3 --start_id 0 --end_id 50
  python3.6 train_datadriven_networks.py --lr 0.001 --monkey $1 --session $2 --type conv --arch_type temporal_spatial --exp_id $3 --start_id 0 --end_id 50
  python3.6 train_datadriven_networks.py --lr 0.001 --monkey $1 --session $2 --type conv --arch_type spatiotemporal --exp_id $3 --start_id 0 --end_id 50
  
  done

IFS=$OLDIFS