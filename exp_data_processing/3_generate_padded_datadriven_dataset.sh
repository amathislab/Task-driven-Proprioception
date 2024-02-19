#!/bin/bash

echo "Making experimental monkey datasets for spike + end-effector regression network models"
echo "Activating conda env DeepProprio"
eval "$(conda shell.bash hook)"
conda activate DeepProprio

OLDIFS=$IFS; IFS=',';
for data in Snap,20190829 Lando,20170917 Butter,20180326 Han,20171122 Chips,20170913 S1Lando,20170917;
  do set -- $data;
  echo "-- Current monkey/session data: "$1 and $2;
  python3.8 generate_monkey_dataset_spikes.py --monkey $1 --session $2
done
IFS=$OLDIFS