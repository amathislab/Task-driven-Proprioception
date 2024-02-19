#!/bin/bash

echo "Running monkey datasets to get activations from experiment ID"

PATH_TO_ACTIVATIONS=$(python3.6 -c "from path_utils import PATH_TO_ACTIVATIONS; print(PATH_TO_ACTIVATIONS)")

OLDIFS=$IFS; IFS=',';
for data in Snap,20190829 Butter,20180326 Lando,20170917 Han,20171122 Chips,20170913 S1Lando,20170917; do set -- $data;

  echo "-- Current monkey/session data: "$1 and $2;
  for id in 15
    do
    echo "Experiment" $id
    python3.6 generate_session_activations_passive.py --monkey $1 --session $2 --exp $id \
        --active_start "mvt" --active_length 0 --train_iter 0 \
        --align 100 --permut_t --permut_m  --constant_input --n_pca 75 --path_to_act "$PATH_TO_ACTIVATIONS/passive/"
  done
done
IFS=$OLDIFS
