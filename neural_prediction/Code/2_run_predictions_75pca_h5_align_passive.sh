#!/bin/bash

echo "(Activating conda env DeepProprio)"
eval "$(conda shell.bash hook)"
conda activate DeepProprio
echo "NEURAL PREDICTIONS ON MONKEY DATASETS"

PATH_TO_ACTIVATIONS=$(python3.6 -c "from path_utils import PATH_TO_ACTIVATIONS;")
PATH_TO_PREDICTIONS=$(python3.6 -c "from path_utils import PATH_TO_PREDICTIONS;")

OLDIFS=$IFS; IFS=',';
for data in Snap,20190829 Butter,20180326 Lando,20170917 Han,20171122 Chips,20170913 S1Lando,20170917; do set -- $data;
  echo "-- Current monkey/session data: "$1 and $2;
  for id in 5015 # {5015,8030}
    do
    echo "Experiment" $id
    python3.8 compute_session_predictivity_h5_cv_pool_align_passive.py --monkey $1 --session $2 --exp $id --active_start "mvt" --active_length 0 \
     --align 100 --permut_t --permut_m  --constant_input --path_to_act "$PATH_TO_ACTIVATIONS/passive/" --path_to_results "$PATH_TO_PREDICTIONS/passive/"
  done
done
IFS=$OLDIFS
IFS=$OLDIFS
