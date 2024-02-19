#!/bin/bash

##### Bash script to generate padded dataset to give as test input to frozed neural networks #######

echo "Activating conda env DeepProprio"
eval "$(conda shell.bash hook)"
conda activate DeepProprio
echo "Generate monkey session datasets - with different movement onset alignment in template input"

OLDIFS=$IFS; IFS=',';
for data in Snap,20190829 Butter,20180326 Lando,20170917 Han,20171122 Chips,20170913 S1Lando,20170917; do set -- $data;
  
  echo "Current monkey/session data: "$1  $2;
  python3.8 generate_padded_datasets_monkey.py --monkey $1 --session $2 --active_start 'mvt' --active_length 0 --align 100 \
                                                --permut_m --permut_t --constant_input

done
IFS=$OLDIFS