#!/bin/bash

echo "Activating conda env DeepProprio"
eval "$(conda shell.bash hook)"
conda activate DeepProprio
echo "Fit tuning curve linear and GLM for each monkey session."

OLDIFS=$IFS; IFS=',';
for data in Snap,20190829 Lando,20170917 Butter,20180326 S1Lando,20170917 Han,20171122 Chips,20170913;
  do set -- $data;
  echo "Tuning models for "$1 and $2;
  python3.8 fit_linear_models_cv.py --monkey $1 --session $2  --start 'mvt' --short 'n'
done
IFS=$OLDIFS