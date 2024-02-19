#!/bin/bash

##### START POINTS are already present in the folder start_points #######

## To generate them again, use the following bash code

echo "Generating start_points"

python3.6 -m pip install joblib
python3.6 generate_startpoints5.py --monkey_name 'snap'