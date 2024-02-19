#!/bin/bash

######### Bash script to generate the pcr dataset of spindle inputs used to train TCNs.  ##########

### Change the init and end folder based on the index used for generating the unprocessed data.

echo "Generating trajectories for snap"

#### SNAP
python3.6 create_dataset_final_all.py --monkey_name 'snap' --lab_count_per_char 90 \
                                      --init_folder 100 --end_folder 181

## If necessary for other monkeys

# #### HAN 01_05
# python3.6 create_dataset_final_all.py --monkey_name 'han01_05' --lab_count_per_char 90 \
#                                       --init_folder 100 --end_folder 181

# echo "Generating trajectories han11_22"

# #### HAN 11__22
# python3.6 create_dataset_final_all.py --monkey_name 'han11_22' --lab_count_per_char 90 \
#                                       --init_folder 100 --end_folder 181

# echo "Generating trajectories chips"

# #### CHIPS
# python3.6 create_dataset_final_all.py --monkey_name 'chips' --lab_count_per_char 90 \
#                                       --init_folder 100 --end_folder 181

# echo "Generating trajectories lando"

# #### LANDO
# python3.6 create_dataset_final_all.py --monkey_name 'lando' --lab_count_per_char 90 \
#                                       --init_folder 100 --end_folder 181

# echo "Generating trajectories butter"

# #### BUTTER
# python3.6 create_dataset_final_all.py --monkey_name 'butter' --lab_count_per_char 90 \
#                                       --init_folder 100 --end_folder 181

