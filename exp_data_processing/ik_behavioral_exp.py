import opensim as osim
import numpy as np
import os, os.path
import scipy.io as sio
from osim_utils import *
import pickle as pickle
import argparse

import sys

sys.path.append('../code/')
from path_utils import PATH_MONKEY_PROCESSED_DATAFRAMES
PATH_TO_MONKEY_ARM = '../PCR-data-generation/all_monkey_arm'
def main(args):
    """Generate trc and sto files from markers.

    This function generates .trc and .sto files from marker data for a specific monkey. The generated files are based on
    the provided monkey name and, optionally, inverse kinematics (IK) data.

    Args:
        args (Namespace): An object containing command-line arguments. It should have the following attributes:
            - monkey_name (str): The name of the monkey, used to determine the file name prefix.
            - model_path (str): The path to the OpenSim model.
            - ik (bool): Flag indicating whether to generate .sto files from inverse kinematics data.

    Returns:
        None

    """
    
    monkey_name = args.monkey_name

    file_name_prefix = ''
    if monkey_name == 'snap':
        file_name_prefix = 'Snap_20190829'
        model_path = os.path.join(PATH_TO_MONKEY_ARM,'Snap','Snap_scaled_fin1.osim')
    elif monkey_name == 'butter':
        file_name_prefix = 'Butter_20180326'
        model_path = os.path.join(PATH_TO_MONKEY_ARM,'Butter','ButterScaledArm_ale.osim')
    elif monkey_name == 'lando':
        file_name_prefix = 'NewLando_20170917'
        model_path = os.path.join(PATH_TO_MONKEY_ARM,'Lando','LandoScaledArm_ale.osim')
    elif monkey_name == 'han':
        file_name_prefix = 'Han_20171122'
        model_path = os.path.join(PATH_TO_MONKEY_ARM,'Han','HanScaledArm20171122.osim')
    elif monkey_name == 'chips':
        file_name_prefix = 'Chips_20170913'
        model_path = os.path.join(PATH_TO_MONKEY_ARM,'Chips','ChipsScaledArm_ale.osim')
    elif monkey_name == 's1lando':
        file_name_prefix = 'S1Lando_20170917'
        model_path = os.path.join(PATH_TO_MONKEY_ARM,'Lando','LandoScaledArm_ale.osim') 

    ## Load data
    a_file = open(os.path.join(PATH_MONKEY_PROCESSED_DATAFRAMES,file_name_prefix+ '_TD_df.pkl'), "rb")
    old_data = pickle.load(a_file)
    a_file.close()

    markers_all = old_data['markers'].to_numpy()

    markers_names_all = old_data['marker_names'].to_numpy()[0]

    markers_names = []
    for ii in range(0,len(markers_names_all),3):
        markers_names.append(str(markers_names_all[ii][:-2]))

    time_step = 0.01
    trc_DIR = os.path.join(PATH_MONKEY_PROCESSED_DATAFRAMES,file_name_prefix+'_trc')
    sto_DIR = os.path.join(PATH_MONKEY_PROCESSED_DATAFRAMES,file_name_prefix+'_sto')

    ## Create .trc files from markers
    generate_trc_files(markers_all, markers_names, time_step, trc_DIR)

    if args.ik:
        ## Create .sto from IK
        generate_sto_files_pool(model_path, trc_DIR, sto_DIR)
    return

    


if __name__=='__main__':
    # main()
    parser = argparse.ArgumentParser(description='Running inverse kinematics for each behavioral experiment.')
    parser.add_argument('--monkey_name', type=str, help='Select the monkey (snap,butter,lando,han,chips,s1lando)', default = 'snap')
    # parser.add_argument('--model_path', type=str, help='Path of the model', default = '../PCR-data-generation/all_monkey_arm/Snap/Snap_scaled_fin1.osim')
    parser.add_argument('--ik', action='store_true', help='Flag to compute inverse kinematics (e.g. run ik if true)', default = False)
    main(parser.parse_args())