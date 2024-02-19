import opensim as osim
from osim_utils import OsimModel, readMotionFile, make_muscle_config, create_states_from_sto
import numpy as np
import scipy.io as sio
import pickle
import os
import argparse
import copy
from multiprocessing import Pool
import sys

sys.path.append('../code/')
from path_utils import PATH_MONKEY_PROCESSED_DATAFRAMES, PATH_MONKEY_PROCESSED_DICT
PATH_TO_MONKEY_ARM = '../PCR-data-generation/all_monkey_arm'
def get_muscle_info_pool(inputs):

    # model_path, input_DIR, nn = inputs
    model_path, input_file = inputs

    time_step = 0.01
    mymodel = osim.Model(model_path) 

    header, labels, data = readMotionFile(input_file)
    
    results = {}

    ## Index of coordinate in the labels of the file
    coord_order = ['shoulder_flexion', 'shoulder_adduction', 'shoulder_rotation', 'elbow_flexion']
    index_list = []
    for ii in range(len(coord_order)):
        index_list.append(labels.index(coord_order[ii]))

    ## Get joint angles
    joint_traj = np.stack(data)[:,index_list].T

    ## Get muscle length, end-effector position and marker6
    musclegth_configurations, marker3, marker6 = make_muscle_config(mymodel, joint_traj)


    ## Get muscle velocity
    muscvel_configurations = np.float32(np.gradient(musclegth_configurations,time_step,axis=1))

    ## Get end-effector position

    ## Store in dictionary
    results['muscle_len'] = musclegth_configurations
    results['muscle_vel'] = muscvel_configurations
    results['joint_coords'] = joint_traj
    results['endeffector_coords'] = marker3
    results['marker6'] = marker6
    return results

def main(args): #args
    """Simulate muscle spindle from joint angle information

    Args:
        input_DIR (str): Path to the computed joint angles (.sto format)
        old_dataset_path (str): Path of the  monkey dataset
        model_path (str): Path to the opensim model
        save_path (str): Path where to save the new dataset 
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

    sto_DIR = os.path.join(PATH_MONKEY_PROCESSED_DATAFRAMES,file_name_prefix+'_sto')

    ## Path to IK results
    if not os.path.exists(PATH_MONKEY_PROCESSED_DICT):
        os.makedirs(PATH_MONKEY_PROCESSED_DICT)
    save_DIR = os.path.join(PATH_MONKEY_PROCESSED_DICT,file_name_prefix+'_mod.pkl')

    n_files = len([name for name in os.listdir(sto_DIR) if os.path.isfile(os.path.join(sto_DIR, name))])

    data_dic = {}

    all_files = [[model_path, sto_DIR + '/trial_' + str(ii) + '.sto'] for ii in range(n_files)]
    
    p = Pool()
    
    result = p.map(get_muscle_info_pool, all_files)
    p.close()
    p.join()

    for ii in range(n_files):
        name_file = 'trial_' + str(ii)
        data_dic[name_file] = copy.copy(result[ii])

    a_file = open(save_DIR, "wb")
    pickle.dump(data_dic, a_file)
    a_file.close()
    return

    


if __name__=='__main__':
    # main()
    parser = argparse.ArgumentParser(description='Generating muscle length and velocities from IK.')
    parser.add_argument('--monkey_name', type=str, help='Select the monkey (snap,butter,lando,han,chips,s1lando)', default = 'snap')
    main(parser.parse_args())