import opensim as osim
import sys
sys.path.append('../exp_data_processing')
sys.path.append('../code/')
sys.path.append('../PCR-data-generation/')

from osim_utils import make_muscle_config
import numpy as np
import pickle
import os
import argparse
import copy
from multiprocessing import Pool
import pandas as pd
from path_utils import PATH_TO_UNPROCESSED_RL_DATA, PATH_TO_CONVERTED_RL_DATA

PATH_TO_MONKEY_ARM = './../PCR-data-generation/all_monkey_arm'

def get_muscle_info_rl_pool(inputs):
    """ Get muscle length and velocity given joint angles stored in files. Parallel implementation.

    This function takes the paths to an OpenSim model and an input file containing joint angles. It reads the joint angles
    from the input file, converts them to degrees, and uses an OpenSim model to compute muscle length and velocity, as well
    as end-effector position and marker coordinates. The computation is performed for each episode in the input data.

    Args:
        inputs (tuple): A tuple containing two elements:
            - model_path (str): The path to the OpenSim model.
            - input_file (str): The path to the input file containing joint angles.

    Returns:
        dict: A dictionary containing the results for each episode. The keys are episode indices, and the values are
        dictionaries with the following information:
            - 'muscle_len' (numpy.ndarray): Muscle length configurations over time.
            - 'muscle_vel' (numpy.ndarray): Muscle velocity configurations over time.
            - 'joint_coords' (numpy.ndarray): Joint angles in degrees over time.
            - 'endeffector_coords' (numpy.ndarray): End-effector position coordinates over time.
            - 'marker6' (numpy.ndarray): Marker coordinates for marker6 over time.
            - 'name_file' (str): The path to the input file.
            - 'ep_idx' (int): The episode index.

    Note:
        This function assumes that the input file contains columns named 'shoulder_rotation_z', 'shoulder_rotation_x',
        'shoulder_rotation_y', and 'elbow_rotation', representing joint angles in radians.

    Example:
        model_path = 'path/to/opensim/model.osim'
        input_file = 'path/to/input/data.csv'
        inputs = (model_path, input_file)
        results = get_muscle_info_rl_pool(inputs)
    """
    model_path, input_file = inputs
    data_all  = pd.read_csv(input_file,index_col=[0])   
    results = {}

    ## Initialize opensim model
    time_step = 0.01
    mymodel = osim.Model(model_path)

    for ep_idx in np.unique(data_all.episode_id):
        data = data_all[data_all.episode_id == ep_idx]
        results[str(ep_idx)] = {}

        ### Get joints and convert in degrees
        joint_angle_traj = np.zeros((4,200))
        joint_angle_traj[0,:] = (data['shoulder_rotation_z']/np.pi)*180
        joint_angle_traj[1,:] = (data['shoulder_rotation_x']/np.pi)*180
        joint_angle_traj[2,:] = (data['shoulder_rotation_y']/np.pi)*180
        joint_angle_traj[3,:] = (data['elbow_rotation']/np.pi)*180

        ## Get muscle length, end-effector position and marker6
        musclegth_configurations, marker3, marker6 = make_muscle_config(mymodel, joint_angle_traj)

        ## Get muscle velocity
        muscvel_configurations = np.float32(np.gradient(musclegth_configurations,time_step,axis=1))

        ## Store in dictionary
        results[str(ep_idx)]['muscle_len'] = musclegth_configurations
        results[str(ep_idx)]['muscle_vel'] = muscvel_configurations
        results[str(ep_idx)]['joint_coords'] = joint_angle_traj
        results[str(ep_idx)]['endeffector_coords'] = marker3
        results[str(ep_idx)]['marker6'] = marker6
        results[str(ep_idx)]['name_file'] = input_file
        results[str(ep_idx)]['ep_idx'] = ep_idx

    return results


def main(args):

    model_path = os.path.join(PATH_TO_MONKEY_ARM,'Snap','Snap_scaled_fin1.osim')
    input_DIR = os.path.join(PATH_TO_UNPROCESSED_RL_DATA,'batch_'+str(args.batch_num))
    save_DIR = os.path.join(PATH_TO_CONVERTED_RL_DATA,'converted_rl_data_'+str(args.batch_num)+'.pkl')
    data_dic = {}

    file_list = os.listdir(input_DIR)
    all_files = [[model_path, os.path.join(input_DIR, name_file)] for name_file in file_list]
    n_files = len(file_list)

    p = Pool()
    
    result = p.map(get_muscle_info_rl_pool, all_files)
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
    parser = argparse.ArgumentParser(description='Generating muscle length and velocities from IK.')
    parser.add_argument('--batch_num', type=int, help='Number of the batch to use (from 1 to 5)', default = 1)
    main(parser.parse_args())