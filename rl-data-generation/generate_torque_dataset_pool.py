import os
import h5py

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import math as m
import argparse
from multiprocessing import Pool
import sys
sys.path.append('../code/')

from path_utils import PATH_TO_CONVERTED_RL_DATA, PATH_TO_SAVE_RL_DATASET

def compute_jerk(joint_trajectory):
    """Compute the jerk in joint space for the obtained joint configurations.

    Returns
    -------
    jerk : np.array, [T,] array of jerk for a given trajectory

    """
    joint_vel = np.gradient(joint_trajectory, axis=1)
    joint_acc = np.gradient(joint_vel, axis=1)
    joint_jerk = np.gradient(joint_acc, axis=1)
    jerk = np.linalg.norm(joint_jerk)
    return jerk

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def replace_nan_single(y):
    nans, x = nan_helper(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    return y

def replace_nan(mus_len_tmp):
    if np.any(np.isnan(mus_len_tmp)):
        idx_x, idx_y = np.where(np.isnan(mus_len_tmp))
        unique_x = np.unique(idx_x)

        for jj in range(len(unique_x)):
            mus_len_tmp[unique_x,:] = replace_nan_single(mus_len_tmp[unique_x,:])
    return mus_len_tmp

def start_end_choice(traj, max_len = 400):
    true_traj = traj[:, np.all(~np.isnan(traj), axis=0)]
    room = max_len - true_traj.shape[1]  #340
    start_idx = np.random.randint(room)
    end_idx = start_idx + true_traj.shape[1]
    return start_idx, end_idx

def apply_shuffle(traj, start_idx, end_idx, max_len = 400):
    true_traj = traj[:, np.all(~np.isnan(traj), axis=0)]
    mytraj = np.zeros((true_traj.shape[0], max_len))
    mytraj[:, start_idx:end_idx] = true_traj
    mytraj[:, :start_idx] = true_traj[:, 0][:, None]
    mytraj[:, end_idx:] = true_traj[:, -1][:, None]
    return mytraj

def apply_shuffle_pad(traj, start_idx, end_idx, pad_value=None, max_len = 400):
    true_traj = traj[:, np.all(~np.isnan(traj), axis=0)]
    mytraj = np.zeros((true_traj.shape[0], max_len))
    mytraj[:, start_idx:end_idx] = true_traj

    pad_value = pad_value * np.ones_like(true_traj[:, 0])
    mytraj[:, :start_idx] = pad_value[:, None]
    mytraj[:, end_idx:] = pad_value[:, None]
    return mytraj
 
def Rx(theta):
    return np.matrix([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])

R_rot = Rx(m.pi/2)

def pool_data(inputs):

    trial_data, latents = inputs
    t_size = 400

    ## Get information from recomputed muscle length and velocity
    mus_len_tmp = trial_data['muscle_len']
    mus_vel_tmp = trial_data['muscle_vel']
    endeff_tmp = trial_data['endeffector_coords']
    joint_coords_tmp = trial_data['joint_coords']

    path_file = trial_data['name_file']
    ep_idx_rl = trial_data['ep_idx']

    data_all  = pd.read_csv(path_file,index_col=[0])
    data = data_all[data_all.episode_id == ep_idx_rl]

    
    ### Get joints and convert in degrees
    torque_tmp = np.zeros((4,200))
    torque_tmp[0,:] = data['shoulder_torque_x']
    torque_tmp[1,:] = data['shoulder_torque_y']
    torque_tmp[2,:] = data['shoulder_torque_z']
    torque_tmp[3,:] = data['elbow_torque']

    target_tmp = np.zeros((3,200))
    target_tmp[0,:] = data['target_position_x']
    target_tmp[1,:] = data['target_position_y']
    target_tmp[2,:] = data['target_position_z']

    target_tmp_one = target_tmp[:,0]

    ## Replace nan from muscle in the trajectory with linear interpolation
    mus_len_tmp = replace_nan(mus_len_tmp)
    mus_vel_tmp = replace_nan(mus_vel_tmp)

    ## Compute muscle jerk
    mus_jerk_tmp = compute_jerk(mus_len_tmp)

    ## Take start and end index to pad
    start_idx, end_idx = start_end_choice(mus_len_tmp, t_size)

    ## Pad and shuffle initial position
    mus_len_tmp = apply_shuffle(mus_len_tmp, start_idx, end_idx)
    mus_vel_tmp = apply_shuffle(mus_vel_tmp, start_idx, end_idx)
    endeff_tmp = apply_shuffle(endeff_tmp, start_idx, end_idx)
    joint_coords_tmp = apply_shuffle(joint_coords_tmp, start_idx, end_idx)
    torque_tmp = apply_shuffle_pad(torque_tmp, start_idx, end_idx, pad_value = 0)
    target_tmp = apply_shuffle(target_tmp, start_idx, end_idx)

    mus_len_tmp1 = np.expand_dims(mus_len_tmp,-1) #[...,].shape)
    mus_vel_tmp1 = np.expand_dims(mus_vel_tmp,-1)
    spindle_input = np.concatenate((mus_len_tmp1, mus_vel_tmp1),axis=2)

    latents['start_idx'] = start_idx
    latents['end_idx'] = end_idx
    latents['mus_jerk'] = mus_jerk_tmp

    return spindle_input, endeff_tmp, joint_coords_tmp, torque_tmp, target_tmp, target_tmp[:,0], latents


def main(args):

    input_DIR =  PATH_TO_CONVERTED_RL_DATA 
    save_DIR = PATH_TO_SAVE_RL_DATASET
    monkey_name = args.monkey_name
    name_dataset = args.name_dataset
    data_dic = {}

    batch_list = os.listdir(input_DIR)
    n_files = len(batch_list)

    all_latents_train = []
    all_latents_val = []
    all_latents_test = []
    bad_trials = []

    t_size = 400
    count = 1

    for batch in batch_list:

        print('Batch: ', batch)
        ## Add the dataset
        batch_file = pickle.load(open(os.path.join(input_DIR,batch), 'rb'))

        for trial_idx in list(batch_file.keys()):

            print('Trial: ', trial_idx)
            latents = dict()

            trial_data = batch_file[trial_idx]
            muscle_len_all = []
            muscle_vel_all = []
            end_eff_all = []
            # too_long = []
            spindle_all = []
            joints_all = []
            torque_all = []
            target_all = []
            target_all_one = []
            
            latents_all = []
            for ep_idx in list(trial_data.keys()):
                latents['batch'] = batch
                latents['trial_idx'] = trial_idx
                latents['ep_idx'] = ep_idx
                latents['monkey_name'] = monkey_name
                latents_all.append(latents)

            all_data = [trial_data[ep_idx] for ep_idx in list(trial_data.keys())]

            all_inputs = zip(all_data, latents_all)

            p = Pool()

            spindle_all, end_eff_all, joints_all, torque_all, target_all, target_all_one, latents_all = zip(*p.map(pool_data, all_inputs))
            p.close()
            p.join()

            ############################ CREATE DATASET ######################################
            ind_all = np.arange(len(spindle_all))

            ## Get test indices
            ind_train, ind_test = train_test_split(ind_all, test_size=0.2)

            ## Get train and val indices
            ind_train, ind_val = train_test_split(ind_train, test_size=0.1)

            ind_train = ind_train.astype(int)
            ind_test = ind_test.astype(int)
            ind_val = ind_val.astype(int)

            spindle_info_train = np.array(spindle_all)[ind_train]
            spindle_info_test = np.array(spindle_all)[ind_test]
            spindle_info_val = np.array(spindle_all)[ind_val]

            mean_all = np.mean(np.array(spindle_info_train),0)

            ee_train = np.array(end_eff_all)[ind_train]
            ee_test = np.array(end_eff_all)[ind_test]
            ee_val = np.array(end_eff_all)[ind_val]

            joints_train = np.array(joints_all)[ind_train]
            joints_test = np.array(joints_all)[ind_test]
            joints_val = np.array(joints_all)[ind_val]

            torque_train = np.array(torque_all)[ind_train]
            torque_test = np.array(torque_all)[ind_test]
            torque_val = np.array(torque_all)[ind_val]

            target_train = np.array(target_all)[ind_train]
            target_test = np.array(target_all)[ind_test]
            target_val = np.array(target_all)[ind_val]

            target_one_train = np.array(target_all_one)[ind_train]
            target_one_test = np.array(target_all_one)[ind_test]
            target_one_val = np.array(target_all_one)[ind_val]

            latents_train = np.array(latents_all)[ind_train]
            latents_test = np.array(latents_all)[ind_test]
            latents_val = np.array(latents_all)[ind_val]

            n_muscle = 39


            ########################## SAVE EVERYTHING #######################################
            if count == 1:
                with h5py.File(save_DIR + 'dataset_train_' + name_dataset +'.hdf5', 'a') as file:
                    file.create_dataset('spindle_info', data=spindle_info_train, chunks=(1,n_muscle,t_size,2), maxshape=(None,n_muscle,t_size,2),compression="gzip", dtype='float32')
                    file.create_dataset('endeffector_coords', data=ee_train, chunks=(1,3,t_size), maxshape=(None,3,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('joint_coords', data=joints_train, chunks=(1,4,t_size), maxshape=(None,4,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('torque_coords', data=torque_train, chunks=(1,4,t_size), maxshape=(None,4,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('target_coords', data=target_train, chunks=(1,3,t_size), maxshape=(None,3,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('target_coords_one', data=target_one_train, chunks=(1,3), maxshape=(None,3),compression="gzip", dtype='float32')

                with h5py.File(save_DIR +'dataset_val_' + name_dataset +'.hdf5', 'a') as file:
                    file.create_dataset('spindle_info', data=spindle_info_val, chunks=(1,n_muscle,t_size,2), maxshape=(None,n_muscle,t_size,2),compression="gzip", dtype='float32')
                    file.create_dataset('endeffector_coords', data=ee_val, chunks=(1,3,t_size), maxshape=(None,3,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('joint_coords', data=joints_val, chunks=(1,4,t_size), maxshape=(None,4,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('torque_coords', data=torque_val, chunks=(1,4,t_size), maxshape=(None,4,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('target_coords', data=target_val, chunks=(1,3,t_size), maxshape=(None,3,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('target_coords_one', data=target_one_val, chunks=(1,3), maxshape=(None,3),compression="gzip", dtype='float32')

                with h5py.File(save_DIR +'dataset_test_' + name_dataset +'.hdf5', 'a') as file:
                    file.create_dataset('spindle_info', data=spindle_info_test, chunks=(1,n_muscle,t_size,2), maxshape=(None,n_muscle,t_size,2),compression="gzip", dtype='float32')
                    file.create_dataset('endeffector_coords', data=ee_test, chunks=(1,3,t_size), maxshape=(None,3,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('joint_coords', data=joints_test, chunks=(1,4,t_size), maxshape=(None,4,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('torque_coords', data=torque_test, chunks=(1,4,t_size), maxshape=(None,4,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('target_coords', data=target_test, chunks=(1,3,t_size), maxshape=(None,3,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('target_coords_one', data=target_one_test, chunks=(1,3), maxshape=(None,3),compression="gzip", dtype='float32')
                    
                with h5py.File(save_DIR +'dataset_train_' + name_dataset +'.hdf5', 'a') as file:
                    file.create_dataset('train_data_mean', data=mean_all)
            else:
                with h5py.File(save_DIR +'dataset_train_' + name_dataset +'.hdf5', 'a') as hf:
                    hf['spindle_info'].resize((hf['spindle_info'].shape[0] + spindle_info_train.shape[0]), axis = 0)
                    hf['spindle_info'][-spindle_info_train.shape[0]:,:,:,:] = spindle_info_train

                    hf['endeffector_coords'].resize((hf['endeffector_coords'].shape[0] + ee_train.shape[0]), axis = 0)
                    hf['endeffector_coords'][-ee_train.shape[0]:,:,:] = ee_train

                    hf['joint_coords'].resize((hf['joint_coords'].shape[0] + joints_train.shape[0]), axis = 0)
                    hf['joint_coords'][-joints_train.shape[0]:,:,:] = joints_train

                    hf['torque_coords'].resize((hf['torque_coords'].shape[0] + torque_train.shape[0]), axis = 0)
                    hf['torque_coords'][-torque_train.shape[0]:,:,:] = torque_train

                    hf['target_coords'].resize((hf['target_coords'].shape[0] + target_train.shape[0]), axis = 0)
                    hf['target_coords'][-target_train.shape[0]:,:,:] = target_train

                    hf['target_coords_one'].resize((hf['target_coords_one'].shape[0] + target_one_train.shape[0]), axis = 0)
                    hf['target_coords_one'][-target_one_train.shape[0]:,:] = target_one_train

                with h5py.File(save_DIR +'dataset_val_' + name_dataset +'.hdf5', 'a') as hf:
                    hf['spindle_info'].resize((hf['spindle_info'].shape[0] + spindle_info_val.shape[0]), axis = 0)
                    hf['spindle_info'][-spindle_info_val.shape[0]:,:,:,:] = spindle_info_val

                    hf['endeffector_coords'].resize((hf['endeffector_coords'].shape[0] + ee_val.shape[0]), axis = 0)
                    hf['endeffector_coords'][-ee_val.shape[0]:,:,:] = ee_val

                    hf['joint_coords'].resize((hf['joint_coords'].shape[0] + joints_val.shape[0]), axis = 0)
                    hf['joint_coords'][-joints_val.shape[0]:,:,:] = joints_val

                    hf['torque_coords'].resize((hf['torque_coords'].shape[0] + torque_val.shape[0]), axis = 0)
                    hf['torque_coords'][-torque_val.shape[0]:,:,:] = torque_val

                    hf['target_coords'].resize((hf['target_coords'].shape[0] + target_val.shape[0]), axis = 0)
                    hf['target_coords'][-target_val.shape[0]:,:,:] = target_val

                    hf['target_coords_one'].resize((hf['target_coords_one'].shape[0] + target_one_val.shape[0]), axis = 0)
                    hf['target_coords_one'][-target_one_val.shape[0]:,:] = target_one_val

                with h5py.File(save_DIR +'dataset_test_' + name_dataset +'.hdf5', 'a') as hf:
                    hf['spindle_info'].resize((hf['spindle_info'].shape[0] + spindle_info_test.shape[0]), axis = 0)
                    hf['spindle_info'][-spindle_info_test.shape[0]:,:,:,:] = spindle_info_test

                    hf['endeffector_coords'].resize((hf['endeffector_coords'].shape[0] + ee_test.shape[0]), axis = 0)
                    hf['endeffector_coords'][-ee_test.shape[0]:,:,:] = ee_test

                    hf['joint_coords'].resize((hf['joint_coords'].shape[0] + joints_test.shape[0]), axis = 0)
                    hf['joint_coords'][-joints_test.shape[0]:,:,:] = joints_test

                    hf['torque_coords'].resize((hf['torque_coords'].shape[0] + torque_test.shape[0]), axis = 0)
                    hf['torque_coords'][-torque_test.shape[0]:,:,:] = torque_test

                    hf['target_coords'].resize((hf['target_coords'].shape[0] + target_test.shape[0]), axis = 0)
                    hf['target_coords'][-target_test.shape[0]:,:,:] = target_test

                    hf['target_coords_one'].resize((hf['target_coords_one'].shape[0] + target_one_test.shape[0]), axis = 0)
                    hf['target_coords_one'][-target_one_test.shape[0]:,:] = target_one_test
            
            count += 1
        

    all_latents_dict = {'all_latents_train': latents_train,
                        'all_latents_test': latents_test,
                        'all_latents_val': latents_val}

    pickle.dump(all_latents_dict, open(save_DIR + 'latents_' + name_dataset +'.p', 'wb'), protocol=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate torque regression dataset')
    parser.add_argument('--name_dataset', type=str, help='Name of the dataset',default='rl_torque')
    parser.add_argument('--monkey_name', type=str, help='Name of the monkey used tp generate proprioceptive data',default='Snap')
    main(parser.parse_args())