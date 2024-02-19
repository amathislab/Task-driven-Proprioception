import os
import sys
import h5py
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import math as m
import argparse

sys.path.append('../neural_prediction/Code/')
from neural_utils import load_monkey_data, from_name_sortfields

import sys

sys.path.append('../code/')
from path_utils import PATH_TO_MATLAB_DATA, PATH_MONKEY_PROCESSED_DICT, PATH_MONKEY_PROCESSED_DATAFRAMES, PATH_TO_DATA_SPIKES

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

def apply_shuffle(traj, start_idx, end_idx, max_len = 400, is_spike=False):
    true_traj = traj[:, np.all(~np.isnan(traj), axis=0)]
    mytraj = np.zeros((true_traj.shape[0], max_len))
    mytraj[:, start_idx:end_idx] = true_traj
    if is_spike: #zero-pad spikes
        mytraj[:, :start_idx] = 0
        mytraj[:, end_idx:] = 0
    else:
        mytraj[:, :start_idx] = true_traj[:, 0][:, None]
        mytraj[:, end_idx:] = true_traj[:, -1][:, None]

    return mytraj
 
def Rx(theta):
    return np.matrix([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])

R_rot = Rx(m.pi/2)


def main(monkey, session):

    ## SAVING INFO
    dataset_path = PATH_TO_DATA_SPIKES

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    name_dataset = '{}_{}_center_out_spikes_datadriven'.format(monkey, session)

    ## DATA PREPARATION
    # Monkey with mismatched number of tials
    if monkey in ['Lando', 'S1Lando', 'Butter']:
        monkey_name = 'New' + monkey
    else:
        monkey_name = monkey

    muscle_len_all = []
    muscle_vel_all = []
    end_eff_all = []
    too_long = []
    spindle_all = []
    spikes_all = []
    latents_all = []
    trial_ids_all = []
    bad_trials = []
    too_long_trials = []

    t_size = 400
        
    # GET NEW MUSCLE DATA OPENSIM DATASET
    path_to_osim_data = os.path.join(PATH_MONKEY_PROCESSED_DICT, monkey_name + '_' + str(session) + '_mod.pkl')
    print('Loading data from:', path_to_osim_data)
    monkey_file = pickle.load(open(path_to_osim_data, 'rb'))

    # Check which dataset has the correct markers
    if monkey_name == 'NewS1Lando':
        monkey_name_marker = 'S1Lando'
    else:
        monkey_name_marker = monkey_name
    monkey_file_marker = pickle.load(open(os.path.join(PATH_MONKEY_PROCESSED_DATAFRAMES,monkey_name_marker + '_' + str(session) + '_TD_df.pkl'), 'rb'))

    # Get the markers position for Butter to remove bad trials
    if monkey_name == 'NewButter':
        print('For Butter, getting marker data...')
        marker_pos = monkey_file_marker['markers'].to_numpy()

    if monkey_name in ['NewLando', 'NewS1Lando']: #Lando - remove unmatched trials
        monkey_file.pop('trial_71')
        monkey_file.pop('trial_412')
        monkey_file.pop('trial_725')

    # LOAD RAW SPIKE DATA
    data_df = load_monkey_data(PATH_TO_MATLAB_DATA,
                               monkey=monkey,
                               session_date=session,
                               keep_kinematics=True,
                               keep_spike_data=True,
                               use_new_muscle=True)

    spike_field, _ ,_ = from_name_sortfields(monkey)
    print('Spike data field:', spike_field)

    # Iterate over trials
    for t_idx, ii in enumerate(monkey_file.keys()):
        latents = dict()
        name = str(ii)

        ## Get information from recomputed muscle length and velocity
        mus_len_tmp = monkey_file[name]['muscle_len']
        mus_vel_tmp = monkey_file[name]['muscle_vel']
        endeff_tmp = monkey_file[name]['endeffector_coords']

        spike_tmp = data_df.iloc[t_idx][spike_field].T

        idx_movement_on = int(data_df.iloc[t_idx]['idx_movement_on'])
        idx_endTime = int(data_df.iloc[t_idx]['idx_endTime'])

        # Take only from movement onset to end time
        mus_len_tmp = mus_len_tmp[:, idx_movement_on:idx_endTime]
        mus_vel_tmp = mus_vel_tmp[:, idx_movement_on:idx_endTime]
        endeff_tmp = endeff_tmp[:, idx_movement_on:idx_endTime]

        # Convolve spikes with 50 ms (before shuffling, padding)
        spike_tmp = np.apply_along_axis(lambda m: np.convolve(m, np.ones(5) / 5, mode='same'),
                                                axis=1, arr=spike_tmp)
        
        spike_tmp = spike_tmp[:, idx_movement_on:idx_endTime]


        flag_traj = True

        # Remove bad trials for Butter
        if monkey_name == 'NewButter':
            marker_pos_tmp = R_rot.dot(marker_pos[t_idx][:,6:9].T)*100

            if ((np.any(np.abs(marker_pos_tmp[0,:]) > 35)) or np.any(np.abs(marker_pos_tmp[0,:]) < 5) or (np.any(np.abs(marker_pos_tmp[1,:]) > 13)) or (np.any(abs(endeff_tmp[2,:]) > 13))):
                flag_traj = False
                bad_trials.append(t_idx)

        ## Remove trials longer than 4 seconds
        if mus_len_tmp.shape[1] > t_size:
            flag_traj = False
            too_long.append(ii)
            too_long_trials.append(t_idx)

        ## If trajectory is okay
        if flag_traj:

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

            spike_tmp = apply_shuffle(spike_tmp, start_idx, end_idx, is_spike=True)

            mus_len_tmp1 = np.expand_dims(mus_len_tmp,-1) #[...,].shape)
            mus_vel_tmp1 = np.expand_dims(mus_vel_tmp,-1)
            spindle_input = np.concatenate((mus_len_tmp1, mus_vel_tmp1),axis=2)
            spikes_output = spike_tmp


            ## Update dict
            latents['idx_movement_on'] = idx_movement_on
            latents['idx_endTime'] = idx_endTime
            latents['trial_id'] = t_idx
            latents['start_idx'] = start_idx #within 400 time points
            latents['end_idx'] = end_idx #within 400 time points
            latents['monkey_name'] = monkey_name
            latents['mus_jerk'] = mus_jerk_tmp

            spindle_all.append(spindle_input)
            end_eff_all.append(endeff_tmp)
            latents_all.append(latents)
            trial_ids_all.append(t_idx) # All kept trial indices
            spikes_all.append(spikes_output)

    ############################ CREATE DATASETS ##################################

    ## SPLIT TRAIN/TEST SHUFFLED TRIAL INDICES --> This will be used for all future predictions
    trial_ids_all_reindex = np.arange(len(trial_ids_all))
    ind_train, ind_test = train_test_split(np.asarray(trial_ids_all_reindex),
                                           shuffle=True, #TRUE, we shuffle within behavioral session
                                           test_size=0.2)

    ind_train = ind_train.astype(int)
    ind_test = ind_test.astype(int)

    ind_train_saved = [trial_ids_all[i] for i in ind_train]
    ind_test_saved = [trial_ids_all[i]  for i in ind_test]

    # Split data
    spindle_info_train = np.array(spindle_all)[ind_train]
    spindle_info_test = np.array(spindle_all)[ind_test]

    mean_spindles_train = np.mean(np.array(spindle_info_train), 0)

    ee_train = np.array(end_eff_all)[ind_train]
    ee_test = np.array(end_eff_all)[ind_test]

    spikes_train = np.array(spikes_all)[ind_train]
    spikes_test = np.array(spikes_all)[ind_test]

    latents_train = np.array(latents_all)[ind_train]
    latents_test = np.array(latents_all)[ind_test]


    ########################## SAVE EVERYTHING #######################################
    n_muscle = 39
    n_neurons = data_df[spike_field][0].shape[-1]
    print('Neurons:', n_neurons)

    # Train
    with h5py.File(dataset_path + '/dataset_train_' + name_dataset +'.hdf5', 'w') as file:
        file.create_dataset('spindle_info', data=spindle_info_train, chunks=(1,n_muscle,t_size,2), maxshape=(None,n_muscle,t_size,2),compression="gzip", dtype='float32')
        file.create_dataset('endeffector_coords', data=ee_train, chunks=(1,3,t_size), maxshape=(None,3,t_size),compression="gzip", dtype='float32')
        file.create_dataset('spike_info', data=spikes_train, chunks=(1,n_neurons,t_size), maxshape=(None,n_neurons,t_size),compression="gzip", dtype='float32')
        file.create_dataset('train_data_mean', data=mean_spindles_train)
        file.create_dataset('indices_info', data=ind_train, chunks=(1,), maxshape=(None,), compression="gzip", dtype='int')

    # Test
    with h5py.File(dataset_path +'/dataset_test_' + name_dataset +'.hdf5', 'w') as file:
        file.create_dataset('spindle_info', data=spindle_info_test, chunks=(1,n_muscle,t_size,2), maxshape=(None,n_muscle,t_size,2),compression="gzip", dtype='float32')
        file.create_dataset('endeffector_coords', data=ee_test, chunks=(1,3,t_size), maxshape=(None,3,t_size),compression="gzip", dtype='float32')
        file.create_dataset('spike_info', data=spikes_test, chunks=(1,n_neurons,t_size), maxshape=(None,n_neurons,t_size),compression="gzip", dtype='float32')
        file.create_dataset('indices_info', data=ind_test, chunks=(1,), maxshape=(None,), compression="gzip", dtype='int')


    # Save latents
    all_latents_dict = {'all_latents_train': latents_train,
                        'all_latents_test': latents_test}
    pickle.dump(all_latents_dict, open(dataset_path + '/latents_' + name_dataset +'.p', 'wb'), protocol=4)

    # Save excluded trials
    excluded_trials = {'bad_trials':bad_trials,
            'too_long':too_long_trials}
    pickle.dump(excluded_trials, open(dataset_path + '/excludedtrials_' + name_dataset + '.p', 'wb'), protocol=4)
    print('Excluded trials', excluded_trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate monkey/session spike dataset for data-driven network models.')

    parser.add_argument('--monkey', type=str, help='Which monkey?', required=True)
    parser.add_argument('--session', type=int, help='Which session date?', required=True)

    args = parser.parse_args()

    main(args.monkey, args.session)