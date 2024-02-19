'''
Script to generate padded kinematic + spike session dataset for a monkey,
in active and passive center-out reaching task.
'''

#Imports
import sys, os
import numpy as np
import random
import h5py
import argparse
import pickle

# Modules
sys.path.append('./../neural_prediction/Code/')
from neural_utils import load_monkey_data, is_cobump_task, generate_spike_active_dataset, generate_spike_passive_dataset
from motion_utils import generate_kin_active_dataset, generate_kin_passive_dataset

sys.path.append('../Code/')
from path_utils import PATH_TO_MATLAB_DATA, PATH_TO_DATASPLITS, PATH_TO_BEH_EXP

def main(monkey_name, session_date, active_start = 'mvt',
         active_length = 0, align=100, permut_m=False, permut_t=False, constant_input=False):

    #Load and pre-process monkey data from .mat file
    data_df = load_monkey_data(PATH_TO_MATLAB_DATA,
                               monkey = monkey_name,
                               session_date = session_date,
                               keep_kinematics = True,
                               keep_spike_data = True,
                               use_new_muscle= True)

    print('MATLAB monkey session data file loaded and pre-processed.')

    #Check if it contains passive bump
    is_cobump = is_cobump_task(data_df)

    # Trial ndices
    data_ids = np.asarray(data_df.index) #ordered rows ids (new muscle data)
    trial_ids = np.asarray(data_df.trial_id) # original MATLAB field

    #Get trial durations (active)
    if active_start == 'cue':
        start_indices = data_df.idx_goCueTime.values
    elif active_start == 'mvt':
        start_indices = data_df.idx_movement_on.values

    if active_length != 0:
        end_indices = start_indices + active_length
    elif active_length == 0:
        end_indices = data_df.idx_endTime

    trial_durations_active = np.asarray(end_indices - start_indices)
    trial_durations_passive = np.ones(len(data_df)) * 13

    #Generate kinematic datasets
    #Active
    active_kin_arr, active_joint_arr, active_endeff_arr = generate_kin_active_dataset(data_df, active_start, active_length, align)

    #Passive
    if is_cobump:
        passive_kin_arr, passive_joint_arr, passive_endeff_arr = generate_kin_passive_dataset(data_df, align)

    #Generate spike datasets
    #Active
    active_spike_arr = generate_spike_active_dataset(data_df, monkey_name, active_start, active_length, align, window=5, latency=0)

    #Passive
    if is_cobump:
        passive_spike_arr = generate_spike_passive_dataset(data_df, monkey_name, align)

    print('All {} session {} datasets generated!'.format(monkey_name, session_date))

    # CONTROL DATASETS
    rng = np.random.default_rng()
    random.seed(42)

    if permut_m:
        active_kin_arr = rng.permutation(active_kin_arr, axis=1)
        passive_kin_arr = rng.permutation(passive_kin_arr, axis=1)

    if permut_t:
        active_kin_arr = rng.permutation(active_kin_arr, axis=2)
        passive_kin_arr = rng.permutation(passive_kin_arr, axis=2)

        active_spike_arr = rng.permutation(active_spike_arr, axis=2)
        passive_spike_arr = rng.permutation(passive_spike_arr, axis=2)

    if permut_m and permut_t:
        active_kin_arr = rng.permutation(active_kin_arr, axis=1)
        passive_kin_arr = rng.permutation(passive_kin_arr, axis=1)
        active_kin_arr = rng.permutation(active_kin_arr, axis=2)
        passive_kin_arr = rng.permutation(passive_kin_arr, axis=2)

    if constant_input:
        # Mean length
        active_kin_arr = np.repeat(np.expand_dims(np.nanmean(active_kin_arr, axis=2), axis=2),
                                   repeats=400, axis=2)
        passive_kin_arr = np.repeat(np.expand_dims(np.nanmean(passive_kin_arr, axis=2), axis=2),
                                   repeats=400, axis=2)
        # Zero velocity
        active_kin_arr[:, :, :, 1] = 0.0
        passive_kin_arr[:, :, :, 1] = 0.0

        print(active_kin_arr[0])

    ### Remove bad trials for Butter
    if monkey_name == 'Butter':
        excl_trial_filename = 'excludedtrials_Butter_20180326_center_out_spikes_datadriven.p'
        excl_trials = pickle.load(open( os.path.join(PATH_TO_DATASPLITS,excl_trial_filename), 'rb'))
        all_bad_trials = excl_trials['bad_trials'] + excl_trials['too_long']
        active_spike_arr = np.delete(active_spike_arr, all_bad_trials, axis=0)
        active_kin_arr = np.delete(active_kin_arr, all_bad_trials, axis =0)
        trial_ids = np.delete(trial_ids, all_bad_trials, axis =0)
        data_ids = np.delete(data_ids, all_bad_trials, axis =0)
        trial_durations_active = np.delete(trial_durations_active, all_bad_trials, axis =0)
        active_endeff_arr = np.delete(active_endeff_arr, all_bad_trials, axis =0)

    # SAVE DATASETS AS HDF5
    n_muscle = 39
    t_size = 400
    n_neurons = active_spike_arr.shape[2]
    _, n_joints, _, _ = active_joint_arr.shape

    #Make file suffix
    active_start_suff = '_' + str(active_start)
    if active_length == 0:
        active_length_suff = '_end'
    else:
        active_length_suff = '_' + str(active_length) + 'ms'
    align_suff = '_at' + str(align)
    permut_suff = ''
    if permut_m or permut_t:
        permut_suff = '_'
    if permut_m:
        permut_suff += 'M'
    if permut_t:
        permut_suff += 'T'
    const_suff = ''
    if constant_input:
        const_suff = '_const'

    file_name_suffix_active = '{}{}{}{}{}'.format(active_start_suff,
                                                  active_length_suff,
                                                  align_suff,
                                                  permut_suff,
                                                  const_suff)
    file_name_suffix_passive = '{}{}{}'.format(align_suff,
                                               permut_suff,
                                               const_suff)

    # Active
    path_to_folder = os.path.join(PATH_TO_BEH_EXP, 'MonkeyAlignedDatasets_prova')
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
    dataset_name = monkey_name + '_' + str(session_date) + '_active{}.hdf5'.format(file_name_suffix_active)
    print('Saving dataset:', dataset_name)
    path_to_dataset = os.path.join(path_to_folder, dataset_name)

    with h5py.File(path_to_dataset, 'a') as file:
        file.create_dataset('trial_ids', data=trial_ids, maxshape=(None), compression='gzip')
        file.create_dataset('data_ids', data=data_ids, maxshape=(None), compression='gzip')
        file.create_dataset('trial_durations', data=trial_durations_active, maxshape=(None), compression='gzip')
        file.create_dataset('muscle_coords', data=active_kin_arr*1000, maxshape=(None, n_muscle, t_size, None), compression="gzip")
        file.create_dataset('joint_coords', data=active_joint_arr, maxshape=(None, n_joints, t_size, None), compression="gzip")
        file.create_dataset('endeff_coords', data=active_endeff_arr, maxshape=(None, t_size, 2,2), compression="gzip")
        file.create_dataset('spike_counts', data=active_spike_arr, maxshape=(None, t_size, n_neurons), compression="gzip")

    print('Saved in: ',path_to_dataset)
    print('Dataset saved:', dataset_name)

    #Passive
    if is_cobump:
        dataset_name = monkey_name + '_' + str(session_date) + '_passive{}.hdf5'.format(file_name_suffix_passive)
        print('Saving dataset:', dataset_name)
        path_to_dataset = os.path.join(path_to_folder, dataset_name)

        with h5py.File(path_to_dataset, 'a') as file:
            file.create_dataset('trial_ids', data=trial_ids, maxshape=(None), compression='gzip')
            file.create_dataset('data_ids', data=data_ids, maxshape=(None), compression='gzip')
            file.create_dataset('trial_durations', data=trial_durations_passive, maxshape=(None), compression='gzip')
            file.create_dataset('muscle_coords', data=passive_kin_arr, maxshape=(None, n_muscle, t_size, None), compression="gzip")
            file.create_dataset('joint_coords', data=passive_joint_arr, maxshape=(None, n_joints, t_size, None), compression="gzip")
            file.create_dataset('endeff_coords', data=passive_endeff_arr, maxshape=(None, t_size, 2, 2), compression="gzip")
            file.create_dataset('spike_counts', data=passive_spike_arr, maxshape=(None, t_size, n_neurons), compression="gzip")
        print('Dataset saved:', dataset_name)
    
    print('*********************************')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate padded datasets for a monkey.')

    parser.add_argument('--monkey', type=str, help='Which monkey?', required=False, default='Butter')  #Snap
    parser.add_argument('--session', type=int, help='Which session data?', required=False, default=20180326)  #20190829
    parser.add_argument('--active_start', type=str,  default='mvt', help='Which active start index?', required=False)
    parser.add_argument('--active_length', type=int, default=0, help='Active data fraction.', required=False)
    parser.add_argument('--align', type=int, default=100, help='Where to align movement onset?', required=False)
    parser.add_argument('--permut_m', action='store_false', help='Permut muscles control?', required=False, default=False)
    parser.add_argument('--permut_t', action='store_false', help='Permut time control?', required=False, default=False)
    parser.add_argument('--constant_input', action='store_false', help='Constant input control?', required=False, default=False)

    args = parser.parse_args()

    main(args.monkey, args.session, args.active_start, args.active_length, args.align,
         args.permut_m, args.permut_t, args.constant_input)