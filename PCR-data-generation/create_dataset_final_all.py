# %matplotlib inline
import os
import h5py

import scipy.signal
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import argparse

import sys

sys.path.append('../code/')
from path_utils import PATH_TO_SAVE_SPINDLEDATASET, PATH_TO_UNPROCESSED_DATA

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

def add_noise(mconf, factor):
    return mconf + factor*mconf.std(axis=1)[:, None]*np.random.randn(*mconf.shape)

def main(args):
    t_size = 400
    n_muscle = 39
    time_step = 0.010 # 0.015 s at 66.7 Hz otherwise --> 0.005 at 200 Hz
    mean_all = np.zeros((n_muscle,t_size,2))

    monkey_name = args.monkey_name
    init_folder = args.init_folder
    end_folder = args.end_folder
    # path_unprocessed_data = args.path_unprocessed_data
    lab_count_per_char = args.lab_count_per_char
    step_loop = args.step_loop
    
    dataset_path = os.path.join(PATH_TO_SAVE_SPINDLEDATASET, monkey_name)

    name_dataset = monkey_name

    # SELECT FOLDER NAME TO GET DATA
    ind_unp = np.arange(init_folder,end_folder,1)

    count = 1
    count_mean = 0

    all_latents_train = []
    all_latents_val = []
    all_latents_test = []

    for j in ind_unp: #ind_unp: #121 range(101,102,1) or 120
        path = PATH_TO_UNPROCESSED_DATA + 'unprocessed_data_' + monkey_name + '/' + str(j) + '/'

        if j % 5 == 0:
            print('folder:',j)

        ## Only vertical or horizontal
        start_loop = 1 # Horizontal

        for i in range(start_loop,41,step_loop): #41 before 22  (1,40,2)
            path_file = path + str(i) + '.p'
            file = open(path_file, "rb" )
            traj_set = pickle.load(file)
            file.close()
            latents_tmp = {}
            
            all_spindle = []
            all_ee = []
            all_elbow = []
            all_joint = []

            all_ee_vel = []
            all_elbow_vel = []
            all_joint_vel = []

            all_ee_acc = []
            all_elbow_acc = []
            all_joint_acc = []
            all_muscle_acc = []

            labels = []
            latents_dict = []

            lab_count = 0

            ind_all_traj = np.arange(len(traj_set[0]))
            np.random.shuffle(ind_all_traj)

            for ii in ind_all_traj: #range(len(traj_set[0])):

                traj = traj_set[0][ii]

                muscle_tmp = np.float32(traj['muscle_coords'])
                endeff_tmp = np.float32(traj['endeffector_coords'])
                elbow_tmp = np.float32(traj['marker6'])
                joint_tmp = np.float32(traj['joint_coords'])
                mus_jerk_tmp = traj['muscle_jerk']
                lab_tmp = traj['label']

                if (mus_jerk_tmp > 1):
                    muscle_tmp = scipy.signal.medfilt(muscle_tmp, kernel_size=(1,5))
                    mus_jerk_tmp = compute_jerk(muscle_tmp)

                ## Discard nan, take only traj with muscle_jer less than one and only a specific amount of trajectory to have a balanced dataset
                if not np.any(np.isnan(muscle_tmp)) and (lab_count < lab_count_per_char) and (mus_jerk_tmp < 1):
                    ## This if only for without monkey
                    start_idx, end_idx = start_end_choice(muscle_tmp, t_size)

                    ## Pad and shuffle initial position
                    muscle_tmp = apply_shuffle(muscle_tmp, start_idx, end_idx)
                    joint_tmp = apply_shuffle(joint_tmp, start_idx, end_idx)
                    endeff_tmp = apply_shuffle(endeff_tmp, start_idx, end_idx)
                    elbow_tmp = apply_shuffle(elbow_tmp, start_idx, end_idx)

                    ## Compute velocity from muscle length
                    vel_tmp = np.float32(np.gradient(muscle_tmp,time_step,axis=1))  #0.015  200 Hz: 0.005
                    stack_tmp = np.stack((muscle_tmp,vel_tmp),axis=-1)

                    ee_vel_tmp = np.float32(np.gradient(endeff_tmp,time_step,axis=1))
                    elbow_vel_tmp = np.float32(np.gradient(elbow_tmp,time_step,axis=1))
                    joints_vel_tmp = np.float32(np.gradient(joint_tmp,time_step,axis=1))

                    ee_acc_tmp = np.float32(np.gradient(ee_vel_tmp,time_step,axis=1))
                    elbow_acc_tmp = np.float32(np.gradient(elbow_vel_tmp,time_step,axis=1))
                    joints_acc_tmp = np.float32(np.gradient(joints_vel_tmp,time_step,axis=1))
                    muscle_acc_tmp = np.float32(np.gradient(vel_tmp,time_step,axis=1))

                    ## Add counter per each label
                    lab_count += 1

        #             traj['spindle_info'] = np.float32(stack_tmp)
                    traj['latents']['absolute_idx'] = np.array([j,i,ii])
                    traj['latents']['start_idx'] = start_idx
                    traj['latents']['end_idx'] = end_idx

                    latents_tmp = traj['latents']

                    labels.append(lab_tmp)
                    latents_dict.append(latents_tmp)

                    all_spindle.append(stack_tmp)
                    all_ee.append(endeff_tmp)
                    all_joint.append(joint_tmp)

                    all_elbow.append(elbow_tmp)

                    all_ee_vel.append(ee_vel_tmp)
                    all_elbow_vel.append(elbow_vel_tmp)
                    all_joint_vel.append(joints_vel_tmp)
                    
                    all_ee_acc.append(ee_acc_tmp)
                    all_elbow_acc.append(elbow_acc_tmp)
                    all_joint_acc.append(joints_acc_tmp)
                    all_muscle_acc.append(muscle_acc_tmp)

            ind_all = np.arange(len(all_spindle))

            ## Get test indices
            ind_train, ind_test = train_test_split(ind_all, test_size=0.2)

            ## Get train and val indices
            ind_train, ind_val = train_test_split(ind_train, test_size=0.1)

            ind_train = ind_train.astype(int)
            ind_test = ind_test.astype(int)
            ind_val = ind_val.astype(int)

            spindle_info_train = np.array(all_spindle)[ind_train]
            spindle_info_test = np.array(all_spindle)[ind_test]
            spindle_info_val = np.array(all_spindle)[ind_val]

            mean_all = mean_all + np.sum(spindle_info_train,0)
            count_mean = count_mean + spindle_info_train.shape[0]

            ee_train = np.array(all_ee)[ind_train]
            ee_test = np.array(all_ee)[ind_test]
            ee_val = np.array(all_ee)[ind_val]

            joint_train = np.array(all_joint)[ind_train]
            joint_test = np.array(all_joint)[ind_test]
            joint_val = np.array(all_joint)[ind_val]

            elbow_train = np.array(all_elbow)[ind_train]
            elbow_test = np.array(all_elbow)[ind_test]
            elbow_val = np.array(all_elbow)[ind_val]

            ee_vel_train = np.array(all_ee_vel)[ind_train]
            ee_vel_test = np.array(all_ee_vel)[ind_test]
            ee_vel_val = np.array(all_ee_vel)[ind_val]

            elbow_vel_train = np.array(all_elbow_vel)[ind_train]
            elbow_vel_test = np.array(all_elbow_vel)[ind_test]
            elbow_vel_val = np.array(all_elbow_vel)[ind_val]

            joint_vel_train = np.array(all_joint_vel)[ind_train]
            joint_vel_test = np.array(all_joint_vel)[ind_test]
            joint_vel_val = np.array(all_joint_vel)[ind_val]

            ee_acc_train = np.array(all_ee_acc)[ind_train]
            ee_acc_test = np.array(all_ee_acc)[ind_test]
            ee_acc_val = np.array(all_ee_acc)[ind_val]

            elbow_acc_train = np.array(all_elbow_acc)[ind_train]
            elbow_acc_test = np.array(all_elbow_acc)[ind_test]
            elbow_acc_val = np.array(all_elbow_acc)[ind_val]

            joint_acc_train = np.array(all_joint_acc)[ind_train]
            joint_acc_test = np.array(all_joint_acc)[ind_test]
            joint_acc_val = np.array(all_joint_acc)[ind_val]

            muscle_acc_train = np.array(all_muscle_acc)[ind_train]
            muscle_acc_test = np.array(all_muscle_acc)[ind_test]
            muscle_acc_val = np.array(all_muscle_acc)[ind_val]

            lab_train = np.array(labels)[ind_train]
            lab_test = np.array(labels)[ind_test]
            lab_val = np.array(labels)[ind_val]

            latents_train = np.array(latents_dict)[ind_train]
            latents_test = np.array(latents_dict)[ind_test]
            latents_val = np.array(latents_dict)[ind_val]

            all_latents_train.append(latents_train)
            all_latents_test.append(latents_test)
            all_latents_val.append(latents_val)

            n_elbow = elbow_acc_train.shape[1]
            n_ee = ee_vel_train.shape[1]
            n_joints = joint_train.shape[1]

            if count == 1:
                with h5py.File(dataset_path + 'dataset_train_' + name_dataset +'.hdf5', 'a') as file:
                    file.create_dataset('spindle_info', data=spindle_info_train, chunks=(1,n_muscle,t_size,2), maxshape=(None,n_muscle,t_size,2),compression="gzip", dtype='float32')
                    file.create_dataset('endeffector_coords', data=ee_train, chunks=(1,3,t_size), maxshape=(None,3,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('joint_coords', data=joint_train, chunks=(1,4,t_size), maxshape=(None,4,t_size),compression="gzip", dtype='float32')

                    file.create_dataset('elbow_coords', data=elbow_train, chunks=(1,n_elbow,t_size), maxshape=(None,n_elbow,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('elbow_vel', data=elbow_vel_train, chunks=(1,n_elbow,t_size), maxshape=(None,n_elbow,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('elbow_acc', data=elbow_acc_train, chunks=(1,n_elbow,t_size), maxshape=(None,n_elbow,t_size),compression="gzip", dtype='float32')

                    file.create_dataset('endeffector_vel', data=ee_vel_train, chunks=(1,n_ee,t_size), maxshape=(None,n_ee,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('endeffector_acc', data=ee_acc_train, chunks=(1,n_ee,t_size), maxshape=(None,n_ee,t_size),compression="gzip", dtype='float32')

                    file.create_dataset('joint_vel', data=joint_vel_train, chunks=(1,n_joints,t_size), maxshape=(None,n_joints,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('joint_acc', data=joint_acc_train, chunks=(1,n_joints,t_size), maxshape=(None,n_joints,t_size),compression="gzip", dtype='float32')

                    file.create_dataset('muscle_acc', data=muscle_acc_train, chunks=(1,n_muscle,t_size), maxshape=(None,n_muscle,t_size),compression="gzip", dtype='float32')

                    file.create_dataset('label', data=lab_train, chunks=(1,), maxshape=(None,),compression="gzip")

                with h5py.File(dataset_path +'dataset_val_' + name_dataset +'.hdf5', 'a') as file:
                    file.create_dataset('spindle_info', data=spindle_info_val, chunks=(1,n_muscle,t_size,2), maxshape=(None,n_muscle,t_size,2),compression="gzip", dtype='float32')
                    file.create_dataset('endeffector_coords', data=ee_val, chunks=(1,3,t_size), maxshape=(None,3,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('joint_coords', data=joint_val, chunks=(1,4,t_size), maxshape=(None,4,t_size),compression="gzip", dtype='float32')

                    file.create_dataset('elbow_coords', data=elbow_val, chunks=(1,n_elbow,t_size), maxshape=(None,n_elbow,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('elbow_vel', data=elbow_vel_val, chunks=(1,n_elbow,t_size), maxshape=(None,n_elbow,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('elbow_acc', data=elbow_acc_val, chunks=(1,n_elbow,t_size), maxshape=(None,n_elbow,t_size),compression="gzip", dtype='float32')

                    file.create_dataset('endeffector_vel', data=ee_vel_val, chunks=(1,n_ee,t_size), maxshape=(None,n_ee,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('endeffector_acc', data=ee_acc_val, chunks=(1,n_ee,t_size), maxshape=(None,n_ee,t_size),compression="gzip", dtype='float32')

                    file.create_dataset('joint_vel', data=joint_vel_val, chunks=(1,n_joints,t_size), maxshape=(None,n_joints,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('joint_acc', data=joint_acc_val, chunks=(1,n_joints,t_size), maxshape=(None,n_joints,t_size),compression="gzip", dtype='float32')

                    file.create_dataset('label', data=lab_val, chunks=(1,), maxshape=(None,),compression="gzip")

                with h5py.File(dataset_path +'dataset_test_' + name_dataset +'.hdf5', 'a') as file:
                    file.create_dataset('spindle_info', data=spindle_info_test, chunks=(1,n_muscle,t_size,2), maxshape=(None,n_muscle,t_size,2),compression="gzip", dtype='float32')
                    file.create_dataset('endeffector_coords', data=ee_test, chunks=(1,3,t_size), maxshape=(None,3,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('joint_coords', data=joint_test, chunks=(1,4,t_size), maxshape=(None,4,t_size),compression="gzip", dtype='float32')

                    file.create_dataset('elbow_coords', data=elbow_test, chunks=(1,n_elbow,t_size), maxshape=(None,n_elbow,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('elbow_vel', data=elbow_vel_test, chunks=(1,n_elbow,t_size), maxshape=(None,n_elbow,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('elbow_acc', data=elbow_acc_test, chunks=(1,n_elbow,t_size), maxshape=(None,n_elbow,t_size),compression="gzip", dtype='float32')

                    file.create_dataset('endeffector_vel', data=ee_vel_test, chunks=(1,n_ee,t_size), maxshape=(None,n_ee,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('endeffector_acc', data=ee_acc_test, chunks=(1,n_ee,t_size), maxshape=(None,n_ee,t_size),compression="gzip", dtype='float32')

                    file.create_dataset('joint_vel', data=joint_vel_test, chunks=(1,n_joints,t_size), maxshape=(None,n_joints,t_size),compression="gzip", dtype='float32')
                    file.create_dataset('joint_acc', data=joint_acc_test, chunks=(1,n_joints,t_size), maxshape=(None,n_joints,t_size),compression="gzip", dtype='float32')

                    file.create_dataset('label', data=lab_test, chunks=(1,), maxshape=(None,),compression="gzip")
            else:
                with h5py.File(dataset_path +'dataset_train_' + name_dataset +'.hdf5', 'a') as hf:
                    hf['spindle_info'].resize((hf['spindle_info'].shape[0] + spindle_info_train.shape[0]), axis = 0)
                    hf['spindle_info'][-spindle_info_train.shape[0]:,:,:,:] = spindle_info_train

                    hf['endeffector_coords'].resize((hf['endeffector_coords'].shape[0] + ee_train.shape[0]), axis = 0)
                    hf['endeffector_coords'][-ee_train.shape[0]:,:,:] = ee_train

                    hf['joint_coords'].resize((hf['joint_coords'].shape[0] + joint_train.shape[0]), axis = 0)
                    hf['joint_coords'][-joint_train.shape[0]:,:,:] = joint_train

                    hf['elbow_coords'].resize((hf['elbow_coords'].shape[0] + elbow_train.shape[0]), axis = 0)
                    hf['elbow_coords'][-elbow_train.shape[0]:,:,:] = elbow_train

                    hf['elbow_vel'].resize((hf['elbow_vel'].shape[0] + elbow_vel_train.shape[0]), axis = 0)
                    hf['elbow_vel'][-elbow_vel_train.shape[0]:,:,:] = elbow_vel_train

                    hf['elbow_acc'].resize((hf['elbow_acc'].shape[0] + elbow_acc_train.shape[0]), axis = 0)
                    hf['elbow_acc'][-elbow_acc_train.shape[0]:,:,:] = elbow_acc_train

                    hf['endeffector_vel'].resize((hf['endeffector_vel'].shape[0] + ee_vel_train.shape[0]), axis = 0)
                    hf['endeffector_vel'][-ee_vel_train.shape[0]:,:,:] = ee_vel_train

                    hf['endeffector_acc'].resize((hf['endeffector_acc'].shape[0] + ee_acc_train.shape[0]), axis = 0)
                    hf['endeffector_acc'][-ee_acc_train.shape[0]:,:,:] = ee_acc_train

                    hf['joint_vel'].resize((hf['joint_vel'].shape[0] + joint_vel_train.shape[0]), axis = 0)
                    hf['joint_vel'][-joint_vel_train.shape[0]:,:,:] = joint_vel_train

                    hf['joint_acc'].resize((hf['joint_acc'].shape[0] + joint_acc_train.shape[0]), axis = 0)
                    hf['joint_acc'][-joint_acc_train.shape[0]:,:,:] = joint_acc_train

                    hf['muscle_acc'].resize((hf['muscle_acc'].shape[0] + muscle_acc_train.shape[0]), axis = 0)
                    hf['muscle_acc'][-muscle_acc_train.shape[0]:,:,:] = muscle_acc_train

                    hf['label'].resize((hf['label'].shape[0] + lab_train.shape[0]), axis = 0)
                    hf['label'][-lab_train.shape[0]:] = lab_train

                with h5py.File(dataset_path +'dataset_val_' + name_dataset +'.hdf5', 'a') as hf:
                    hf['spindle_info'].resize((hf['spindle_info'].shape[0] + spindle_info_val.shape[0]), axis = 0)
                    hf['spindle_info'][-spindle_info_val.shape[0]:,:,:,:] = spindle_info_val

                    hf['endeffector_coords'].resize((hf['endeffector_coords'].shape[0] + ee_val.shape[0]), axis = 0)
                    hf['endeffector_coords'][-ee_val.shape[0]:,:,:] = ee_val

                    hf['joint_coords'].resize((hf['joint_coords'].shape[0] + joint_val.shape[0]), axis = 0)
                    hf['joint_coords'][-joint_val.shape[0]:,:,:] = joint_val

                    hf['elbow_coords'].resize((hf['elbow_coords'].shape[0] + elbow_val.shape[0]), axis = 0)
                    hf['elbow_coords'][-elbow_val.shape[0]:,:,:] = elbow_val

                    hf['elbow_vel'].resize((hf['elbow_vel'].shape[0] + elbow_vel_val.shape[0]), axis = 0)
                    hf['elbow_vel'][-elbow_vel_val.shape[0]:,:,:] = elbow_vel_val

                    hf['elbow_acc'].resize((hf['elbow_acc'].shape[0] + elbow_acc_val.shape[0]), axis = 0)
                    hf['elbow_acc'][-elbow_acc_val.shape[0]:,:,:] = elbow_acc_val

                    hf['endeffector_vel'].resize((hf['endeffector_vel'].shape[0] + ee_vel_val.shape[0]), axis = 0)
                    hf['endeffector_vel'][-ee_vel_val.shape[0]:,:,:] = ee_vel_val

                    hf['endeffector_acc'].resize((hf['endeffector_acc'].shape[0] + ee_acc_val.shape[0]), axis = 0)
                    hf['endeffector_acc'][-ee_acc_val.shape[0]:,:,:] = ee_acc_val

                    hf['joint_vel'].resize((hf['joint_vel'].shape[0] + joint_vel_val.shape[0]), axis = 0)
                    hf['joint_vel'][-joint_vel_val.shape[0]:,:,:] = joint_vel_val

                    hf['joint_acc'].resize((hf['joint_acc'].shape[0] + joint_acc_val.shape[0]), axis = 0)
                    hf['joint_acc'][-joint_acc_val.shape[0]:,:,:] = joint_acc_val

                    hf['muscle_acc'].resize((hf['muscle_acc'].shape[0] + muscle_acc_val.shape[0]), axis = 0)
                    hf['muscle_acc'][-muscle_acc_val.shape[0]:,:,:] = muscle_acc_val

                    hf['label'].resize((hf['label'].shape[0] + lab_val.shape[0]), axis = 0)
                    hf['label'][-lab_val.shape[0]:] = lab_val

                with h5py.File(dataset_path +'dataset_test_' + name_dataset +'.hdf5', 'a') as hf:
                    hf['spindle_info'].resize((hf['spindle_info'].shape[0] + spindle_info_test.shape[0]), axis = 0)
                    hf['spindle_info'][-spindle_info_test.shape[0]:,:,:,:] = spindle_info_test

                    hf['endeffector_coords'].resize((hf['endeffector_coords'].shape[0] + ee_test.shape[0]), axis = 0)
                    hf['endeffector_coords'][-ee_test.shape[0]:,:,:] = ee_test

                    hf['joint_coords'].resize((hf['joint_coords'].shape[0] + joint_test.shape[0]), axis = 0)
                    hf['joint_coords'][-joint_test.shape[0]:,:,:] = joint_test

                    hf['elbow_coords'].resize((hf['elbow_coords'].shape[0] + elbow_test.shape[0]), axis = 0)
                    hf['elbow_coords'][-elbow_test.shape[0]:,:,:] = elbow_test

                    hf['elbow_vel'].resize((hf['elbow_vel'].shape[0] + elbow_vel_test.shape[0]), axis = 0)
                    hf['elbow_vel'][-elbow_vel_test.shape[0]:,:,:] = elbow_vel_test

                    hf['elbow_acc'].resize((hf['elbow_acc'].shape[0] + elbow_acc_test.shape[0]), axis = 0)
                    hf['elbow_acc'][-elbow_acc_test.shape[0]:,:,:] = elbow_acc_test

                    hf['endeffector_vel'].resize((hf['endeffector_vel'].shape[0] + ee_vel_test.shape[0]), axis = 0)
                    hf['endeffector_vel'][-ee_vel_test.shape[0]:,:,:] = ee_vel_test

                    hf['endeffector_acc'].resize((hf['endeffector_acc'].shape[0] + ee_acc_test.shape[0]), axis = 0)
                    hf['endeffector_acc'][-ee_acc_test.shape[0]:,:,:] = ee_acc_test

                    hf['joint_vel'].resize((hf['joint_vel'].shape[0] + joint_vel_test.shape[0]), axis = 0)
                    hf['joint_vel'][-joint_vel_test.shape[0]:,:,:] = joint_vel_test

                    hf['joint_acc'].resize((hf['joint_acc'].shape[0] + joint_acc_test.shape[0]), axis = 0)
                    hf['joint_acc'][-joint_acc_test.shape[0]:,:,:] = joint_acc_test

                    hf['muscle_acc'].resize((hf['muscle_acc'].shape[0] + muscle_acc_test.shape[0]), axis = 0)
                    hf['muscle_acc'][-muscle_acc_test.shape[0]:,:,:] = muscle_acc_test

                    hf['label'].resize((hf['label'].shape[0] + lab_test.shape[0]), axis = 0)
                    hf['label'][-lab_test.shape[0]:] = lab_test

            count += 1

    mean_all = mean_all / count_mean
    with h5py.File(dataset_path +'dataset_train_' + name_dataset +'.hdf5', 'a') as file:
        file.create_dataset('train_data_mean', data=mean_all)


    all_latents_dict = {'all_latents_train': all_latents_train,
                        'all_latents_test': all_latents_test,
                        'all_latents_val': all_latents_val}

    pickle.dump(all_latents_dict, open(dataset_path + 'latents_' + name_dataset +'.p', 'wb'), protocol=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Proprioceptive Character Recognition dataset')
    parser.add_argument('--monkey_name', type=str, help='Name of the monkey',default='han01_05')
    parser.add_argument('--init_folder', type=int, help='Start of the datapoints folder',default=100)
    parser.add_argument('--end_folder', type=int, help='End of the datapoints folder',default=180)
    parser.add_argument('--lab_count_per_char', type=int, help='How many characters per label per datapoint', default=90)
    parser.add_argument('--step_loop', type=int, help='Step for the loop. If it is 2, it gets only hor or ver based on starting', default=1)
    main(parser.parse_args())