'''Scipt to generate the dataset for `Proprioceptive Character Recognition` task.'''

import os
import random
from collections import namedtuple
import argparse
import pickle
import copy

## Decomment for docker
# os.system('sudo pip install h5py')
import h5py

import numpy as np
from scipy.interpolate import interp1d
import opensim as osim
from pcr_data_utils import make_joint_config_seed, make_muscle_config, compute_jerk
from multiprocessing import Pool
import functools

import sys

sys.path.append('../code/')
from path_utils import PATH_TO_UNPROCESSED_DATA

PATH_TO_TRAJECTORIES = './' # Will be used to save data 
PATH_TO_STARTPOINTS = './start_points'
PATH_TO_MONKEY_ARM = './all_monkey_arm'

def resize(trajectory, size):
    '''Resize the pen-tip trajectory, keeping the velocity profile constant.'''
    true_velocity = np.hstack((np.array([0, 0])[:, None], np.diff(trajectory, axis=1)))
    true_timestamps = np.arange(trajectory.shape[1])
    n_timestamps_new = int(true_timestamps.size*size)
    new_timestamps = np.linspace(0, true_timestamps[-1], n_timestamps_new)

    vel_func = interp1d(true_timestamps, true_velocity)
    new_velocity = vel_func(new_timestamps)
    new_traj = np.cumsum(new_velocity, axis=1) + trajectory[:, 0][:, None]
    return new_traj


def apply_rotations(trajectory, rot, shear_x, shear_y):
    aff = np.array([[1, np.tan(shear_x)], [np.tan(shear_y), 1]])
    aff = aff.dot(np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]))
    return aff.dot(trajectory)


def speedify(trajectory, speed):
    true_timestamps = np.arange(trajectory.shape[1])
    n_timestamps_new = int(true_timestamps.size/speed)
    new_timestamps = np.linspace(0, true_timestamps[-1], n_timestamps_new)

    func = interp1d(true_timestamps, trajectory)
    return func(new_timestamps)


def sample_latent_vars():
    '''Sample all latent transformations to be applied to the given trajectory.

    Returns
    -------
    latents: tuple, (size, rot, shear_x, shear_y, speed, noise)

    '''
    size_set = [0.7, 1., 1.3]
    rot_set = [-np.pi/6, -np.pi/12, 0, np.pi/12, np.pi/6]
    speed_set = [0.8, 1., 1.2, 1.4] #[0.7, 1.4, 2, 4, 6] #[0.8, 1., 1.2, 1.4]
    noise_set = [0, 0.1, 0.3]

    latents = (
        random.choice(size_set),
        *np.random.choice(rot_set, size=3),
        random.choice(speed_set),
        random.choice(noise_set))

    return latents

def par_loop(traj,seed,monkey_name):
    
    traj_sel = traj
    seed = seed
    monkey_name = monkey_name

    with h5py.File(os.path.join(PATH_TO_STARTPOINTS, 'pcr_startingpoints_'+monkey_name+'_scaled_5.hdf5'), 'r') as myfile:  #kibleur3_grid2
        startpts = myfile['horizontal'][()]
    
    ## Set the seed to have the same default position for horizontal and vertical in the datapoint
    rng = np.random.RandomState(seed)
    min_angles = np.array([-50, -45, -90, 0])
    max_angles = np.array([180, 150, 90, 140])

    if monkey_name == 'snap':
        model = osim.Model(os.path.join(PATH_TO_MONKEY_ARM,'Snap','Snap_scaled_fin1.osim'))
    elif monkey_name == 'butter':
        model = osim.Model(os.path.join(PATH_TO_MONKEY_ARM,'Butter','ButterScaledArm_ale.osim'))
    elif monkey_name == 'lando':
        model = osim.Model(os.path.join(PATH_TO_MONKEY_ARM,'Lando','LandoScaledArm_ale.osim'))
    elif monkey_name == 'han_01_05':
        model = osim.Model(os.path.join(PATH_TO_MONKEY_ARM,'Han','HanScaledArm20170105_ale.osim'))
    elif monkey_name == 'han_11_22':
        model = osim.Model(os.path.join(PATH_TO_MONKEY_ARM,'Han','HanScaledArm20171122.osim'))
    elif monkey_name == 'chips':
        model = osim.Model(os.path.join(PATH_TO_MONKEY_ARM,'Chips','ChipsScaledArm_ale.osim'))

    # Aligning pen-tip trajectories to {W}
    plane_to_world = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    
    Latents = namedtuple('Latents', ('size', 'rot', 'shear_x', 'shear_y', 'speed', 'noise'))
    traj_sel = traj_sel[:, np.all(~np.isnan(traj_sel), axis=0)]

    tracing_error = 1e16
    joint_jerk = 1e16
    muscle_jerk = 1e16

    while (tracing_error > 1e-2) or (joint_jerk > 2) or (muscle_jerk > 4):
        latent_vars = Latents(*sample_latent_vars())
        mytraj = resize(traj_sel, latent_vars.size)
        mytraj = apply_rotations(mytraj, latent_vars.rot, latent_vars.shear_x, latent_vars.shear_y)
        mytraj = speedify(mytraj, latent_vars.speed)
        mytraj = np.insert(mytraj, 2, 0, axis=0)

        endeffector_coordinates = plane_to_world.dot(mytraj)
        startingpoint = random.choice(startpts)
        endeffector_coordinates += startingpoint[:, None]

        q = np.array([rng.uniform(min_angles[0], max_angles[0]), 
                      rng.uniform(min_angles[1], max_angles[1]),
                      rng.uniform(min_angles[2], max_angles[2]),
                      rng.uniform(min_angles[3], max_angles[3])])
        q0 = q

        joint_coordinates, tracing_error = make_joint_config_seed(endeffector_coordinates, q = q, q0 = q, monkey_name = monkey_name)
        joint_jerk = compute_jerk(joint_coordinates)

        muscle_coordinates, marker3, marker6 = make_muscle_config(model, joint_coordinates)
        muscle_jerk = compute_jerk(muscle_coordinates)

        # Save the datapoint
        datapoint = {
            'plane': 'horizontal',
            'monkey': monkey_name,
            'startpt': startingpoint,
            'endeffector_coords': marker3,
            'marker6': marker6,
            'joint_coords': joint_coordinates,
            'muscle_coords': muscle_coordinates,
            'joint_jerk': joint_jerk,
            'muscle_jerk': muscle_jerk,
            'latents': {'size': latent_vars[0], 'rot': latent_vars[1], 'shear_x': latent_vars[2],
                        'shear_y': latent_vars[3], 'speed': latent_vars[4], 'noise': latent_vars[5],'seed': seed}}

    return datapoint

def main(args):
    '''Generate label specific joint angle and muscle length trajectories.

    '''

    ll = args.plane #, args.label]
    monkey_name = args.monkey_name

    # Load character trajectories and starting point data
    with h5py.File(os.path.join(PATH_TO_TRAJECTORIES, 'pcr_trajectories_5.hdf5'), 'r') as myfile:
        trajectories = myfile['trajectories'][()]
        labels = myfile['labels'][()]

    char_idx = labels == args.label
    char_trajectories = trajectories[char_idx]

    char_data = []

    path_data = os.path.join(PATH_TO_UNPROCESSED_DATA, 'unprocessed_data_' + monkey_name)
    path_data = os.path.join(path_data, str(args.folder))
    if not os.path.isdir(path_data):
        os.makedirs(path_data)

    seed = np.full((len(char_trajectories)),args.seed)
    monkey_name_all = np.full((len(char_trajectories)),monkey_name)

    list_traj_seed = zip(char_trajectories,seed,monkey_name_all)

    pool = Pool()
    result = pool.starmap(par_loop, list_traj_seed)
    for rr in result:
        rr.update({'label': args.label})
    char_data.append(copy.copy(result))

    pickle.dump(char_data, open(os.path.join(path_data, '{}.p'.format(args.name)), 'wb'), protocol=4)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Proprioceptive Character Recognition dataset')
    parser.add_argument('--label', type=int, help='Character label')
    parser.add_argument('--plane', type=str, help='Plane of writing {horizontal, vertical}')
    parser.add_argument('--seed', type=int, help='Seed for selecting the default position of the arm')
    parser.add_argument('--monkey_name', type=str, help='Monkey name')
    parser.add_argument('name', type=int, help='Job id for generating dataset')
    parser.add_argument('folder', type=int, help='folder to save the sample')
    Latents = namedtuple('Latents', ('size', 'rot', 'shear_x', 'shear_y', 'speed', 'noise'))
    main(parser.parse_args())
