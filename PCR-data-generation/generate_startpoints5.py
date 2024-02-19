"""Scipt to generate candidate starting points in both vertical and horizontal planes for writing the
characters.

"""
import os
import argparse
import itertools
import h5py
import numpy as np
from joblib import delayed, Parallel

from pcr_data_utils import Arm
import sys

sys.path.append('../code/')
from path_utils import PATH_TO_STARTPOINT

def isreachable_point(target_xyz,monkey_name):
    myarm = Arm(monkey_name=monkey_name)
    if np.linalg.norm(target_xyz) < np.sum(myarm.L):
        shoulder_to_world = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        target_xyz = (shoulder_to_world.T).dot(np.array(target_xyz)[:, None]).T # Same point in {S}
        angles = myarm.inv_kin(target_xyz)
        return np.linalg.norm(target_xyz - myarm.get_xyz(q=angles)[0])
    else:
        return 1
    return 

def coarse_startpt_search(workspace, size, grid, grid_resolution, dim):
    '''Run a coarse grained search over the workspace to find candidate starting points, that is,
    the set of all reachable points for which [NE, NW, SE, SW] corners of a (size x size) square
    are also reachable.

    Arguments
    ---------
    workspace: np.ndarray, [N, 3] array of all reachable points of the Arm
    size: float, size of the bounding box for all characters [usually, 5x5 here]
    grid: np.array, levels of the grid used in generating the workspace
    grid_resolution: int, resolution used for generating the workspace
    dim: int, 2 if horizontal trajectory [z-plane], 0 if vertical [x-plane]

    Returns
    -------
    candidate_startpts: np.ndarray, candidate starting points after coarse search

    '''
    candidates = []
    for plane in grid:
        plane_mask = np.array((workspace[:, dim] == plane))
        candidates_inplane = workspace[plane_mask]
        candidates.append(search_inplane(candidates_inplane, size, grid_resolution, dim))
    return np.concatenate(candidates)


def search_inplane(candidates, size, grid_resolution, dim):
    temp_mask = np.array([True, True, False]) if dim == 2 else np.array([False, True, True])
    planar_workspace = candidates[:, temp_mask].tolist()
    neighbors = [generate_neighbors(pt, size, grid_resolution) for pt in planar_workspace]
    positives = []
    for pt in range(len(planar_workspace)):
        bools = [corner.tolist() in planar_workspace for corner in neighbors[pt]]
        positives.append(all(bools))
    valid_candidates = candidates[positives]
    return valid_candidates


def generate_neighbors(point, size, grid_resolution):
    point = np.array(point)
    n = grid_resolution*np.ceil(size/(2*grid_resolution))
    neast = point + [n, n]
    nwest = point + [-n, n]
    seast = point + [n, -n]
    swest = point + [-n, -n]
    return [neast, nwest, seast, swest]


def fine_startpt_search(candidates, monkey_name, size, dim):
    '''Run a fine-grained search over candidate starting points by checking reachability of a fine
    grid of points surrounding each candidate starting point.

    '''
    errors = []
    if size % 2 ==1:
        one_grid_size = (size //2) +1
    else:
        one_grid_size = size //2
    # one_grid_size = size - (size // 3)
    for start_point in candidates:
        grid = make_grid(start_point, one_grid_size, dim)
        errors.append(isreachable_grid(grid,monkey_name))
    errors = np.array(errors)
    valid_candidates_idx = np.sum(errors < 1e-2, axis=1) == (one_grid_size*2)**2  #(look at make_grid, size of the square)
    return candidates[valid_candidates_idx]


def make_grid(start_point, one_grid_size, dim):
    '''Make a grid around a starting point, in either vertical or horizontal planes. As a somewhat
    arbitrary choice, the grid is 8x8 (square is 5x5 of the traj x Monkey) (10x10 traj human and 15x15 grid).

    '''
    x, y, z = start_point
    if dim == 0:
        z, x, y = start_point
    
    xmin, xmax = x - one_grid_size, x + one_grid_size  #Should be 4
    ymin, ymax = y - one_grid_size, y + one_grid_size
    X, Y = np.mgrid[xmin:xmax, ymin:ymax]
    grid_points = np.vstack([X.ravel(), Y.ravel()])
    grid_points_3d = np.insert(grid_points, dim, z, axis=0)
    return grid_points_3d


def isreachable_grid(grid,monkey_name):
    errors = Parallel(n_jobs=24)(delayed(isreachable_point)(grid[:, i],monkey_name) for i in range(grid.shape[1]))
    return errors


def main(args):

    monkey_name = args.monkey_name

    # Define search space for the workspace
    grid_size = 5
    grid_resolution = 2   # in cm
    xmax, xmin = 38, 0   # forward and backward
    ymax, ymin = 38, -38  # left and right
    zmax, zmin = 38, -38  # up and down

    X = np.linspace(xmin, xmax, (xmax - xmin)//grid_resolution + 1)
    Y = np.linspace(ymin, ymax, (ymax - ymin)//grid_resolution + 1)
    Z = np.linspace(zmin, zmax, (zmax - zmin)//grid_resolution + 1)

    # Compute whether each point in (X x Y x Z) is reachable
    errors = Parallel(n_jobs=24)(delayed(isreachable_point)(i,monkey_name) for i in itertools.product(X, Y, Z))

    points = np.array(list(itertools.product(X, Y, Z)))
    errors = np.array(errors)
    workspace = points[errors < 1e-2]

    # Run coarse-grained search for candidate starting points
    if grid_size % 2 == 1: grid_size1 = grid_size -1 

    startpoints_horizontal = coarse_startpt_search(workspace, grid_size1, Z, grid_resolution, 2) #8
    startpoints_vertical = coarse_startpt_search(workspace, grid_size1, X, grid_resolution, 0)


    # Run fine-grained search for reduced set of candidate starting points
    startpoints_horizontal = fine_startpt_search(startpoints_horizontal, monkey_name, grid_size, 2)
    startpoints_vertical = fine_startpt_search(startpoints_vertical, monkey_name, grid_size, 0)

    with h5py.File(os.path.join(PATH_TO_STARTPOINT, 'pcr_startingpoints_'+monkey_name+'_scaled_5.hdf5'), 'w') as file:
        file.create_dataset('horizontal', data=startpoints_horizontal)
        file.create_dataset('vertical', data=startpoints_vertical)

    return

if __name__ == "__main__":
    # main()
    parser = argparse.ArgumentParser(description='Generate starting point to augment trajectories')
    parser.add_argument('--monkey_name', type=str, help='Name of the monkey (snap, butter, han01_05, han_11_22, chips)',default='snap')
    main(parser.parse_args())
