"""
Scipt to generate the dataset for `Proprioceptive Character Recognition` task using a monkey arm.

"""
import numpy as np
import opensim as osim
import scipy.optimize
import math as m

class Arm:

    def __init__(self, q=None, q0=None, L=None, monkey_name=None):

        # Initial joint angles and default arm position (last position in the traj) - Chan&Moran scaled
        self.q = np.array([36, 5, 5, 50]) if q is None else q
        self.q0 = np.array([36, 5, 5, 50]) if q0 is None else q0
        
        # Lengths of humerus and radius (in cm).
        if monkey_name == 'snap':
            ## Monkey length Snap scaled
            self.L = np.array([16.1, 18.6]) if L is None else L   #16, 17.58   #_plus_len 17.2, 18.6  ## Snap_scaled fin1: 16.3  18.7
        elif monkey_name == 'butter':
            ## Monkey length Butter scaled
            self.L = np.array([12.58, 16]) if L is None else L
        elif monkey_name == 'han_01_05':
            ## Monkey length Han_01_05 scaled
            self.L = np.array([16.6, 19.5]) if L is None else L
        elif monkey_name == 'han_11_22':
            ## Monkey length Han_11_22 scaled
            self.L = np.array([15, 18.5]) if L is None else L
        elif monkey_name == 'chips':
            ## Monkey length Chips scaled
            self.L = np.array([17.2, 22.1]) if L is None else L

        ## Joint angles (Euler):
        self.min_angles = np.array([-50, -45, -90, 0])
        self.max_angles = np.array([180, 150, 90, 140])

    def get_xyz(self, q=None):
        """Implements forward kinematics:
        Returns the end-effector coordinates (euclidean) for a given set of joint
        angle values.
        Inputs :
        Returns :
        """
        if q is None:
            q = self.q

        # Define rotation matrices about the shoulder and elbow.
        # Translations for the shoulder frame will be introduced later.
        def shoulder_rotation(shoulder_flexion, shoulder_adduction, shoulder_rot):
            return rotz(shoulder_adduction).dot(roty(shoulder_rot)).dot(rotx(shoulder_flexion))

        def elbow_rotation(elbow_flexion):
            return rotx(elbow_flexion)

        # Unpack variables
        shoulder_flexion, shoulder_adduction, shoulder_rot, elbow_flexion = q
        upperarmgth, forearmgth = self.L

        # Define initial joint locations:
        origin = np.array([0, 0, 0])
        elbow = np.array([0, -upperarmgth, 0])
        hand = np.array([0, -forearmgth, 0])

        new_elbow_loc = shoulder_rotation(
            shoulder_flexion, shoulder_adduction, shoulder_rot).dot(elbow)
        
        new_hand_loc = shoulder_rotation(shoulder_flexion, shoulder_adduction, shoulder_rot)\
            .dot(elbow_rotation(elbow_flexion).dot(hand) + elbow)

        link_pos = np.column_stack((origin, new_elbow_loc, new_hand_loc))

        return new_hand_loc, link_pos

    def inv_kin(self, xyz):
        """Implements inverse kinematics:
        Given an xyz position of the hand, return a set of joint angles (q)
        using constraint based minimization. Constraint is to match hand xyz and
        minimize the distance of each joint from it's default position (q0).
        Inputs :
        Returns :
        """

        def distance_to_default(q, *args):
            return np.linalg.norm(q - np.asarray(self.q0))*5

        def pos_constraint(q, xyz):
            return np.linalg.norm(self.get_xyz(q=q)[0] - xyz)

        def joint_limits_upper_constraint(q, xyz):
            return self.max_angles - q

        def joint_limits_lower_constraint(q, xyz):
            return q - self.min_angles

        return scipy.optimize.fmin_slsqp(
            func=distance_to_default, x0=self.q, eqcons=[pos_constraint],
            ieqcons=[joint_limits_upper_constraint, joint_limits_lower_constraint],
            args=(xyz,), iprint=0)


# Auxiliary function definitions:
# Define rotation matrices which take angle inputs in degrees.
def rotx(angle):
    angle = angle*np.pi/180
    return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])


def roty(angle):
    angle = angle*np.pi/180
    return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])


def rotz(angle):
    angle = angle*np.pi/180
    return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])


# Inverse kinematics and muscle length generation utilities

def make_joint_config(trajectory, monkey_name):
    """Compute joint configurations of a 4Dof Arm given end-effector trajectory in {W}.

    Returns
    -------
    joint_coordinates: np.array, 4 joint angles at each of T time points
    error: np.float, error in tracing out the given end-effector trajectory

    """
    myarm = Arm(monkey_name = monkey_name)
    shoulder_to_world = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])

    # Inverse Kinematics to obtain joint configurations for the character trajectory
    # Project character trajectories into {S}
    char_traj_s = (shoulder_to_world.T).dot(trajectory)
    trajgth = char_traj_s.shape[1]
    joint_trajectory = np.zeros((4, trajgth))

    # For each point in the trajectory derive the joint angle configuration
    # After finding the joint configuration for a particular point, change q0
    # to the current joint configuration!
    error = 0.
    for i in range(trajgth):
        dest = char_traj_s[:, i]
        joint_config = myarm.inv_kin(xyz=dest)
        myarm.q0 = joint_config
        joint_trajectory[:, i] = joint_config
        error += np.linalg.norm(dest - myarm.get_xyz(joint_config)[0])

    return joint_trajectory, error

def make_joint_config_seed(trajectory, q, q0, monkey_name):
    """Compute joint configurations of a 4Dof Arm given end-effector trajectory in {W}.

    Returns
    -------
    joint_coordinates: np.array, 4 joint angles at each of T time points
    error: np.float, error in tracing out the given end-effector trajectory

    """
    myarm = Arm(q =q, q0 = q0, monkey_name = monkey_name)
    shoulder_to_world = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])

    # Inverse Kinematics to obtain joint configurations for the character trajectory
    # Project character trajectories into {S}
    char_traj_s = (shoulder_to_world.T).dot(trajectory)
    trajgth = char_traj_s.shape[1]
    joint_trajectory = np.zeros((4, trajgth))

    # For each point in the trajectory derive the joint angle configuration
    # After finding the joint configuration for a particular point, change q0
    # to the current joint configuration!
    error = 0.
    for i in range(trajgth):
        dest = char_traj_s[:, i]
        joint_config = myarm.inv_kin(xyz=dest)
        myarm.q0 = joint_config
        joint_trajectory[:, i] = joint_config
        error += np.linalg.norm(dest - myarm.get_xyz(joint_config)[0])

    return joint_trajectory, error


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


def make_muscle_config(model, joint_trajectory):
    """Compute muscle configurations of a given opensim model to given coordinate (joint angle) trajectories.

    Arguments
    ---------
    model : opensim model object, the MoBL Dynamic Arm.
    joint_trajectory : np.array, shape=(4, T) joint angles at each of T time points

    Returns
    -------
    musclegth_configurations : np.array, shape=(39, T) muscle lengths at each of T time points

    """
    init_state = model.initSystem()
    model.equilibrateMuscles(init_state)

    # Name of muscles
    muscleSet = model.getMuscles()
    muscles_name = []
    for muscle in muscleSet:
        muscles_name.append(muscle.getName())

    ### Change the sign of the shoulder adduction. It is necessary to adapt the 2-link arm to the Opensim arm
    sign_axis = np.array([[1, -1, 1, 1]]).reshape(4,1)
    joint_trajectory = joint_trajectory*sign_axis

    # Prepare for simulation
    num_coords, num_timepoints = joint_trajectory.shape
    muscle_set = model.getMuscles() # returns a Set<Muscles> object
    num_muscles = muscle_set.getSize()

    markers_names_all = ['Shoulder_JC', 'Marker_6', 'Pronation_Pt1', 'Marker_5', 'Marker_4', 'Marker_3', 'Marker_2', 'Marker_1']

    names_marker_osim = []
    for ii in range(model.getMarkerSet().getSize()):
        names_marker_osim.append(model.getMarkerSet().get(ii).getName())

    ind_marker3 = names_marker_osim.index(markers_names_all[5])
    ind_marker6 = names_marker_osim.index(markers_names_all[1])

    def Rx(theta):
        return np.matrix([[ 1, 0           , 0           ],
                    [ 0, m.cos(theta),-m.sin(theta)],
                    [ 0, m.sin(theta), m.cos(theta)]])

    R_rot = Rx(m.pi/2)
    marker3 = np.zeros((3,num_timepoints))
    marker6= np.zeros((3,num_timepoints))

    # Set the order of the coordinates
    coord_order = ['shoulder_flexion', 'shoulder_adduction', 'shoulder_rotation', 'elbow_flexion']
    # For each time step of the trajectory, compute equibrium muscle states
    # Create a dictionary of muscle configurations
    muscle_config = {}
    for i in range(num_muscles):
        muscle_config[muscle_set.get(i).getName()] = np.zeros(num_timepoints)

    for timepoint in range(num_timepoints):
        for i in range(num_coords):
            model.getCoordinateSet().get(coord_order[i]).\
            setValue(init_state, np.pi*(joint_trajectory[i, timepoint] / 180))
        model.equilibrateMuscles(init_state)
        marker3[:,timepoint] = R_rot.dot(np.array([model.getMarkerSet().get(ind_marker3).getLocationInGround(init_state)[jj] for jj in range(3)]))*100
        marker6[:,timepoint] = R_rot.dot(np.array([model.getMarkerSet().get(ind_marker6).getLocationInGround(init_state)[jj] for jj in range(3)]))*100

        for muscle_num in range(num_muscles):
            muscle = muscle_set.get(muscle_num)
            name = muscle.getName()

            tendon_len = muscle.getTendonLength(init_state)*1000
            muscle_config[name][timepoint] = muscle.getFiberLength(init_state)*1000 + tendon_len # change to mm

    for i in list(muscle_config.keys()):
        if not (i in muscles_name):
            del muscle_config[i]

    mconf = [muscle_config[i] for i in muscles_name]
    musclegth_configurations = np.asarray(mconf)

    return musclegth_configurations, marker3, marker6


def make_muscle_matrix(muscle_config):
    # Arrange muscles configurations in a 25xT matrix, given a dictionary of muscle configurations.
    order = ['CORB', 'DELT1', 'DELT2', 'DELT3', 'INFSP', 'LAT1', 'LAT2', 'LAT3', 'PECM1',
             'PECM2', 'PECM3', 'SUBSC', 'SUPSP', 'TMAJ', 'TMIN', 'ANC', 'BIClong', 'BICshort',
             'BRA', 'BRD', 'ECRL', 'PT', 'TRIlat', 'TRIlong', 'TRImed']
    mconf = [muscle_config[i] for i in order]
    return np.asarray(mconf)

def signpow(a,b): return np.sign(a)*(np.abs(a)**b)

def make_spindle_coords(muscle_traj):
    stretch = np.gradient(muscle_traj, 1, axis=1)
    stretch_vel = np.gradient(muscle_traj, 0.015, axis=1)
    p_rate = 2*stretch + 4.3*signpow(stretch_vel, 0.6)
    return p_rate

def start_end_choice(traj):
    true_traj = traj[:, np.all(~np.isnan(traj), axis=0)]
    room = 320 - true_traj.shape[1]
    start_idx = np.random.randint(room)
    end_idx = start_idx + true_traj.shape[1]
    return start_idx, end_idx

def apply_shuffle(traj, start_idx, end_idx):
    true_traj = traj[:, np.all(~np.isnan(traj), axis=0)]
    mytraj = np.zeros((true_traj.shape[0], 320))
    mytraj[:, start_idx:end_idx] = true_traj
    mytraj[:, :start_idx] = true_traj[:, 0][:, None]
    mytraj[:, end_idx:] = true_traj[:, -1][:, None]
    return mytraj

def add_noise(mconf, factor):
    noisy_mconf = mconf + factor*mconf.std(axis=1)[:, None]*np.random.randn(*mconf.shape)
    return noisy_mconf