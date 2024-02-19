import numpy as np 
import opensim as osim
import os
import math as m
from multiprocessing import Pool

def generate_trc_files(markers_all,markers_names, time_step, trc_folder):
    """ Generate .trc files from markers in numpy
    """
    if not os.path.exists(trc_folder):
        os.makedirs(trc_folder)

    for ii in range(markers_all.shape[0]):
        ## Generate time vector
        time = np.arange(markers_all[ii].shape[0])*time_step #+ 3.55300000
        time = time[...,np.newaxis] 

        data = np.concatenate((time,markers_all[ii]),1)
        numpy_to_trc(data, markers_names, trc_folder + '/trial_' + str(ii) + '.trc')
    return

def generate_sto_files(model_path, input_DIR, output_DIR):
        """ Run IK and store results in .sto file
        """
        n_files = len([name for name in os.listdir(input_DIR) if os.path.isfile(os.path.join(input_DIR, name))])
        mymodel = osim.Model(model_path) 
        markers_names = [marker.getName() for marker in mymodel.getMarkerSet()]

        # Create output folder
        if not os.path.exists(output_DIR):
            os.makedirs(output_DIR)

        ## Define weight for each marker
        task = form_task_set(markers_names, [1] * len(markers_names))

        # Experimental data
        for ii in range(n_files):
            # Data is a 2D numpy array
            # First column contains time indices,
            # and subsequent columns are X1, Y1, Z1, X2, Y2, Z2...
            trc_name = input_DIR + '/trial_' + str(ii)
            out_name = output_DIR + '/trial_' + str(ii) + '.sto'
            run_marker_ik(mymodel, trc_name, task, output_name=out_name)
        return

def generate_sto_files_pool(model_path, input_DIR, output_DIR):
        """ Run IK and store results in .sto file
        """
        n_files = len([name for name in os.listdir(input_DIR) if os.path.isfile(os.path.join(input_DIR, name))])
        # mymodel = osim.Model(model_path) 

        # Create output folder
        if not os.path.exists(output_DIR):
            os.makedirs(output_DIR)

        all_files = [[model_path,input_DIR + '/trial_' + str(ii),output_DIR + '/trial_' + str(ii) + '.sto'] for ii in range(n_files)]
    
        p = Pool()
        
        result = p.map(run_marker_ik_pool, all_files)
        p.close()
        p.join()

        return


def get_data_ready(trial):
    '''trial is the second level of data_dict- action-instance_XX
    transforms trial in np 2D array rows = time points, columns = X,Y,Z position for each data_marker
    '''
    frequency = 100;
    time = np.arange(0,trial['PALM'].shape[0],1)/frequency
    to_fill = time[np.newaxis, :]
    for count, key in enumerate(trial.keys()):
        
        trial[key] = rotate(trial[key])
        if count == 0:
            traj_table = np.concatenate([to_fill.T,trial[key]],1)
        else:
            traj_table = np.concatenate([traj_table,trial[key]],1)
    return traj_table

def run_ik(model, data, marker_names, output_name):
    task = form_task_set(marker_names, [1] * len(marker_names))
    # Data is a 2D numpy array
    # First column contains time indices,
    # and subsequent columns are X1, Y1, Z1, X2, Y2, Z2...
    numpy_to_trc(data, marker_names, output_name) #markers
    run_marker_ik(model, output_name, task)
    return

def form_task_set(markers, weights):
    if len(markers) != len(weights):
        raise ValueError('Number of markers and weights must match')
    task_set = osim.IKTaskSet()
    for marker, weight in zip(markers, weights):
        if marker:
            task = osim.IKMarkerTask()
            task.setName(marker)
            task.setWeight(weight)
            if not weight:
                task.setApply(False)
            task_set.cloneAndAppend(task)
    # coordinates = ['r_x','r_y','r_z']
    
    # for coordinate in coordinates:
    #     coo = osim.IKCoordinateTask()
    #     coo.setName(coordinate)        
    #     coo.setWeight(1)
    #     task_set.cloneAndAppend(coo)    
    
    return task_set

def numpy_to_trc(data, labels, trcfile):
    time = data[:, 0]
    freq = int(1 / (time[1] - time[0]))
    nframes = data.shape[0]
    frames = np.arange(nframes)
    markers = data[:, 1:]
    # data are in cm --> transform to mm
    markers *= 1000
    nmarkers = markers.shape[1] // 3
    header = '\n'.join(
        ['\t'.join('PathFileType 4 (X/Y/Z) {}'.format(trcfile).split()),
         '\t'.join('DataRate CameraRate NumFrames NumMarkers Units '
                   'OrigDataRate OrigDataStartFrame OrigNumFrames'.split()),
         '\t'.join('{} {} {} {} mm {} 1 {}'.format(freq, freq, nframes, nmarkers, freq, nframes).split()),
         'Frame#\tTime\t' + '\t\t\t'.join(labels),
         '\t\t' + '\t'.join(['X{}\tY{}\tZ{}'.format(i, i, i) for i in range(1, nmarkers + 1)])]) + '\n'
    kine = np.c_[frames, time, markers]
    with open(trcfile, 'w') as output:
        output.write(header)
        np.savetxt(output, kine, delimiter='\t', fmt=['%05d', '%.3f'] + nmarkers * 3 * ['%.3f'])
    return

def run_marker_ik_pool(inputs):
    """

    :param model_name: str
        Path to osim model
    :param dynamic_data: str
        Path to a TRC file NO extension
    :param task_set: osim.tools.IKTaskSet
        Tasks to be solved by Inverse Kinematics
    :param time_range: tuple of float, optional (default=None)
        Desired frames as (start, end) time to be averaged
    :return: path to the Storage file
    :rtype: str
    """
    model_name, dynamic_data, output_name = inputs
    time_range = None

    model = osim.Model(model_name)

    markers_names = [marker.getName() for marker in model.getMarkerSet()]
    task_set = form_task_set(markers_names, [1] * len(markers_names))

    ik_tool = osim.InverseKinematicsTool()
    ik_tool.setModel(model)
    ik_tool.setMarkerDataFileName(dynamic_data + '.trc')
    if not output_name:
        output_name = dynamic_data+'.sto'
    ik_tool.setOutputMotionFileName(output_name)

    if time_range is not None:
        start, end = time_range
        ik_tool.setStartTime(start)
        ik_tool.setEndTime(end)

    ik = ik_tool.getIKTaskSet()
    for task in task_set:
        ik.cloneAndAppend(task)

    # uncomment to save the IK setting file ( .xml)
   
    #ik_tool.printToXML(dynamic_data+'.xml')

    
    ik_tool.run()
    return 

def run_marker_ik(model_name, dynamic_data, task_set, time_range=None, output_name=None):
    """

    :param model_name: str
        Path to osim model
    :param dynamic_data: str
        Path to a TRC file NO extension
    :param task_set: osim.tools.IKTaskSet
        Tasks to be solved by Inverse Kinematics
    :param time_range: tuple of float, optional (default=None)
        Desired frames as (start, end) time to be averaged
    :return: path to the Storage file
    :rtype: str
    """
    model = osim.Model(model_name)
    ik_tool = osim.InverseKinematicsTool()
    ik_tool.setModel(model)
    ik_tool.setMarkerDataFileName(dynamic_data + '.trc')
    if not output_name:
        output_name = dynamic_data+'.sto'
    ik_tool.setOutputMotionFileName(output_name)

    if time_range is not None:
        start, end = time_range
        ik_tool.setStartTime(start)
        ik_tool.setEndTime(end)

    ik = ik_tool.getIKTaskSet()
    for task in task_set:
        ik.cloneAndAppend(task)

    # uncomment to save the IK setting file ( .xml)
   
    #ik_tool.printToXML(dynamic_data+'.xml')

    
    ik_tool.run()
    return 


## osim interface
# The amin purpose of this class is to provide wrap all 
# the necessery elements of osim in one place
# The actual RL environment then only needs to:
# - open a model
# - actuate
# - integrate
# - read the high level description of the state
# The objective, stop condition, and other gym-related
# methods are enclosed in the OsimEnv class
class OsimModel(object):
    # Initialize simulation
    stepsize = 0.01

    model = None
    state = None
    state0 = None
    joints = []
    bodies = []
    brain = None
    verbose = False
    istep = 0
    
    state_desc_istep = None
    prev_state_desc = None
    state_desc = None
    integrator_accuracy = None

    maxforces = []
    curforces = []

    def __init__(self, model_path, visualize, integrator_accuracy = 5e-5):  #5e-5
        self.integrator_accuracy = integrator_accuracy
        self.model = osim.Model(model_path)
        self.model_state = self.model.initSystem()
        self.brain = osim.PrescribedController()

        # Enable the visualizer
        self.model.setUseVisualizer(visualize)

        self.muscleSet = self.model.getMuscles()
        self.forceSet = self.model.getForceSet()
        self.bodySet = self.model.getBodySet()
        self.jointSet = self.model.getJointSet()
        self.markerSet = self.model.getMarkerSet()
        self.contactGeometrySet = self.model.getContactGeometrySet()

        if self.verbose:
            self.list_elements()

        # Add actuators as constant functions. Then, during simulations
        # we will change levels of constants.
        # One actuartor per each muscle
        for j in range(self.muscleSet.getSize()):
            func = osim.Constant(1.0)
            self.brain.addActuator(self.muscleSet.get(j))
            self.brain.prescribeControlForActuator(j, func)

            self.maxforces.append(self.muscleSet.get(j).getMaxIsometricForce())
            self.curforces.append(1.0)

        self.noutput = self.muscleSet.getSize()
            
        self.model.addController(self.brain)
        self.model_state = self.model.initSystem()

    def list_elements(self):
        print("JOINTS")
        for i in range(self.jointSet.getSize()):
            print(i,self.jointSet.get(i).getName())
        print("\nBODIES")
        for i in range(self.bodySet.getSize()):
            print(i,self.bodySet.get(i).getName())
        print("\nMUSCLES")
        for i in range(self.muscleSet.getSize()):
            print(i,self.muscleSet.get(i).getName())
        print("\nFORCES")
        for i in range(self.forceSet.getSize()):
            print(i,self.forceSet.get(i).getName())
        print("\nMARKERS")
        for i in range(self.markerSet.getSize()):
            print(i,self.markerSet.get(i).getName())

    def actuate(self, action):
        if np.any(np.isnan(action)):
            raise ValueError("NaN passed in the activation vector. Values in [0,1] interval are required.")

        action = np.clip(np.array(action), 0.0, 1.0)
        self.last_action = action
            
        brain = osim.PrescribedController.safeDownCast(self.model.getControllerSet().get(0))
        functionSet = brain.get_ControlFunctions()

        for j in range(functionSet.getSize()):
            func = osim.Constant.safeDownCast(functionSet.get(j))
            func.setValue( float(action[j]) )

    """
    Directly modifies activations in the current state.
    """
    def set_activations(self, activations):
        if np.any(np.isnan(activations)):
            raise ValueError("NaN passed in the activation vector. Values in [0,1] interval are required.")
        for j in range(self.muscleSet.getSize()):
            self.muscleSet.get(j).setActivation(self.state, activations[j])
        self.reset_manager()

    """
    Get activations in the given state.
    """
    def get_activations(self):
        return [self.muscleSet.get(j).getActivation(self.state) for j in range(self.muscleSet.getSize())]

    def compute_state_desc(self):
        self.model.realizeAcceleration(self.state)

        res = {}

        ## Joints
        res["joint_pos"] = {}
        res["joint_vel"] = {}
        res["joint_acc"] = {}
        for i in range(self.jointSet.getSize()):
            joint = self.jointSet.get(i)
            name = joint.getName()
            res["joint_pos"][name] = [joint.get_coordinates(i).getValue(self.state) for i in range(joint.numCoordinates())]
            res["joint_vel"][name] = [joint.get_coordinates(i).getSpeedValue(self.state) for i in range(joint.numCoordinates())]
            res["joint_acc"][name] = [joint.get_coordinates(i).getAccelerationValue(self.state) for i in range(joint.numCoordinates())]

        ## Bodies
        res["body_pos"] = {}
        res["body_vel"] = {}
        res["body_acc"] = {}
        res["body_pos_rot"] = {}
        res["body_vel_rot"] = {}
        res["body_acc_rot"] = {}
        for i in range(self.bodySet.getSize()):
            body = self.bodySet.get(i)
            name = body.getName()
            res["body_pos"][name] = [body.getTransformInGround(self.state).p()[i] for i in range(3)]
            res["body_vel"][name] = [body.getVelocityInGround(self.state).get(1).get(i) for i in range(3)]
            res["body_acc"][name] = [body.getAccelerationInGround(self.state).get(1).get(i) for i in range(3)]
            
            res["body_pos_rot"][name] = [body.getTransformInGround(self.state).R().convertRotationToBodyFixedXYZ().get(i) for i in range(3)]
            res["body_vel_rot"][name] = [body.getVelocityInGround(self.state).get(0).get(i) for i in range(3)]
            res["body_acc_rot"][name] = [body.getAccelerationInGround(self.state).get(0).get(i) for i in range(3)]

        ## Forces
        res["forces"] = {}
        for i in range(self.forceSet.getSize()):
            force = self.forceSet.get(i)
            name = force.getName()
            values = force.getRecordValues(self.state)
            res["forces"][name] = [values.get(i) for i in range(values.size())]

        ## Muscles
        res["muscles"] = {}
        for i in range(self.muscleSet.getSize()):
            muscle = self.muscleSet.get(i)
            name = muscle.getName()
            res["muscles"][name] = {}
            res["muscles"][name]["activation"] = muscle.getActivation(self.state)
            res["muscles"][name]["fiber_length"] = muscle.getFiberLength(self.state)
            res["muscles"][name]["fiber_velocity"] = muscle.getFiberVelocity(self.state)
            res["muscles"][name]["fiber_force"] = muscle.getFiberForce(self.state)
            # We can get more properties from here http://myosin.sourceforge.net/2125/classosim_1_1Muscle.html 
        
        ## Markers
        res["markers"] = {}
        for i in range(self.markerSet.getSize()):
            marker = self.markerSet.get(i)
            name = marker.getName()
            res["markers"][name] = {}
            res["markers"][name]["pos"] = [marker.getLocationInGround(self.state)[i] for i in range(3)]
            res["markers"][name]["vel"] = [marker.getVelocityInGround(self.state)[i] for i in range(3)]
            res["markers"][name]["acc"] = [marker.getAccelerationInGround(self.state)[i] for i in range(3)]

        ## Other
        res["misc"] = {}
        res["misc"]["mass_center_pos"] = [self.model.calcMassCenterPosition(self.state)[i] for i in range(3)]
        res["misc"]["mass_center_vel"] = [self.model.calcMassCenterVelocity(self.state)[i] for i in range(3)]
        res["misc"]["mass_center_acc"] = [self.model.calcMassCenterAcceleration(self.state)[i] for i in range(3)]

        return res

    def get_state_desc(self):
        if self.state_desc_istep != self.istep:
            self.prev_state_desc = self.state_desc
            self.state_desc = self.compute_state_desc()
            self.state_desc_istep = self.istep
        return self.state_desc

    def set_strength(self, strength):
        self.curforces = strength
        for i in range(len(self.curforces)):
            self.muscleSet.get(i).setMaxIsometricForce(self.curforces[i] * self.maxforces[i])

    def get_body(self, name):
        return self.bodySet.get(name)

    def get_joint(self, name):
        return self.jointSet.get(name)

    def get_muscle(self, name):
        return self.muscleSet.get(name)

    def get_marker(self, name):
        return self.markerSet.get(name)

    def get_contact_geometry(self, name):
        return self.contactGeometrySet.get(name)

    def get_force(self, name):
        return self.forceSet.get(name)

    def get_action_space_size(self):
        return self.noutput

    def set_integrator_accuracy(self, integrator_accuracy):
        self.integrator_accuracy = integrator_accuracy

    def reset_manager(self):
        self.manager = osim.Manager(self.model)
        self.manager.setIntegratorAccuracy(self.integrator_accuracy)
        self.manager.initialize(self.state)

    def reset(self):
        self.state = self.model.initializeState()
        self.model.equilibrateMuscles(self.state)
        self.state.setTime(0)
        self.istep = 0

        self.reset_manager()

    def get_state(self):
        return osim.State(self.state)

    def set_state(self, state):
        self.state = state
        self.istep = int(self.state.getTime() / self.stepsize) # TODO: remove istep altogether
        self.reset_manager()

    def integrate(self):
        # Define the new endtime of the simulation
        self.istep = self.istep + 1

        # Integrate till the new endtime
        self.state = self.manager.integrate(self.stepsize * self.istep)

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
        # marker3 = R_rot.dot(marker3).T*100
        # marker6 = R_rot.dot(marker6).T*100
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


def readMotionFile(filename):
    """ Reads OpenSim .sto files. From: https://gist.github.com/mitkof6/03c887ccc867e1c8976694459a34edc3
    Parameters
    ----------
    filename: absolute path to the .sto file
    Returns
    -------
    header: the header of the .sto
    labels: the labels of the columns
    data: an array of the data
    """

    if not os.path.exists(filename):
        print('file do not exists')

    file_id = open(filename, 'r')

    # read header
    next_line = file_id.readline()
    header = [next_line]
    nc = 0
    nr = 0
    while not 'endheader' in next_line:
        if 'datacolumns' in next_line:
            nc = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'datarows' in next_line:
            nr = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'nColumns' in next_line:
            nc = int(next_line[next_line.index('=') + 1:len(next_line)])
        elif 'nRows' in next_line:
            nr = int(next_line[next_line.index('=') + 1:len(next_line)])

        next_line = file_id.readline()
        header.append(next_line)

    # process column labels
    next_line = file_id.readline()
    if next_line.isspace() == True:
        next_line = file_id.readline()

    labels = next_line.split()

    # get data
    data = []
    for i in range(1, nr + 1):
        d = [float(x) for x in file_id.readline().split()]
        data.append(d)

    file_id.close()

    return header, labels, data

def create_states_from_sto(sto_file, model_posed):
    sto = osim.Storage(sto_file)
    if sto.isInDegrees():
        sto_file = _convert_sto_to_radians(sto_file)
        sto = osim.Storage(sto_file)
    model = osim.Model(model_posed)
    model.initSystem()
    for coord in model.updCoordinateSet():
        if coord.get_locked():
            coord.set_locked(False)
    states = osim.StatesTrajectory.createFromStatesStorage(model, sto, True, False, False)
    return model, states

def _convert_sto_to_radians(sto_file):
    header = []
    with open(sto_file) as file:
        for n, line in enumerate(file):
            header.append(line)
            if line.startswith('time'):
                break
    header[4] = header[4].replace('yes', 'no')
    data = np.loadtxt(sto_file, delimiter='\t', skiprows=len(header))
    data[:, 1:] = np.deg2rad(data[:, 1:])
    new_sto = sto_file.replace('.sto', '_radians.sto')
    with open(new_sto, 'w') as output:
        output.write(''.join(header))
        np.savetxt(output, data, delimiter='\t')
    return new_sto