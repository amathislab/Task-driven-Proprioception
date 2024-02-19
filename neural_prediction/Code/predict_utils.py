##
# Utilities necessary in predictions of monkey neural data using network activations.
# Author: Axel Bisi
##

### IMPORTS
import sys, os
import numpy as np
import h5py
import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy import stats
import yaml
#import joblib


# Modules
sys.path.append('../../code/')
sys.path.append('../code/')
from neural_utils import get_CN_monkeys, get_S1_monkeys
sys.path.append(os.path.join(sys.path[0],'..'))
from global_utils import get_PCR_exp_type, get_PCR_exp_type_combined, get_network_family

from path_utils import MODELS_DIR, PATH_TO_SAVE_SPINDLEDATASET

# Paths
PATH_TO_NETWORK_DATASETS = PATH_TO_SAVE_SPINDLEDATASET
PATH_TO_MODELS = MODELS_DIR
PATH_TO_NEURANALYSIS = '..'

###---- GENERAL ----####

def get_activation_path(exp_id=None):
    return '../activations/experiment_{}/'.format(str(exp_id))

def get_results_path():
    return '..'

def get_results_path_datadriven():
    return '..'

def get_results_path_datadriven_tasktransfer():
    return '..'

def get_figfolder_path():
    return '..'

def get_neuraldata_path():
    return '..'

def make_res_directory(model_type, instance_id, instance_rand, monkey):
    ''' Create directory to store results if new.'''
    
    #Create result directory
    prefix = model_type + str(instance_id)
    if instance_rand == True:
        prefix += 'r'
        
    directory = prefix + '/'+ monkey + '/'
    result_path = os.path.join(get_results_path(), directory)
    
    #Make directory of that monkey
    if not os.path.exists(result_path):
        print('First-time predictions. Directory created :', result_path)
        os.makedirs(result_path)
    
    return result_path

def make_res_directory_kin(monkey):
    ''' Create directory to store results if new.'''
    prefix=''
    #Create result directory
    if monkey in get_CN_monkeys():
        prefix = 'CN'
    elif monkey in get_S1_monkeys():
        prefix = 'S1'
    directory = prefix + '/'+ monkey + '/'
    result_path = os.path.join(get_results_path(), directory)
    
    #Make directory of that monkey
    if not os.path.exists(result_path):
        print('First-time predictions. Directory created :', result_path)
        os.makedirs(result_path)
    
    return result_path


###---- GENERATE ACTIVATIONS ----###

def get_monkey_kin_datasets(path_to_data, monkey_name, session_date, has_passive, suffix):
    '''Load aligned and padded muscle kinematic datasets for one monkey session,
    both active and passive as a list if passive available.
    Arguments:
    path_to_data - (str) Path to folder containing padded kinematic datasets.
    monkey_name - (str) Name of the monkey.
    session_date - (int) Session date (YYYYMMDD).
    has_passive - (bool) Whether the data contains the passive task.'''

    kin_datasets = []

    #Active
    file_name_act = 'muscle_kin_active_{}_{}{}.npy'.format(monkey_name, session_date, suffix)
    dataset_act = np.load(os.path.join(path_to_data, file_name_act))
    kin_datasets.append(dataset_act)
    print('Loaded {} {} kinematic active data set of size:'.format(monkey_name, session_date), dataset_act.shape)

    # Passive
    if has_passive:
        file_name_pas = 'muscle_kin_passive_{}_{}{}.npy'.format(monkey_name, session_date, suffix)
        dataset_pas = np.load(os.path.join(path_to_data, file_name_pas))
        kin_datasets.append(dataset_pas)
        print('Loaded {} {} kinematic passive data set of size:'.format(monkey_name, session_date), dataset_act.shape)

    return kin_datasets


def load_monkey_datasets(path_to_data, monkey_name, session_date, has_passive,
                         active_start='mvt', active_length=0, align=100, control_dict=None):
    ''' Get monkey datasets.'''

    datasets = {'active':{}, 'passive':{}, 'hold':{}}
    print('Loading datasets with control:', control_dict)

    # Make suffix for title name #TODO: turn this in function later in utils?
    active_start_suff = '_'+str(active_start)
    if active_length == 0:
        active_length_suff = '_end'
    else:
        active_length_suff = '_'+str(active_length)+'ms'
    align_suff = '_at'+str(align)
    permut_suff = ''
    if control_dict['permut_m'] or control_dict['permut_t']:
        permut_suff = '_'
    if control_dict['permut_m']:
        permut_suff += 'M'
    if control_dict['permut_t']:
        permut_suff += 'T'
    const_suff=''
    if control_dict['constant_input']:
        const_suff = '_const'

    file_name_suffix_active = '{}{}{}{}{}'.format(active_start_suff,
                                                  active_length_suff,
                                                  align_suff,
                                                  permut_suff,
                                                  const_suff)
    file_name_suffix_passive = '{}{}{}'.format(align_suff,
                                               permut_suff,
                                               const_suff)
    file_name_suffix_hold = '{}{}{}'.format(align_suff,
                                            permut_suff,
                                            const_suff)


    file_name = '{}_{}_active{}.hdf5'.format(monkey_name, session_date, file_name_suffix_active)
    print('Loading muscle dataset:', file_name)
    with h5py.File(os.path.join(path_to_data, file_name), 'r') as file:
        datasets['active']['muscle_coords'] = file['muscle_coords'][()] /1000
        datasets['active']['endeffector_coords'] = file['endeff_coords'][()] /1000

    # Passive
    if has_passive:
        file_name = '{}_{}_passive{}.hdf5'.format(monkey_name, session_date, file_name_suffix_passive)
        print('Loading muscle dataset:', file_name)
        with h5py.File(os.path.join(path_to_data, file_name), 'r') as file:
            datasets['passive']['muscle_coords'] = file['muscle_coords'][()] /1000
            datasets['passive']['endeffector_coords'] = file['endeff_coords'][()] /1000

    #Hold
    #file_name = '{}_{}_hold{}.hdf5'.format(monkey_name, session_date, file_name_suffix_hold)
    #print('Loading muscle dataset:', file_name)
    #with h5py.File(os.path.join(path_to_data, file_name), 'r') as file:
    #    datasets['hold']['muscle_coords'] = file['muscle_coords'][()]
    #    datasets['hold']['endeffector_coords'] = file['endeff_coords'][()]

    return datasets

def load_monkey_datasets_align(path_to_data, monkey_name, session_date, has_passive,
                         active_start='mvt', active_length=0, align=100, control_dict=None):
    ''' Get monkey datasets.'''

    datasets = {'active':{}, 'passive':{}, 'hold':{}}
    print('Loading datasets with control:', control_dict)

    # Make suffix for title name #TODO: turn this in function later in utils?
    active_start_suff = '_'+str(active_start)
    if active_length == 0:
        active_length_suff = '_end'
    else:
        active_length_suff = '_'+str(active_length)+'ms'
    align_suff = '_at'+str(align)
    permut_suff = ''
    if control_dict['permut_m'] or control_dict['permut_t']:
        permut_suff = '_'
    if control_dict['permut_m']:
        permut_suff += 'M'
    if control_dict['permut_t']:
        permut_suff += 'T'
    const_suff=''
    if control_dict['constant_input']:
        const_suff = '_const'

    file_name_suffix_active = '{}{}{}{}{}'.format(active_start_suff,
                                                  active_length_suff,
                                                  align_suff,
                                                  permut_suff,
                                                  const_suff)
    file_name_suffix_passive = '{}{}{}'.format(align_suff,
                                               permut_suff,
                                               const_suff)
    file_name_suffix_hold = '{}{}{}'.format(align_suff,
                                            permut_suff,
                                            const_suff)


    file_name = '{}_{}_active{}.hdf5'.format(monkey_name, session_date, file_name_suffix_active)
    print('Loading muscle dataset:', file_name)
    with h5py.File(os.path.join(path_to_data, file_name), 'r') as file:
        datasets['active']['muscle_coords'] = file['muscle_coords'][()]
        datasets['active']['endeffector_coords'] = file['endeff_coords'][()]

    # Passive
    if has_passive:
        file_name = '{}_{}_passive{}.hdf5'.format(monkey_name, session_date, file_name_suffix_passive)
        print('Loading muscle dataset:', file_name)
        with h5py.File(os.path.join(path_to_data, file_name), 'r') as file:
            datasets['passive']['muscle_coords'] = file['muscle_coords'][()]
            datasets['passive']['endeffector_coords'] = file['endeff_coords'][()]

    #Hold
    #file_name = '{}_{}_hold{}.hdf5'.format(monkey_name, session_date, file_name_suffix_hold)
    #print('Loading muscle dataset:', file_name)
    #with h5py.File(os.path.join(path_to_data, file_name), 'r') as file:
    #    datasets['hold']['muscle_coords'] = file['muscle_coords'][()]
    #    datasets['hold']['endeffector_coords'] = file['endeff_coords'][()]

    return datasets

def load_monkey_datasets_align_new(path_to_data, monkey_name, session_date, has_passive,
                         active_start='mvt', active_length=0, align=100, control_dict=None):
    ''' Get monkey datasets.'''

    datasets = {'active':{}, 'passive':{}, 'hold':{}}
    print('Loading datasets with control:', control_dict)

    # Make suffix for title name #TODO: turn this in function later in utils?
    active_start_suff = '_'+str(active_start)
    if active_length == 0:
        active_length_suff = '_end'
    else:
        active_length_suff = '_'+str(active_length)+'ms'
    align_suff = '_at'+str(align)
    permut_suff = ''
    if control_dict['permut_m'] or control_dict['permut_t']:
        permut_suff = '_'
    if control_dict['permut_m']:
        permut_suff += 'M'
    if control_dict['permut_t']:
        permut_suff += 'T'
    const_suff=''
    if control_dict['constant_input']:
        const_suff = '_const'

    file_name_suffix_active = '{}{}{}{}{}'.format(active_start_suff,
                                                  active_length_suff,
                                                  align_suff,
                                                  permut_suff,
                                                  const_suff)
    file_name_suffix_passive = '{}{}{}'.format(align_suff,
                                               permut_suff,
                                               const_suff)
    file_name_suffix_hold = '{}{}{}'.format(align_suff,
                                            permut_suff,
                                            const_suff)


    file_name = '{}_{}_active{}.hdf5'.format(monkey_name, session_date, file_name_suffix_active)
    print('Loading muscle dataset:', file_name)
    with h5py.File(os.path.join(path_to_data, file_name), 'r') as file:
        datasets['active']['muscle_coords'] = file['muscle_coords'][()] /1000
        datasets['active']['endeffector_coords'] = file['endeff_coords'][()] /1000

    # Passive
    if has_passive:
        file_name = '{}_{}_passive{}.hdf5'.format(monkey_name, session_date, file_name_suffix_passive)
        print('Loading muscle dataset:', file_name)
        with h5py.File(os.path.join(path_to_data, file_name), 'r') as file:
            datasets['passive']['muscle_coords'] = file['muscle_coords'][()] /1000
            datasets['passive']['endeffector_coords'] = file['endeff_coords'][()] /1000

    #Hold
    #file_name = '{}_{}_hold{}.hdf5'.format(monkey_name, session_date, file_name_suffix_hold)
    #print('Loading muscle dataset:', file_name)
    #with h5py.File(os.path.join(path_to_data, file_name), 'r') as file:
    #    datasets['hold']['muscle_coords'] = file['muscle_coords'][()]
    #    datasets['hold']['endeffector_coords'] = file['endeff_coords'][()]

    return datasets

def load_monkey_datasets_align_new_joints_input(path_to_data, monkey_name, session_date, has_passive,
                         active_start='mvt', active_length=0, align=100, control_dict=None):
    ''' Get monkey datasets.'''

    datasets = {'active':{}, 'passive':{}, 'hold':{}}
    print('Loading datasets with control:', control_dict)

    # Make suffix for title name #TODO: turn this in function later in utils?
    active_start_suff = '_'+str(active_start)
    if active_length == 0:
        active_length_suff = '_end'
    else:
        active_length_suff = '_'+str(active_length)+'ms'
    align_suff = '_at'+str(align)
    permut_suff = ''
    if control_dict['permut_m'] or control_dict['permut_t']:
        permut_suff = '_'
    if control_dict['permut_m']:
        permut_suff += 'M'
    if control_dict['permut_t']:
        permut_suff += 'T'
    const_suff=''
    if control_dict['constant_input']:
        const_suff = '_const'

    file_name_suffix_active = '{}{}{}{}{}'.format(active_start_suff,
                                                  active_length_suff,
                                                  align_suff,
                                                  permut_suff,
                                                  const_suff)
    file_name_suffix_passive = '{}{}{}'.format(align_suff,
                                               permut_suff,
                                               const_suff)
    file_name_suffix_hold = '{}{}{}'.format(align_suff,
                                            permut_suff,
                                            const_suff)


    file_name = '{}_{}_active{}.hdf5'.format(monkey_name, session_date, file_name_suffix_active)
    print('Loading muscle dataset:', file_name)
    with h5py.File(os.path.join(path_to_data, file_name), 'r') as file:
        datasets['active']['muscle_coords'] = file['muscle_coords'][()] /1000
        datasets['active']['joint_coords'] = file['joint_coords'][()]
        datasets['active']['endeffector_coords'] = file['endeff_coords'][()] /1000

    # Passive
    if has_passive:
        file_name = '{}_{}_passive{}.hdf5'.format(monkey_name, session_date, file_name_suffix_passive)
        print('Loading muscle dataset:', file_name)
        with h5py.File(os.path.join(path_to_data, file_name), 'r') as file:
            datasets['passive']['muscle_coords'] = file['muscle_coords'][()] /1000
            datasets['passive']['joint_coords'] = file['joint_coords'][()]
            datasets['passive']['endeffector_coords'] = file['endeff_coords'][()] /1000

    #Hold
    #file_name = '{}_{}_hold{}.hdf5'.format(monkey_name, session_date, file_name_suffix_hold)
    #print('Loading muscle dataset:', file_name)
    #with h5py.File(os.path.join(path_to_data, file_name), 'r') as file:
    #    datasets['hold']['muscle_coords'] = file['muscle_coords'][()]
    #    datasets['hold']['endeffector_coords'] = file['endeff_coords'][()]

    return datasets

def load_monkey_spike_datasets(path_to_data, monkey_name, session_date, has_passive, active_start='mvt'): #TODO: combine these two functions
    ''' Get monkey spikes datasets.'''

    datasets = {'active':{}, 'passive':{}, 'hold':{}}

    # Active
    active_start='mvt'
    file_name = '{}_{}_active_{}.hdf5'.format(monkey_name, session_date, active_start)
    with h5py.File(os.path.join(path_to_data, file_name), 'r') as file:
        datasets['active']['spike_counts'] = file['spike_counts'][()]

    # Passive
    if has_passive:
        file_name = '{}_{}_passive.hdf5'.format(monkey_name, session_date, active_start)
        with h5py.File(os.path.join(path_to_data, file_name), 'r') as file:
            datasets['passive']['spike_counts'] = file['spike_counts'][()]

    #Whole
    file_name = '{}_{}_whole.hdf5'.format(monkey_name, session_date)
    with h5py.File(os.path.join(path_to_data, file_name), 'r') as file:
        datasets['whole']['spike_counts'] = file['spike_counts'][()]

    return datasets

def load_activations_data(monkey_name, data_path, model_type, instance_number = 1, random_instance = False, raw=False): #For best instances with random
    '''Function to load all DeepDraw instance layers' activations in a list.
    Arguments:
    monkey_name - (str) name of the monkey used to generate activations.
    data_path - (str) path to data folder containing the model.
    model_name - (str) name of parametrized model.
    instance_number - (int) number of model instance (1-5).
    random_instance - (bool) whether to load activations instance with random weights.'''
 
    #Define model type, name and max n. of layers
    if model_type == 'S-T': #spatial_temporal, S-T
        model_name = 'spatial_temporal_4_8-16-16-32_32-32-64-64_7293'
        n_max_layers = 8
    elif model_type == 'ST': #ST
        model_name = 'spatiotemporal_4_8-8-32-64_7272'
        n_max_layers = 4
    elif model_type == 'T-S': #T-S
        model_name = 'temporal_spatial_4_16-16-32-64_64-64-64-64_5272'
        n_max_layers = 8
    
    #Get particular path to instance
    model_instance = model_name + '_' + str(instance_number)
    if random_instance == True:
        model_instance = model_name + '_' + str(instance_number) + 'r'
    #elif random_instance == True:
    #    model_instance = model_name + '_' + str(instance_number) + 'r'
        
    print('Loading model activations: {}'.format(model_instance)) 
    path_to_model_folder = os.path.join(data_path, model_name)
    path_to_file = os.path.join(path_to_model_folder, model_instance)
    
    #Init. storage list
    layer_activations = []
    
    #Load each layer activation file separately
    for layer in range(n_max_layers):
        if raw:   
            layer_file = monkey_name+'l'+str(layer)+'_raw.pkl'
        else:
            layer_file = monkey_name+'l'+str(layer)+'.pkl'
        path_to_pkl_layer = os.path.join(path_to_file, layer_file)
        with open(path_to_pkl_layer, 'rb') as f:
            data = pickle.load(f)
            print('Load layer l{}, shape: {}'.format(layer, data.shape))
            layer_activations.append(data)
        f.close()
    print('Activations loaded !')
    return layer_activations

def load_activations(monkey_name, session_date, path_to_act_data, model_name, passive=False):
    '''Function to load all DeepDraw instance layers' activations in a list.
    Arguments:
    monkey_name - (str) name of the monkey used to generate activations: CN or S1.
    data_path - (str) path to data folder containing the model: which experiment.'''
    
    path_to_file = os.path.join(path_to_act_data, model_name) #exp/model
    #Init. storage list
    layer_activations = []
    if passive:
        suff='passive'
    else:
        suff='active'
    
    #Load each layer activation file separately
    n_max_layers=8
    for layer in range(n_max_layers):
        layer_file = '{}_{}_{}_l{}.pkl'.format(monkey_name, session_date, suff, layer)
        path_to_pkl_layer = os.path.join(path_to_file, layer_file)
        if os.path.isfile(path_to_pkl_layer):
            with open(path_to_pkl_layer, 'rb') as f:
                data = pickle.load(f)
                print('Load layer l{}, shape: {}'.format(layer, data.shape))
                layer_activations.append(data)
            f.close()
        else:
            break
    print('Activations loaded !')
    return layer_activations

def center_session_data(task_array, model_config):
    ''' Centering of muscle kinematic input (nsamples, n_time, n_muscles, 2) data using training dataset mean as per model config file..
    Arguments:
    task_data (array) - Session input task kinematic array e.g. active or passive.
    model_config (dict) - Config info for current model.
    '''
    train_data_mean = model_config['train_mean']
    task_array = np.subtract(task_array, train_data_mean)
    return task_array

###---- NEURAL PREDICTIONS ----###

def get_dilation_factor(model_config, layer_idx):
    '''Get temporal dilation factor based on model architecture and layer index.'''

    #model_config['nlayers'] = float(model_config['nlayers'])
    #model_config['t_stride'] = float(model_config['t_stride'])

    #TCNs
    if 't_stride' in model_config.keys():
        # MODELS WITH EXPONENTIAL TEMP. STRIDE
        if isinstance(model_config['t_stride'], (int, float)):
            if (model_config['arch_type'] == 'spatial_temporal') and (layer_idx < model_config['nlayers']):
                temp_factor = 1
            elif (model_config['arch_type'] == 'spatial_temporal') and (layer_idx >= model_config['nlayers']):
                temp_factor = model_config['t_stride'] ** (layer_idx % model_config['nlayers'] + 1)
            elif (model_config['arch_type'] == 'temporal_spatial') and (layer_idx < model_config['nlayers']):
                temp_factor = model_config['t_stride'] ** (layer_idx + 1)
            elif (model_config['arch_type'] == 'temporal_spatial') and (layer_idx >= model_config['nlayers']):
                temp_factor = model_config['t_stride'] ** (model_config['nlayers'] )
            elif (model_config['arch_type'] == 'spatiotemporal'):
                temp_factor = model_config['t_stride'] ** (layer_idx + 1)

        # MODELS WITH SAMPLED TEMP. STRIDE
        elif isinstance(model_config['t_stride'], list):
            if (model_config['arch_type'] == 'spatial_temporal') and (layer_idx < model_config['nlayers']):
                temp_factor = 1
            elif (model_config['arch_type'] == 'spatial_temporal') and (layer_idx >= model_config['nlayers']):
                layer_idx_tmp = layer_idx % int(model_config['nlayers'])
                temp_factor = np.prod(model_config['t_stride'][:layer_idx_tmp+1])
            elif (model_config['arch_type'] == 'temporal_spatial') and (layer_idx < model_config['nlayers']):
                temp_factor = np.prod(model_config['t_stride'][:layer_idx+1])
            elif (model_config['arch_type'] == 'temporal_spatial') and (layer_idx >= model_config['nlayers']):
                temp_factor = np.prod(model_config['t_stride'][:int(model_config['nlayers'])])
            elif (model_config['arch_type'] == 'spatiotemporal'):
                temp_factor = np.prod(model_config['t_stride'][:layer_idx+1])
                # temp_factor = np.prod(model_config['t_stride'][:layer_idx+1])
                # print(temp_factor)
    #LSTMs
    else:
        temp_factor = 1

    return int(temp_factor)

def dilate_spike_trials(ori_spike_data, temp_factor):
    '''Function to dilate spike data by taking every other time bin.'''

    aggreg_spike_data = []
    for t_idx in range(0, ori_spike_data.shape[1], temp_factor):
        aggreg_spike_data.append(np.nanmean(ori_spike_data[:, t_idx:t_idx+temp_factor, :], axis=1))
    aggreg_spike_data = np.asarray(aggreg_spike_data).swapaxes(0,1)
    return aggreg_spike_data

def interp_single(model_act):
    a_interp = np.interp(x = np.linspace(0, model_act.shape[0], 400),
                                 xp = np.linspace(0, model_act.shape[0], model_act.shape[0]),
                                 fp = model_act)
    return a_interp

def interpolate_layer_act(model_act, trial_duration):
    """

    :param layer_act:
    :param ori_spike_data:
    :return:
    """
    from scipy import interpolate

    interp_act = []
    # print('INTERPOLATE ACTIVATIONS')
    # print('act shape', model_act.shape)

    for t_idx in range(0, model_act.shape[0]):
        ## Update with map
        trial_interp_act = map(interp_single,list(model_act[t_idx].T))
        interp_act.append(np.asarray(list(trial_interp_act)).T)

    return np.asarray(interp_act)

def compute_regression_rates(model_activations, spikes, mvt_durations, model_config, layer_idx, align, current_temp_factor, interp_flag=True,shuffle_flag=False,active_as_passive_flag=False):
    '''Function to compute regression data, i.e., in successive time windows, mean rates as inputs and
        spike counts as outputs (possibly delayed).
        Arguments:
        input_data - (array) Regressors, input muscle signals or activations.
        spike_data - (array) Regressand, spike data.
        win_length - (int) Length of window in which to predict, forms a data point
        delay - (int) How much earlier spike counts should be predicted from inputs.
        layer_idx - (int) Which layer: spatial or temporal.
        bump_phase - (bool) Whether to predict passive case.'''

    # Init. storage
    rng = np.random.RandomState(2022)
    spike_rates = []
    model_rates = []
    global_start = int(align)

    if not interp_flag:
        # Dilate spike data (aggregate)
        layer_t_stride = get_dilation_factor(model_config, layer_idx)

        print('Current temporal stride', current_temp_factor)
        # Define trial starts/ends (valid for both active and passive)
        global_start = global_start / current_temp_factor
        global_ends = global_start + (mvt_durations / current_temp_factor)
    else:
        model_activations = interpolate_layer_act(model_activations, mvt_durations)

        global_start = global_start
        global_ends = global_start + mvt_durations

    # print(mvt_durations)

    # print(global_start)
    # print(global_ends)

    # print('Average trial duration: {:.3f}ms '.format(np.mean(mvt_durations)*10))
    # Iterate over trials
    new_mvt_duration = []
    for trial_idx in range(mvt_durations.shape[0]):

        # Trial limit indices, round below and above
        global_start_tmp = int(np.floor(global_start))
        global_end_tmp = int(np.ceil(global_ends[trial_idx]))

        # print('************')
        # print('Start ', global_start_tmp)
        # print('End ', global_end_tmp)

        # print('Diff before ', global_end - global_start)
        if active_as_passive_flag:
            if global_end_tmp - 13 > global_start_tmp:
                global_start_tmp = rng.randint(global_start_tmp,global_end_tmp-13)
                global_end_tmp = global_start_tmp + 13
        
        # print('Start ', global_start_tmp)
        # print('End ', global_end_tmp)
        # print('Diff now ', global_end - global_start)

        #Get rates
        trial_model_rates = model_activations[trial_idx, global_start_tmp:global_end_tmp, :]

        if shuffle_flag:
            # print(layer_activations.shape)
            shuffle_ind = rng.choice(np.arange(trial_model_rates.shape[0]),trial_model_rates.shape[0],False)
            trial_model_rates = trial_model_rates[shuffle_ind,:]

        trial_neural_rates = spikes[trial_idx, global_start_tmp:global_end_tmp,:]
        # Concatenate single trials
        spike_rates.extend(trial_neural_rates)
        model_rates.extend(trial_model_rates)
        new_mvt_duration.append(global_end_tmp-global_start_tmp)

    # Make as array
    spike_rates = np.asarray(spike_rates)
    model_rates = np.asarray(model_rates)
    new_mvt_duration = np.asarray(new_mvt_duration)

    return model_rates, spike_rates, new_mvt_duration

# def compute_regression_rates(model_activations, spikes, mvt_durations, model_config, layer_idx, align, current_temp_factor):
#     '''Function to compute regression data, i.e., in successive time windows, mean rates as inputs and
#         spike counts as outputs (possibly delayed).
#         Arguments:
#         input_data - (array) Regressors, input muscle signals or activations.
#         spike_data - (array) Regressand, spike data.
#         win_length - (int) Length of window in which to predict, forms a data point
#         delay - (int) How much earlier spike counts should be predicted from inputs.
#         layer_idx - (int) Which layer: spatial or temporal.
#         bump_phase - (bool) Whether to predict passive case.'''

#     # Init. storage
#     spike_rates = []
#     model_rates = []
#     global_start = int(align)

#     # Dilate spike data (aggregate)
#     #layer_t_stride = get_dilation_factor(model_config, layer_idx)

#     print('Current temporal stride', current_temp_factor)
#     spikes = dilate_spike_trials(spikes, current_temp_factor)

#     # Define trial starts/ends (valid for both active and passive)
#     print('Average trial duration: {:.3f}ms '.format(np.mean(mvt_durations)*10))
#     global_start = global_start / current_temp_factor
#     global_ends = global_start + (mvt_durations / current_temp_factor)

#     # Iterate over trials
#     for trial_idx in range(mvt_durations.shape[0]):

#         # Trial limit indices, round below and above
#         global_start = int(np.floor(global_start))
#         global_end = int(np.ceil(global_ends[trial_idx]))

#         #Get rates
#         trial_model_rates = model_activations[trial_idx, global_start:global_end, :]
#         trial_neural_rates = spikes[trial_idx, global_start:global_end,:]
#         # Concatenate single trials
#         spike_rates.extend(trial_neural_rates)
#         model_rates.extend(trial_model_rates)

#     # Make as array
#     spike_rates = np.asarray(spike_rates)
#     model_rates = np.asarray(model_rates)

#     return model_rates, spike_rates

def compute_trial_rates(model_activations, spikes, mvt_durations, model_config, layer_idx):
    '''Function to compute time-averaged regression rates, i.e., in successive time windows, mean rates as inputs and
        spike counts as outputs (possibly delayed).
        Arguments:
        input_data - (array) Regressors, input muscle signals or activations.
        spike_data - (array) Regressand, spike data.
        win_length - (int) Length of window in which to predict, forms a data point
        delay - (int) How much earlier spike counts should be predicted from inputs.
        layer_idx - (int) Which layer: spatial or temporal.
        bump_phase - (bool) Whether to predict passive case.'''

    # Init. storage, get timescale & model parameters
    spike_rates = []
    model_rates = []
    global_start = 0

    #Dilate spike data (aggregate)
    temp_factor = get_dilation_factor(model_config, layer_idx)
    spikes = dilate_spike_trials(spikes, temp_factor)


    # Define trial ends (valid for both active and passive)
    print('Average trial duration (ms):', np.mean(mvt_durations)*10)
    global_ends = global_start + (mvt_durations / temp_factor)

    # Iterate over trials because each trial end at different time
    print('INPUT/OUTPUT DATA SHAPES:', model_activations.shape, spikes.shape)
    for trial_idx in range(mvt_durations.shape[0]):

        # Trial limit indices, round below and above
        global_start = int(np.floor(global_start))
        global_end = int(np.ceil(global_ends[trial_idx]))

        #Get time-averaged rates
        trial_model_rates = np.nanmean(model_activations[trial_idx, global_start:global_end, :], axis=0)
        trial_neural_rates = np.nanmean(spikes[trial_idx, global_start:global_end,:], axis=0)

        # Append single trials
        spike_rates.append(trial_neural_rates)
        model_rates.append(trial_model_rates)

    # Make as array
    spike_rates = np.asarray(spike_rates)
    model_rates = np.asarray(model_rates)

    return model_rates, spike_rates


def compute_input_regression_rates(input_kin_array, spikes, mvt_durations):
    '''Function to compute regression data, i.e., in successive time windows, mean rates as inputs and
        spike counts as outputs.
        Arguments:
        input_data - (array) Regressors, input muscle signals or activations.
        spike_data - (array) Regressand, spike data.
        mvt_durations - (array) Trial ends.'''

    # Init. storage, get timescale & model parameters
    spike_rates = []
    kin_rates = []
    global_start = 0

    # Define trial ends (valid for both active and passive)
    print('Average trial duration (ms):', np.mean(mvt_durations)*10)
    global_ends = global_start + mvt_durations

    # Iterate over trials because each trial end at different time
    for trial_idx in range(mvt_durations.shape[0]):

        # Trial limit indices, round below and above
        global_start = int(np.floor(global_start))
        global_end = int(np.ceil(global_ends[trial_idx]))

        #Get rates
        trial_kin_rates = input_kin_array[trial_idx, global_start:global_end, :]
        trial_neural_rates = spikes[trial_idx, global_start:global_end,:]

        # Concatenate single trials
        spike_rates.extend(trial_neural_rates)
        kin_rates.extend(trial_kin_rates)

    # Make as array
    spike_rates = np.asarray(spike_rates)
    kin_rates = np.asarray(kin_rates)

    return kin_rates, spike_rates

###---- DATA FORMATTING ----###

#def load_ev_scores(exp_id, monkey_name, session_date, model_name, ispassive, normalize=False, active_start='mvt', suffix=None):
def load_ev_scores(exp_id, monkey_name, session_date, model_name, ispassive, normalize=False, params_dict=None,
                   nlayers=None,train_iter=None,shuffle_flag=False,result_path='..',window=5,latency=0):
    '''Load layers results into list.'''

    #if get_PCR_exp_type(exp_id) == 'classification': #This works for TCNs #TODO: improve
    #    nlayer_loc_s, nlayer_loc_ts_st = 15, 17
    #    nlayer_loc_rec = 5
    #elif get_PCR_exp_type(exp_id) == 'regression':
    #    nlayer_loc_s, nlayer_loc_ts_st = 17, 19
    #    nlayer_loc_rec = 7
    #else: #BTs
    #    nlayer_loc_s, nlayer_loc_ts_st = 15, 17


    #if 'spatiotemporal' in model_name:
    #    n_layers_max = int(model_name[nlayer_loc_s])
    #elif 'spatial' in model_name:
    #    n_layers_max = 2*int(model_name[nlayer_loc_ts_st])
    #elif 'lstm' in model_name:
    #    n_layers_max = int(model_name[nlayer_loc_rec])

    if 'spatiotemporal' in model_name:
        n_layers_max = int(nlayers)
    elif 'spatial' in model_name:
        n_layers_max = 2 * int(nlayers)
    elif 'lstm' in model_name:
        n_layers_max = int(nlayers)

    # GET RESULT FILE NAME
    if ispassive:
        task='passive'
        fname_end='_at'+str(params_dict['align'])
    else:
        task='active'
        fname_end = '_' + params_dict['active_start'] #TODO: remove two _
        if params_dict['active_length'] == 0:
            fname_end += '_end'
        else:
            fname_end += '_' + str(params_dict['active_length']) + 'ms'
        fname_end += '_at' + str(params_dict['align'])

    if params_dict['permut_m'] or params_dict['permut_t']:
        fname_end += '_'
    if params_dict['permut_m']:
        fname_end += 'M'
    if params_dict['permut_t']:
        fname_end += 'T'
    if params_dict['constant_input']:
        fname_end += '_const'

    # New: suffix for dev., controls and comparisons
    if 'suffix' in params_dict.keys():
        if params_dict['suffix'] is not None:
            fname_end += '_'+params_dict['suffix']

    ev_scores_lm_te = []
    ev_scores_lm_tr = []

    path_to_results = result_path + 'experiment_{}/'.format(exp_id) + model_name + '/_w{}l{}/'.format(window, latency)
    # path_to_results = get_results_path() + 'experiment_{}/'.format(exp_id) + model_name + '/_w5l0/'

    # LOAD EACH LAYER
    for i in range(n_layers_max):
        try:
            if train_iter is not None:
                layer_file = task + '_' + monkey_name + '_' + str(session_date) + '_l' + str(i) + '_w{}l{}{}_ckpt{}.h5'.format(window, latency,fname_end,str(train_iter))
            else:
                if shuffle_flag:
                    layer_file = task + '_' + monkey_name + '_' + str(session_date) + '_l' + str(i) + '_w{}l{}{}_shuffled.h5'.format(window, latency,fname_end)
                else:
                    layer_file = task + '_' + monkey_name + '_' + str(session_date) + '_l' + str(i) + '_w{}l{}{}.h5'.format(window, latency,fname_end)
            
            with h5py.File(os.path.join(path_to_results, layer_file), 'r') as f:
                scores_lm_te = f['ev_test'][()]
                scores_lm_tr = f['ev_train'][()]

                if normalize == True:
                    scores_lm_te = normalize_ceiling(scores_lm_te, monkey_name, session_date, ispassive) #TODO: normalize per dilation
                    scores_lm_tr = normalize_ceiling(scores_lm_tr, monkey_name, session_date, ispassive) #TODO: normalize per dilation

                ev_scores_lm_te.append(scores_lm_te)
                ev_scores_lm_tr.append(scores_lm_tr)
        except:
            if train_iter is not None:
                layer_file = task + '_' + monkey_name + '_' + str(session_date) + '_l' + str(i) + '_w{}l{}{}_ckpt{}.json'.format(window, latency,fname_end,str(train_iter))
            else:
                if shuffle_flag:
                    layer_file = task + '_' + monkey_name + '_' + str(session_date) + '_l' + str(i) + '_w{}l{}{}_shuffled.json'.format(window, latency,fname_end)
                else:
                    layer_file = task + '_' + monkey_name + '_' + str(session_date) + '_l' + str(i) + '_w{}l{}{}.json'.format(window, latency,fname_end)
            with open(os.path.join(path_to_results, layer_file)) as json_file:
                data = json.load(json_file)
                scores_lm_te = data['lm']['ev_test']
                scores_lm_tr = data['lm']['ev_train']

                if normalize == True:
                    scores_lm_te = normalize_ceiling(scores_lm_te, monkey_name, session_date, ispassive) #TODO: normalize per dilation
                    scores_lm_tr = normalize_ceiling(scores_lm_tr, monkey_name, session_date, ispassive) #TODO: normalize per dilation

                ev_scores_lm_te.append(scores_lm_te)
                ev_scores_lm_tr.append(scores_lm_tr)

    return np.asarray(ev_scores_lm_te), np.asarray(ev_scores_lm_tr)

def load_ev_scores_combined(exp_id_list, monkey_name, session_date, model_name, ispassive, normalize=False, params_dict=None,
                   nlayers=None,train_iter=None,path_to_results='..'):
    '''Load layers results into list.'''

    #if get_PCR_exp_type(exp_id) == 'classification': #This works for TCNs #TODO: improve
    #    nlayer_loc_s, nlayer_loc_ts_st = 15, 17
    #    nlayer_loc_rec = 5
    #elif get_PCR_exp_type(exp_id) == 'regression':
    #    nlayer_loc_s, nlayer_loc_ts_st = 17, 19
    #    nlayer_loc_rec = 7
    #else: #BTs
    #    nlayer_loc_s, nlayer_loc_ts_st = 15, 17


    #if 'spatiotemporal' in model_name:
    #    n_layers_max = int(model_name[nlayer_loc_s])
    #elif 'spatial' in model_name:
    #    n_layers_max = 2*int(model_name[nlayer_loc_ts_st])
    #elif 'lstm' in model_name:
    #    n_layers_max = int(model_name[nlayer_loc_rec])

    if 'spatiotemporal' in model_name:
        n_layers_max = int(nlayers)
    elif 'spatial' in model_name:
        n_layers_max = 2 * int(nlayers)
    elif 'lstm' in model_name:
        n_layers_max = int(nlayers)

    # GET RESULT FILE NAME
    if ispassive:
        task='passive'
        fname_end='_at'+str(params_dict['align'])
    else:
        task='active'
        fname_end = '_' + params_dict['active_start'] #TODO: remove two _
        if params_dict['active_length'] == 0:
            fname_end += '_end'
        else:
            fname_end += '_' + str(params_dict['active_length']) + 'ms'
        fname_end += '_at' + str(params_dict['align'])

    if params_dict['permut_m'] or params_dict['permut_t']:
        fname_end += '_'
    if params_dict['permut_m']:
        fname_end += 'M'
    if params_dict['permut_t']:
        fname_end += 'T'
    if params_dict['constant_input']:
        fname_end += '_const'

    # New: suffix for dev., controls and comparisons
    if 'suffix' in params_dict.keys():
        if params_dict['suffix'] is not None:
            fname_end += '_'+params_dict['suffix']

    ev_scores_lm_te = []
    ev_scores_lm_tr = []

    exp_id_name = '_'.join(map(str, exp_id_list))

    path_to_results = path_to_results + 'experiment_{}/'.format(exp_id_name) + model_name + '/_w5l0/'
    # path_to_results = get_results_path() + 'experiment_{}/'.format(exp_id_name) + model_name + '/_w5l0/'

    # LOAD EACH LAYER
    for i in range(n_layers_max):
        try:
            if train_iter is not None:
                layer_file = task + '_' + monkey_name + '_' + str(session_date) + '_l' + str(i) + '_w5l0{}_ckpt{}.h5'.format(fname_end,str(train_iter))
            else:

                layer_file = task + '_' + monkey_name + '_' + str(session_date) + '_l' + str(i) + '_w5l0{}.h5'.format(fname_end)
            
            with h5py.File(os.path.join(path_to_results, layer_file), 'r') as f:
                scores_lm_te = f['ev_test'][()]
                scores_lm_tr = f['ev_train'][()]

                if normalize == True:
                    scores_lm_te = normalize_ceiling(scores_lm_te, monkey_name, session_date, ispassive) #TODO: normalize per dilation
                    scores_lm_tr = normalize_ceiling(scores_lm_tr, monkey_name, session_date, ispassive) #TODO: normalize per dilation

                ev_scores_lm_te.append(scores_lm_te)
                ev_scores_lm_tr.append(scores_lm_tr)
        except:
            if train_iter is not None:
                layer_file = task + '_' + monkey_name + '_' + str(session_date) + '_l' + str(i) + '_w5l0{}_ckpt{}.json'.format(fname_end,str(train_iter))
            else:
                layer_file = task + '_' + monkey_name + '_' + str(session_date) + '_l' + str(i) + '_w5l0{}.json'.format(fname_end)
            with open(os.path.join(path_to_results, layer_file)) as json_file:
                data = json.load(json_file)
                scores_lm_te = data['lm']['ev_test']
                scores_lm_tr = data['lm']['ev_train']

                if normalize == True:
                    scores_lm_te = normalize_ceiling(scores_lm_te, monkey_name, session_date, ispassive) #TODO: normalize per dilation
                    scores_lm_tr = normalize_ceiling(scores_lm_tr, monkey_name, session_date, ispassive) #TODO: normalize per dilation

                ev_scores_lm_te.append(scores_lm_te)
                ev_scores_lm_tr.append(scores_lm_tr)


    return np.asarray(ev_scores_lm_te), np.asarray(ev_scores_lm_tr)

def load_ev_scores_datadriven(exp_id, monkey_name, session_date, model_name, ispassive, normalize=False, params_dict=None,
                   nlayers=None,train_iter=None,task_transfer=False,path_to_results='..'):
    '''Load datadriven results into list.'''

    if 'spatiotemporal' in model_name:
        n_layers_max = int(nlayers)
    elif 'spatial' in model_name:
        n_layers_max = 2 * int(nlayers)
    elif 'lstm' in model_name:
        n_layers_max = int(nlayers)

    # GET RESULT FILE NAME
    if ispassive:
        task='passive'
        fname_end='_at'+str(params_dict['align'])
    else:
        task='active'
        fname_end = '_' + params_dict['active_start'] #TODO: remove two _
        if params_dict['active_length'] == 0:
            fname_end += '_end'
        else:
            fname_end += '_' + str(params_dict['active_length']) + 'ms'
        fname_end += '_at' + str(params_dict['align'])

    if params_dict['permut_m'] or params_dict['permut_t']:
        fname_end += '_'
    if params_dict['permut_m']:
        fname_end += 'M'
    if params_dict['permut_t']:
        fname_end += 'T'
    if params_dict['constant_input']:
        fname_end += '_const'

    # New: suffix for dev., controls and comparisons
    if 'suffix' in params_dict.keys():
        if params_dict['suffix'] is not None:
            fname_end += '_'+params_dict['suffix']

    ev_scores_lm_te = []
    ev_scores_lm_tr = []

    path_to_results = path_to_results + 'experiment_{}/'.format(exp_id) + model_name + '/_w5l0/'
    # if task_transfer:
    #     path_to_results = get_results_path_datadriven_tasktransfer() + 'experiment_{}/'.format(exp_id) + model_name + '/_w5l0/'
    # else:    
    #     path_to_results = get_results_path_datadriven() + 'experiment_{}/'.format(exp_id) + model_name + '/_w5l0/'

    # LOAD EACH LAYER
    if train_iter is not None:
        layer_file = task + '_' + monkey_name + '_' + str(session_date) + '_ckpt{}.json'.format(str(train_iter))
    else:
        layer_file = task + '_' + monkey_name + '_' + str(session_date) + '.json'
    with open(os.path.join(path_to_results, layer_file)) as json_file:
        try:
            data = json.load(json_file)

            scores_lm_te = data['ev_test']
            scores_lm_tr = data['ev_train']

            if normalize == True:
                scores_lm_te = normalize_ceiling(scores_lm_te, monkey_name, session_date, ispassive) #TODO: normalize per dilation
                scores_lm_tr = normalize_ceiling(scores_lm_tr, monkey_name, session_date, ispassive) #TODO: normalize per dilation

            ev_scores_lm_te.append(scores_lm_te)
            ev_scores_lm_tr.append(scores_lm_tr)

        except ValueError:
            print("Response content is not valid JSON")

    return np.asarray(ev_scores_lm_te), np.asarray(ev_scores_lm_tr)

#TODO: parallelize!
def load_exp_results(exp_id, monkey_session_tuples, normalize=False, load_passive=False, params_dict=None, tuned_ids=False, shuffle_flag=False,
                     path=None, train_iter=None, result_path=None,window=5,latency=0):
    ''' Load all single-neuron results for each experiment along with DNN model info.
    Arguments:
    exp_id - (int) Experiment ID.
    monkey_session_tuples - (list of tuples) (monkey, session) to include.
    '''
    # if result_path is not None:
    #     print('Updated result path!')
    #     def get_results_path():
    #         return result_path
    #     print(get_results_path())

    # Select experiment models
    # path_to_res = os.path.join(get_results_path(), 'experiment_{}'.format(exp_id))

    path_to_res = os.path.join(result_path, 'experiment_{}'.format(exp_id))
    model_list = os.listdir(path_to_res)
    model_list = [m for m in model_list if 'experiment_'.format(exp_id) not in m]

    # Init. dataframe cols
    df_columns = ['neuron_ids',
                  'exp_id',
                  'model_task',
                  'model_name',
                  'arch_type',
                  'model_val_acc',
                  'model_test_acc',
                  'model_train_acc',
                  'monkey',
                  'session',
                  'area',
                  'co_task',
                  'model_layer',
                  'layer_type',
                  'model_max_layer',
                  't_stride',
                  's_stride',
                  't_kernelsize',
                  's_kernelsize',
                  'n_tkernels',
                  'n_skernels',
                  'ev_train',
                  'ev_test']
    res_df = pd.DataFrame(columns=df_columns)

    neuron_ids = []
    exp_id_all = []
    model_name_all = []
    arch_type_all = []
    model_task_all = []
    model_val_acc_all = []
    model_test_acc_all = []
    model_train_acc_all = []
    monkey_all = []
    session_all = []
    area_all = []
    co_task_all = []
    model_layer_all = []
    model_type_layer_all = []
    model_max_layer_all = []
    model_t_stride_all = []
    model_s_stride_all = []
    model_t_kernelsize_all = []
    model_s_kernelsize_all = []
    model_n_tkernels_all = []
    model_n_skernels_all = []
    ev_train_all = []
    ev_test_all = []


    if params_dict is None: #default predictions
        params_dict = {'active_start':'mvt', 'active_length':0, 'align':100,
                'permut_m':False, 'permut_t':False, 'constant_input':False, 'suffix':None}

    for (monkey, session_date) in monkey_session_tuples:
        print('Loading results for {}...'.format(monkey))

        # LOAD TUNED NEURONS ONLY
        neurons_to_add = None
        if tuned_ids:
            path_to_tuned_ids = os.path.join(PATH_TO_NEURANALYSIS,
                                             'Results',
                                             'analysis_neurons_ranksum_multicorr_all_sessions.pkl')
            neuron_id_dict = np.load(path_to_tuned_ids, allow_pickle=True)
            if monkey == 'S1Lando':
                session_key = 'S1' + str(session_date)
            else:
                session_key = session_date
            tuned_ids = neuron_id_dict[session_key]
            neurons_to_add = tuned_ids
            print('Loading {} neuron ids:{}'.format(len(neurons_to_add), neurons_to_add))

        # LOAD MODEL RESULTS
        PCR_exp_type = get_PCR_exp_type(exp_id)
        for model_name in model_list:
            
            # Load model config file
            path_to_exp_models = os.path.join(PATH_TO_MODELS, 'experiment_{}'.format(exp_id))
            path_to_config_file = os.path.join(path_to_exp_models, '{}/config.yaml'.format(model_name))
            with open(path_to_config_file, 'r') as myfile:
                model_config = yaml.safe_load(myfile)
                if exp_id in [4045, 4046, 10045, 10545, 13545, 8045] + list(range(12645,13045,100)) + list(range(12646,13046,100)) + list(range(12647,13047,100)) + list(range(17046,18046,100)) + list(range(20046,21046,100)):
                    l_key = 'npplayers'
                    nlayers = model_config[l_key] + 1
                else:
                    l_key = 'nlayers'
                    nlayers = model_config[l_key]


            # ACTIVE
            try:
                if train_iter is not None:
                    ev_lm_te, ev_lm_tr = load_ev_scores(exp_id=exp_id, monkey_name=monkey, session_date=session_date,
                                                model_name=model_name, ispassive=load_passive, normalize=normalize,
                                                    params_dict=params_dict, nlayers=nlayers, train_iter=train_iter,result_path=result_path,window=window,latency=latency)
                else:
                    if shuffle_flag:
                        ev_lm_te, ev_lm_tr = load_ev_scores(exp_id=exp_id, monkey_name=monkey, session_date=session_date,
                                                model_name=model_name, ispassive=load_passive, normalize=normalize,
                                                    params_dict=params_dict, nlayers=nlayers, train_iter=train_iter,shuffle_flag=shuffle_flag,result_path=result_path,window=window,latency=latency)
                    else:
                        ev_lm_te, ev_lm_tr = load_ev_scores(exp_id=exp_id, monkey_name=monkey, session_date=session_date,
                                                    model_name=model_name, ispassive=load_passive, normalize=normalize,
                                                        params_dict=params_dict, nlayers=nlayers,result_path=result_path,window=window,latency=latency)
                

                if neurons_to_add is None:
                    neurons_to_add = np.arange(ev_lm_te.shape[1])
                    #print('Loading {} neuron ids:{}'.format(len(neurons_to_add), neurons_to_add))

                # LOAD RESULTS AND FIELDS
                N_NEURONS = len(neurons_to_add)
                
                # print('****************************')
                # print(PCR_exp_type)

                regression_task_list = ['regression', 'torque', 'center_out', 'joints_input', \
                                    'regress_joints_pos', 'regress_joints_vel', 'regress_joints_pos_vel', 'regress_joints_pos_vel_acc', \
                                    'regress_ee_vel', 'regress_ee_velocity', 'regress_ee_pos_vel', 'regress_ee_pos_vel_acc', \
                                    'autoencoder', 'autoencoder_lin', \
                                    'regress_ee_elbow_pos', 'regress_ee_elbow_vel', 'regress_ee_elbow_pos_vel', 'regress_ee_elbow_pos_vel_acc', \
                                    'regress_ee_pos_forward', 'regress_ee_pos_vel_forward', 'regress_ee_pos_vel_acc_forward', \
                                    'regress_joint_pos_forward', 'regress_joint_pos_vel_forward', 'regress_joint_pos_vel_acc_forward', \
                                    'regress_muscles_forward', 'regress_muscles_acc_forward']

                if PCR_exp_type == 'data_driven':
                    model_val_acc_tmp, model_test_acc_tmp, model_train_acc_tmp = get_model_accuracy_datadriven_pca(exp_id, model_name)
                elif PCR_exp_type in regression_task_list: #(PCR_exp_type == 'center_out') or (PCR_exp_type == 'torque'):
                    model_val_acc_tmp, model_test_acc_tmp, model_train_acc_tmp = get_model_accuracy_general(exp_id, model_name)
                else:
                    # print(PCR_exp_type)
                    model_val_acc_tmp, model_test_acc_tmp, model_train_acc_tmp = get_model_accuracy(PCR_exp_type, model_name)

                for l_idx in range(ev_lm_te.shape[0]):
                # for l_idx in range(int(nlayers)):
                    #print('Loading results layer', l_idx, 'model_name', model_name)
                    neuron_ids.extend(neurons_to_add)
                    exp_id_all.extend(N_NEURONS * [exp_id])
                    model_name_all.extend(N_NEURONS * [model_name])

                    if  'arch_type' in model_config.keys():
                        arch_type_all.extend(N_NEURONS * [model_config['arch_type']])
                    else:
                        arch_type_all.extend(N_NEURONS * [model_config['rec_blocktype']])

                    # model_task_all.extend(N_NEURONS * [get_PCR_exp_type(exp_id)])
                    model_task_all.extend(N_NEURONS * [PCR_exp_type])
                    # model_val_acc_all.extend(N_NEURONS * [get_model_accuracy(exp_id, model_name)[0]])
                    # model_test_acc_all.extend(N_NEURONS * [get_model_accuracy(exp_id, model_name)[1]])
                    model_val_acc_all.extend(N_NEURONS * [model_val_acc_tmp])
                    model_test_acc_all.extend(N_NEURONS * [model_test_acc_tmp])
                    model_train_acc_all.extend(N_NEURONS * [model_train_acc_tmp])
                    
                    monkey_all.extend(N_NEURONS * [monkey])
                    session_all.extend(N_NEURONS * [session_date])
                    area = 'CN' if monkey in get_CN_monkeys() else 'S1'
                    area_all.extend(N_NEURONS * [area])
                    co_task_all.extend(N_NEURONS * ['active'])
                    model_layer_all.extend(N_NEURONS * [l_idx])

                    if 'npplayers' in model_config.keys():
                        n_tot_layer = model_config['npplayers'] + 1
                    elif model_config['arch_type'] == 'spatiotemporal':
                        n_tot_layer = model_config['nlayers']
                    else:
                        n_tot_layer = 2 * model_config['nlayers']
                    model_max_layer_all.extend(N_NEURONS * [n_tot_layer])

                    # model_type_layer_all.extend(N_NEURONS * [get_model_layer_type(l_idx,n_tot_layer,model_name)])
                    layer_type = get_model_layer_type(l_idx,n_tot_layer,model_name)
                    model_type_layer_all.extend(N_NEURONS * [layer_type])

                    if 'npplayers' in model_config.keys():
                        model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'])])
                        model_t_stride_all.extend(N_NEURONS * [0])
                    elif isinstance(model_config['t_stride'], list):
                        if layer_type == 'temporal':
                            l_idx_stride = l_idx
                            if l_idx >= n_tot_layer/2:
                                l_idx_stride = int(l_idx - n_tot_layer/2)
                            model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'][l_idx_stride])])
                            model_s_stride_all.extend(N_NEURONS * [0])
                        elif layer_type == 'spatial':
                            l_idx_stride = l_idx
                            if l_idx >= n_tot_layer/2:
                                l_idx_stride = int(l_idx - n_tot_layer/2)
                            model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'][l_idx_stride])])
                            model_t_stride_all.extend(N_NEURONS * [0])
                        elif layer_type == 'spatiotemporal':
                            model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'][l_idx])])
                            model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'][l_idx])])
                    else:
                        if layer_type == 'temporal':
                            model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'])])
                            model_s_stride_all.extend(N_NEURONS * [0])
                        elif layer_type == 'spatial':
                            model_t_stride_all.extend(N_NEURONS * [0])
                            model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'])])
                        elif layer_type == 'spatiotemporal':
                            model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'])])
                            model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'])])

                    if 'npplayers' in model_config.keys():
                        model_t_kernelsize_all.extend(N_NEURONS * [0])
                        model_s_kernelsize_all.extend(N_NEURONS * [int(model_config['s_kernelsize'])])
                    else:
                        model_t_kernelsize_all.extend(N_NEURONS * [int(model_config['t_kernelsize'])])
                        model_s_kernelsize_all.extend(N_NEURONS * [int(model_config['s_kernelsize'])])
                    
                    if 'npplayers' in model_config.keys():
                        model_n_tkernels_all.extend(N_NEURONS * [0])
                        if l_idx < n_tot_layer -1:
                            model_n_skernels_all.extend(N_NEURONS * [int(model_config['nppfilters'][l_idx])])
                        else:
                            model_n_skernels_all.extend(N_NEURONS * [0])
                    else:
                        if layer_type == 'temporal':
                            l_idx_kernel = l_idx
                            if l_idx >= n_tot_layer/2:
                                l_idx_kernel = int(l_idx - n_tot_layer/2)
                            model_n_tkernels_all.extend(N_NEURONS * [int(model_config['n_tkernels'][l_idx_kernel])])
                            model_n_skernels_all.extend(N_NEURONS * [0])

                        if layer_type == 'spatial':
                            l_idx_kernel = l_idx
                            if l_idx >= n_tot_layer/2:
                                l_idx_kernel = int(l_idx - n_tot_layer/2)
                            model_n_skernels_all.extend(N_NEURONS * [int(model_config['n_skernels'][l_idx_kernel])])
                            model_n_tkernels_all.extend(N_NEURONS * [0])
                        
                        if layer_type == 'spatiotemporal':
                            model_n_skernels_all.extend(N_NEURONS * [int(model_config['n_skernels'][l_idx])])
                            model_n_tkernels_all.extend(N_NEURONS * [int(model_config['n_tkernels'][l_idx])])

                    ev_train_all.extend(ev_lm_tr[l_idx, neurons_to_add])
                    ev_test_all.extend(ev_lm_te[l_idx, neurons_to_add])

            except FileNotFoundError as e:
                print('Results not found', e)
                continue

            # # PASSIVE
            # if load_passive == True:
            #     try:
            #         ev_lm_te, ev_lm_tr = load_ev_scores(exp_id=exp_id, monkey_name=monkey, session_date=session_date,
            #                                   model_name=model_name, ispassive=True, normalize=normalize,
            #                                   params_dict=params_dict, nlayers=nlayers,shuffle_flag=shuffle_flag,result_path=result_path)

            #         # LOAD RESULTS AND FIELDS
            #         N_NEURONS = len(neurons_to_add)

            #         if PCR_exp_type == 'data_driven':
            #             model_val_acc_tmp, model_test_acc_tmp, model_train_acc_tmp = get_model_accuracy_datadriven_pca(exp_id, model_name)
            #         elif (PCR_exp_type == 'center_out') or (PCR_exp_type == 'torque'):
            #             model_val_acc_tmp, model_test_acc_tmp, model_train_acc_tmp = get_model_accuracy_general(exp_id, model_name)
            #         else:
            #             model_val_acc_tmp, model_test_acc_tmp, model_train_acc_tmp = get_model_accuracy(PCR_exp_type, model_name)

            #         for l_idx in range(ev_lm_te.shape[0]):
            #         # for l_idx in range(int(nlayers)):
            #             neuron_ids.extend(neurons_to_add)
            #             exp_id_all.extend(N_NEURONS * [exp_id])
            #             model_name_all.extend(N_NEURONS * [model_name])

            #             if 'arch_type' in model_config.keys():
            #                 arch_type_all.extend(N_NEURONS * [model_config['arch_type']])
            #             else:
            #                 arch_type_all.extend(N_NEURONS * [model_config['rec_blocktype']])

            #             # model_task_all.extend(N_NEURONS * [get_PCR_exp_type(exp_id)])
            #             # model_val_acc_all.extend(N_NEURONS * [get_model_accuracy(exp_id, model_name)[0]])
            #             # model_test_acc_all.extend(N_NEURONS * [get_model_accuracy(exp_id, model_name)[1]])
            #             model_task_all.extend(N_NEURONS * [PCR_exp_type])
            #             model_val_acc_all.extend(N_NEURONS * [model_val_acc_tmp])
            #             model_test_acc_all.extend(N_NEURONS * [model_test_acc_tmp])
            #             model_train_acc_all.extend(N_NEURONS * [model_train_acc_tmp])

            #             monkey_all.extend(N_NEURONS * [monkey])
            #             session_all.extend(N_NEURONS * [session_date])
            #             area = 'CN' if monkey in get_CN_monkeys() else 'S1'
            #             area_all.extend(N_NEURONS * [area])
            #             co_task_all.extend(N_NEURONS * ['passive'])
            #             model_layer_all.extend(N_NEURONS * [l_idx])

            #             if 'npplayers' in model_config.keys():
            #                 n_tot_layer = model_config['npplayers']
            #             elif model_config['arch_type'] == 'spatiotemporal':
            #                 n_tot_layer = model_config['nlayers']
            #             else:
            #                 n_tot_layer = 2 * model_config['nlayers']
            #             model_max_layer_all.extend(N_NEURONS * [n_tot_layer])

            #             # model_type_layer_all.extend(N_NEURONS * [get_model_layer_type(l_idx,n_tot_layer,model_name)])

            #             layer_type = get_model_layer_type(l_idx,n_tot_layer,model_name)
            #             model_type_layer_all.extend(N_NEURONS * [layer_type])

            #             if 'npplayers' in model_config.keys():
            #                 model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'])])
            #                 model_t_stride_all.extend(N_NEURONS * [0])
            #             elif isinstance(model_config['t_stride'], list):
            #                 if layer_type == 'temporal':
            #                     l_idx_stride = l_idx
            #                     if l_idx >= n_tot_layer/2:
            #                         l_idx_stride = int(l_idx - n_tot_layer/2)
            #                     model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'][l_idx_stride])])
            #                     model_s_stride_all.extend(N_NEURONS * [0])
            #                 elif layer_type == 'spatial':
            #                     l_idx_stride = l_idx
            #                     if l_idx >= n_tot_layer/2:
            #                         l_idx_stride = int(l_idx - n_tot_layer/2)
            #                     model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'][l_idx_stride])])
            #                     model_t_stride_all.extend(N_NEURONS * [0])
            #                 elif layer_type == 'spatiotemporal':
            #                     model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'][l_idx])])
            #                     model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'][l_idx])])
            #             else:
            #                 if layer_type == 'temporal':
            #                     model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'])])
            #                     model_s_stride_all.extend(N_NEURONS * [0])
            #                 elif layer_type == 'spatial':
            #                     model_t_stride_all.extend(N_NEURONS * [0])
            #                     model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'])])
            #                 elif layer_type == 'spatiotemporal':
            #                     model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'])])
            #                     model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'])])

            #             if 'npplayers' in model_config.keys():
            #                 model_t_kernelsize_all.extend(N_NEURONS * [0])
            #                 model_s_kernelsize_all.extend(N_NEURONS * [int(model_config['s_kernelsize'])])
            #             else:
            #                 model_t_kernelsize_all.extend(N_NEURONS * [int(model_config['t_kernelsize'])])
            #                 model_s_kernelsize_all.extend(N_NEURONS * [int(model_config['s_kernelsize'])])
                        
            #             if 'npplayers' in model_config.keys():
            #                 model_n_tkernels_all.extend(N_NEURONS * [0])
            #                 if l_idx < n_tot_layer -1:
            #                     model_n_skernels_all.extend(N_NEURONS * [int(model_config['nppfilters'][l_idx])])
            #                 else:
            #                     model_n_skernels_all.extend(N_NEURONS * [0])
            #             else:
            #                 if layer_type == 'temporal':
            #                     l_idx_kernel = l_idx
            #                     if l_idx >= n_tot_layer/2:
            #                         l_idx_kernel = int(l_idx - n_tot_layer/2)
            #                     model_n_tkernels_all.extend(N_NEURONS * [int(model_config['n_tkernels'][l_idx_kernel])])
            #                     model_n_skernels_all.extend(N_NEURONS * [0])

            #                 if layer_type == 'spatial':
            #                     l_idx_kernel = l_idx
            #                     if l_idx >= n_tot_layer/2:
            #                         l_idx_kernel = int(l_idx - n_tot_layer/2)
            #                     model_n_skernels_all.extend(N_NEURONS * [int(model_config['n_skernels'][l_idx_kernel])])
            #                     model_n_tkernels_all.extend(N_NEURONS * [0])
                            
            #                 if layer_type == 'spatiotemporal':
            #                     model_n_skernels_all.extend(N_NEURONS * [int(model_config['n_skernels'][l_idx])])
            #                     model_n_tkernels_all.extend(N_NEURONS * [int(model_config['n_tkernels'][l_idx])])

            #             ev_train_all.extend(ev_lm_tr[l_idx, neurons_to_add])
            #             ev_test_all.extend(ev_lm_te[l_idx, neurons_to_add])
            #     except FileNotFoundError as e:
            #         print('Results not found', e)
            #         continue

    # CREATE DATAFRAME
    seqs = [neuron_ids, exp_id_all, model_task_all, model_name_all, arch_type_all,
            model_val_acc_all, model_test_acc_all, model_train_acc_all,
            monkey_all, session_all, area_all, co_task_all, model_layer_all, model_type_layer_all, model_max_layer_all,
            model_t_stride_all, model_s_stride_all, model_t_kernelsize_all, model_s_kernelsize_all, model_n_tkernels_all, model_n_skernels_all,
            ev_train_all, ev_test_all]

    for col, seq in zip(res_df.columns, seqs):
        # print('Adding col {} to DataFrame...'.format(col))
        res_df[col] = seq

    print('Results DataFrame done!')
    return res_df

def load_exp_results_datadriven(exp_id, monkey_session_tuples, normalize=False, load_passive=False, params_dict=None, tuned_ids=False,
                     train_iter=None,task_transfer=False,result_path='..'):
    ''' Load all single-neuron results for each experiment along with DNN model info.
    Arguments:
    exp_id - (int) Experiment ID.
    monkey_session_tuples - (list of tuples) (monkey, session) to include.
    '''
    # if path is not None:

    # Select experiment models
    if task_transfer:
        result_path = result_path[:-1] + '_task_transfer/'
        path_to_res = os.path.join(result_path, 'experiment_{}'.format(exp_id))
        # path_to_res = os.path.join(get_results_path_datadriven_tasktransfer(), 'experiment_{}'.format(exp_id))
    else:
        path_to_res = os.path.join(result_path, 'experiment_{}'.format(exp_id))
        # path_to_res = os.path.join(get_results_path_datadriven(), 'experiment_{}'.format(exp_id))
    model_list = os.listdir(path_to_res)
    model_list = [m for m in model_list if 'experiment_'.format(exp_id) not in m]

    # Init. dataframe cols
    df_columns = ['neuron_ids',
                  'exp_id',
                  'model_task',
                  'model_name',
                  'arch_type',
                  'model_test_acc',
                  'monkey',
                  'session',
                  'area',
                  'co_task',
                  'model_layer',
                  'layer_type',
                  'model_max_layer',
                  't_stride',
                  's_stride',
                  't_kernelsize',
                  's_kernelsize',
                  'n_tkernels',
                  'n_skernels',
                  'ev_train',
                  'ev_test']
    res_df = pd.DataFrame(columns=df_columns)

    neuron_ids = []
    exp_id_all = []
    model_name_all = []
    arch_type_all = []
    model_task_all = []
    model_test_acc_all = []
    monkey_all = []
    session_all = []
    area_all = []
    co_task_all = []
    model_layer_all = []
    model_max_layer_all = []
    model_type_layer_all = []
    model_max_layer_all = []
    model_t_stride_all = []
    model_s_stride_all = []
    model_t_kernelsize_all = []
    model_s_kernelsize_all = []
    model_n_tkernels_all = []
    model_n_skernels_all = []
    ev_train_all = []
    ev_test_all = []


    if params_dict is None: #default predictions
        params_dict = {'active_start':'mvt', 'active_length':0, 'align':100,
                'permut_m':False, 'permut_t':False, 'constant_input':False, 'suffix':None}

    for (monkey, session_date) in monkey_session_tuples:
        print('Loading results for {}...'.format(monkey))

        # LOAD TUNED NEURONS ONLY
        neurons_to_add = None
        if tuned_ids:
            path_to_tuned_ids = os.path.join(PATH_TO_NEURANALYSIS,
                                             'Results',
                                             'analysis_neurons_ranksum_multicorr_all_sessions.pkl')
            neuron_id_dict = np.load(path_to_tuned_ids, allow_pickle=True)
            if monkey == 'S1Lando':
                session_key = 'S1' + str(session_date)
            else:
                session_key = session_date
            tuned_ids = neuron_id_dict[session_key]
            neurons_to_add = tuned_ids
            print('Loading {} neuron ids:{}'.format(len(neurons_to_add), neurons_to_add))

        # LOAD MODEL RESULTS
        for model_name in model_list:

            # Load model config file
            path_to_exp_models = os.path.join(PATH_TO_MODELS, 'experiment_{}'.format(exp_id))
            path_to_config_file = os.path.join(path_to_exp_models, '{}/config.yaml'.format(model_name))
            with open(path_to_config_file, 'r') as myfile:
                model_config = yaml.safe_load(myfile)
                if exp_id in [4045, 4046, 10045] + list(range(10545,12045,100)) + list(range(11445,12045,100)) + list(range(11446,12046,100)) + list(range(11447,12047,100)) \
                             + list(range(13445,14045,100)) + list(range(14445,15045,100)) + list(range(14446,15046,100)) + list(range(14447,15047,100)):
                    l_key = 'npplayers'
                else:
                    l_key = 'nlayers'
                nlayers = model_config[l_key]

            # ACTIVE
            try:
                if train_iter is not None:
                    ev_lm_te, ev_lm_tr = load_ev_scores_datadriven(exp_id=exp_id, monkey_name=monkey, session_date=session_date,
                                                model_name=model_name, ispassive=False, normalize=normalize,
                                                    params_dict=params_dict, nlayers=nlayers, train_iter=train_iter, task_transfer=task_transfer,result_path=result_path)
                else:
                    ev_lm_te, ev_lm_tr = load_ev_scores_datadriven(exp_id=exp_id, monkey_name=monkey, session_date=session_date,
                                                model_name=model_name, ispassive=False, normalize=normalize,
                                                    params_dict=params_dict, nlayers=nlayers, task_transfer=task_transfer,result_path=result_path)


                if neurons_to_add is None:
                    neurons_to_add = np.arange(ev_lm_te.shape[1])
                    #print('Loading {} neuron ids:{}'.format(len(neurons_to_add), neurons_to_add))

                # LOAD RESULTS AND FIELDS
                N_NEURONS = len(neurons_to_add)

                neuron_ids.extend(neurons_to_add)
                exp_id_all.extend(N_NEURONS * [exp_id])
                model_name_all.extend(N_NEURONS * [model_name])

                if  'arch_type' in model_config.keys():
                    arch_type_all.extend(N_NEURONS * [model_config['arch_type']])
                else:
                    arch_type_all.extend(N_NEURONS * [model_config['rec_blocktype']])

                model_task_all.extend(N_NEURONS * [get_PCR_exp_type(exp_id)])
                model_test_acc_all.extend(N_NEURONS * [get_model_accuracy_datadriven(exp_id, model_name)[1]])
                monkey_all.extend(N_NEURONS * [monkey])
                session_all.extend(N_NEURONS * [session_date])
                area = 'CN' if monkey in get_CN_monkeys() else 'S1'
                area_all.extend(N_NEURONS * [area])
                co_task_all.extend(N_NEURONS * ['active'])

                if 'npplayers' in model_config.keys():
                    n_tot_layer = model_config['npplayers']
                elif model_config['arch_type'] == 'spatiotemporal':
                    n_tot_layer = model_config['nlayers']
                else:
                    n_tot_layer = 2 * model_config['nlayers']
                model_max_layer_all.extend(N_NEURONS * [n_tot_layer])

                l_idx = int(n_tot_layer -1)
                model_layer_all.extend(N_NEURONS * [l_idx])
                layer_type = get_model_layer_type(l_idx,n_tot_layer,model_name)
                model_type_layer_all.extend(N_NEURONS * [layer_type])

                if 'npplayers' in model_config.keys():
                    model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'])])
                    model_t_stride_all.extend(N_NEURONS * [0])
                elif isinstance(model_config['t_stride'], list):
                    if layer_type == 'temporal':
                        l_idx_stride = l_idx
                        if l_idx >= n_tot_layer/2:
                            l_idx_stride = int(l_idx - n_tot_layer/2)
                        model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'][l_idx_stride])])
                        model_s_stride_all.extend(N_NEURONS * [0])
                    elif layer_type == 'spatial':
                        l_idx_stride = l_idx
                        if l_idx >= n_tot_layer/2:
                            l_idx_stride = int(l_idx - n_tot_layer/2)
                        model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'][l_idx_stride])])
                        model_t_stride_all.extend(N_NEURONS * [0])
                    elif layer_type == 'spatiotemporal':
                        model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'][l_idx])])
                        model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'][l_idx])])
                else:
                    if layer_type == 'temporal':
                        model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'])])
                        model_s_stride_all.extend(N_NEURONS * [0])
                    elif layer_type == 'spatial':
                        model_t_stride_all.extend(N_NEURONS * [0])
                        model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'])])
                    elif layer_type == 'spatiotemporal':
                        model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'])])
                        model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'])])

                if 'npplayers' in model_config.keys():
                    model_t_kernelsize_all.extend(N_NEURONS * [0])
                    model_s_kernelsize_all.extend(N_NEURONS * [int(model_config['s_kernelsize'])])
                else:
                    model_t_kernelsize_all.extend(N_NEURONS * [int(model_config['t_kernelsize'])])
                    model_s_kernelsize_all.extend(N_NEURONS * [int(model_config['s_kernelsize'])])
                
                if 'npplayers' in model_config.keys():
                    model_n_tkernels_all.extend(N_NEURONS * [0])
                    if l_idx < n_tot_layer -1:
                        model_n_skernels_all.extend(N_NEURONS * [int(model_config['nppfilters'][l_idx])])
                    else:
                        model_n_skernels_all.extend(N_NEURONS * [0])
                else:
                    if layer_type == 'temporal':
                        l_idx_kernel = l_idx
                        if l_idx >= n_tot_layer/2:
                            l_idx_kernel = int(l_idx - n_tot_layer/2)
                        model_n_tkernels_all.extend(N_NEURONS * [int(model_config['n_tkernels'][l_idx_kernel])])
                        model_n_skernels_all.extend(N_NEURONS * [0])

                    if layer_type == 'spatial':
                        l_idx_kernel = l_idx
                        if l_idx >= n_tot_layer/2:
                            l_idx_kernel = int(l_idx - n_tot_layer/2)
                        model_n_skernels_all.extend(N_NEURONS * [int(model_config['n_skernels'][l_idx_kernel])])
                        model_n_tkernels_all.extend(N_NEURONS * [0])
                    
                    if layer_type == 'spatiotemporal':
                        model_n_skernels_all.extend(N_NEURONS * [int(model_config['n_skernels'][l_idx])])
                        model_n_tkernels_all.extend(N_NEURONS * [int(model_config['n_tkernels'][l_idx])])

                ev_train_all.extend(ev_lm_tr[0,neurons_to_add])
                ev_test_all.extend(ev_lm_te[0,neurons_to_add])

            except FileNotFoundError as e:
                print('Results not found', e)
                continue

            # PASSIVE
            if load_passive == True:
                try:
                    if train_iter is not None:
                        ev_lm_te, ev_lm_tr = load_ev_scores_datadriven(exp_id=exp_id, monkey_name=monkey, session_date=session_date,
                                                    model_name=model_name, ispassive=False, normalize=normalize,
                                                        params_dict=params_dict, nlayers=nlayers, train_iter=train_iter, task_transfer=task_transfer,result_path=result_path)
                    else:
                        ev_lm_te, ev_lm_tr = load_ev_scores_datadriven(exp_id=exp_id, monkey_name=monkey, session_date=session_date,
                                                model_name=model_name, ispassive=True, normalize=normalize,
                                                params_dict=params_dict, nlayers=nlayers, task_transfer=task_transfer,result_path=result_path)

                    # LOAD RESULTS AND FIELDS
                    N_NEURONS = len(neurons_to_add)
                    for l_idx in range(ev_lm_te.shape[0]):
                    # for l_idx in range(int(nlayers)):
                        neuron_ids.extend(neurons_to_add)
                        exp_id_all.extend(N_NEURONS * [exp_id])
                        model_name_all.extend(N_NEURONS * [model_name])

                        if 'arch_type' in model_config.keys():
                            arch_type_all.extend(N_NEURONS * [model_config['arch_type']])
                        else:
                            arch_type_all.extend(N_NEURONS * [model_config['rec_blocktype']])

                        model_task_all.extend(N_NEURONS * [get_PCR_exp_type(exp_id)])
                        model_test_acc_all.extend(N_NEURONS * [get_model_accuracy_datadriven(exp_id, model_name)[1]])
                        monkey_all.extend(N_NEURONS * [monkey])
                        session_all.extend(N_NEURONS * [session_date])
                        area = 'CN' if monkey in get_CN_monkeys() else 'S1'
                        area_all.extend(N_NEURONS * [area])
                        co_task_all.extend(N_NEURONS * ['passive'])

                        if 'npplayers' in model_config.keys():
                            n_tot_layer = model_config['npplayers']
                        elif model_config['arch_type'] == 'spatiotemporal':
                            n_tot_layer = model_config['nlayers']
                        else:
                            n_tot_layer = 2 * model_config['nlayers']
                        model_max_layer_all.extend(N_NEURONS * [n_tot_layer])

                        l_idx = int(n_tot_layer -1)
                        model_layer_all.extend(N_NEURONS * [l_idx])
                        layer_type = get_model_layer_type(l_idx,n_tot_layer,model_name)
                        model_type_layer_all.extend(N_NEURONS * [layer_type])

                        if 'npplayers' in model_config.keys():
                            model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'])])
                            model_t_stride_all.extend(N_NEURONS * [0])
                        elif isinstance(model_config['t_stride'], list):
                            if layer_type == 'temporal':
                                l_idx_stride = l_idx
                                if l_idx >= n_tot_layer/2:
                                    l_idx_stride = int(l_idx - n_tot_layer/2)
                                model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'][l_idx_stride])])
                                model_s_stride_all.extend(N_NEURONS * [0])
                            elif layer_type == 'spatial':
                                l_idx_stride = l_idx
                                if l_idx >= n_tot_layer/2:
                                    l_idx_stride = int(l_idx - n_tot_layer/2)
                                model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'][l_idx_stride])])
                                model_t_stride_all.extend(N_NEURONS * [0])
                            elif layer_type == 'spatiotemporal':
                                model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'][l_idx])])
                                model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'][l_idx])])
                        else:
                            if layer_type == 'temporal':
                                model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'])])
                                model_s_stride_all.extend(N_NEURONS * [0])
                            elif layer_type == 'spatial':
                                model_t_stride_all.extend(N_NEURONS * [0])
                                model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'])])
                            elif layer_type == 'spatiotemporal':
                                model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'])])
                                model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'])])

                        if 'npplayers' in model_config.keys():
                            model_t_kernelsize_all.extend(N_NEURONS * [0])
                            model_s_kernelsize_all.extend(N_NEURONS * [int(model_config['s_kernelsize'])])
                        else:
                            model_t_kernelsize_all.extend(N_NEURONS * [int(model_config['t_kernelsize'])])
                            model_s_kernelsize_all.extend(N_NEURONS * [int(model_config['s_kernelsize'])])
                        
                        if 'npplayers' in model_config.keys():
                            model_n_tkernels_all.extend(N_NEURONS * [0])
                            if l_idx < n_tot_layer -1:
                                model_n_skernels_all.extend(N_NEURONS * [int(model_config['nppfilters'][l_idx])])
                            else:
                                model_n_skernels_all.extend(N_NEURONS * [0])
                        else:
                            if layer_type == 'temporal':
                                l_idx_kernel = l_idx
                                if l_idx >= n_tot_layer/2:
                                    l_idx_kernel = int(l_idx - n_tot_layer/2)
                                model_n_tkernels_all.extend(N_NEURONS * [int(model_config['n_tkernels'][l_idx_kernel])])
                                model_n_skernels_all.extend(N_NEURONS * [0])

                            if layer_type == 'spatial':
                                l_idx_kernel = l_idx
                                if l_idx >= n_tot_layer/2:
                                    l_idx_kernel = int(l_idx - n_tot_layer/2)
                                model_n_skernels_all.extend(N_NEURONS * [int(model_config['n_skernels'][l_idx_kernel])])
                                model_n_tkernels_all.extend(N_NEURONS * [0])
                            
                            if layer_type == 'spatiotemporal':
                                model_n_skernels_all.extend(N_NEURONS * [int(model_config['n_skernels'][l_idx])])
                                model_n_tkernels_all.extend(N_NEURONS * [int(model_config['n_tkernels'][l_idx])])

                        ev_train_all.extend(ev_lm_tr[0,neurons_to_add])
                        ev_test_all.extend(ev_lm_te[0,neurons_to_add])
                except FileNotFoundError as e:
                    print('Results not found', e)
                    continue

    # CREATE DATAFRAME
    seqs = [neuron_ids, exp_id_all, model_task_all, model_name_all, arch_type_all,
            model_test_acc_all,
            monkey_all, session_all, area_all, co_task_all, model_layer_all, model_type_layer_all, model_max_layer_all,
            model_t_stride_all, model_s_stride_all, model_t_kernelsize_all, model_s_kernelsize_all, model_n_tkernels_all, model_n_skernels_all,
            ev_train_all, ev_test_all]

    for col, seq in zip(res_df.columns, seqs):
        print('Adding col {} to DataFrame...'.format(col))
        res_df[col] = seq

    print('Results DataFrame done!')
    return res_df

def load_exp_results_combined(exp_id_list, monkey_session_tuples, normalize=False, load_passive=False, params_dict=None, tuned_ids=False,
                     path=None, train_iter=None, result_path=None):
    ''' Load all single-neuron results for each experiment along with DNN model info.
    Arguments:
    exp_id - (int) Experiment ID.
    monkey_session_tuples - (list of tuples) (monkey, session) to include.
    '''

    # if result_path is not None:
    #     print('Updated result path!')
    #     def get_results_path():
    #         return result_path
    #     print(get_results_path())

    # Select experiment models
    exp_id_name = '_'.join(map(str, exp_id_list))
    exp_id = exp_id_list[0]


    # path_to_res = os.path.join(get_results_path(), 'experiment_{}'.format(exp_id))
    path_to_res = os.path.join(result_path, 'experiment_{}'.format(exp_id))
    model_list = os.listdir(path_to_res)
    model_list = [m for m in model_list if 'experiment_'.format(exp_id) not in m]
    # model_list = np.array([model.replace('_r_','_') for model in model_list])

    # Init. dataframe cols
    df_columns = ['neuron_ids',
                  'exp_id',
                  'model_task',
                  'model_name',
                  'arch_type',
                  'model_val_acc',
                  'model_test_acc',
                  'model_train_acc',
                  'monkey',
                  'session',
                  'area',
                  'co_task',
                  'model_layer',
                  'layer_type',
                  'model_max_layer',
                  't_stride',
                  's_stride',
                  't_kernelsize',
                  's_kernelsize',
                  'n_tkernels',
                  'n_skernels',
                  'ev_train',
                  'ev_test']
    res_df = pd.DataFrame(columns=df_columns)

    neuron_ids = []
    exp_id_all = []
    model_name_all = []
    arch_type_all = []
    model_task_all = []
    model_val_acc_all = []
    model_test_acc_all = []
    model_train_acc_all = []
    monkey_all = []
    session_all = []
    area_all = []
    co_task_all = []
    model_layer_all = []
    model_type_layer_all = []
    model_max_layer_all = []
    model_t_stride_all = []
    model_s_stride_all = []
    model_t_kernelsize_all = []
    model_s_kernelsize_all = []
    model_n_tkernels_all = []
    model_n_skernels_all = []
    ev_train_all = []
    ev_test_all = []


    if params_dict is None: #default predictions
        params_dict = {'active_start':'mvt', 'active_length':0, 'align':100,
                'permut_m':False, 'permut_t':False, 'constant_input':False, 'suffix':None}

    for (monkey, session_date) in monkey_session_tuples:
        print('Loading results for {}...'.format(monkey))

        # LOAD TUNED NEURONS ONLY
        neurons_to_add = None
        if tuned_ids:
            path_to_tuned_ids = os.path.join(PATH_TO_NEURANALYSIS,
                                             'Results',
                                             'analysis_neurons_ranksum_multicorr_all_sessions.pkl')
            neuron_id_dict = np.load(path_to_tuned_ids, allow_pickle=True)
            if monkey == 'S1Lando':
                session_key = 'S1' + str(session_date)
            else:
                session_key = session_date
            tuned_ids = neuron_id_dict[session_key]
            neurons_to_add = tuned_ids
            print('Loading {} neuron ids:{}'.format(len(neurons_to_add), neurons_to_add))

        # LOAD MODEL RESULTS
        all_task_name = get_PCR_exp_type_combined(exp_id_list)
        for model_name in model_list:
            
            model_name_config = model_name
            model_name = model_name.replace('_r_','_')
            
            # Load model config file
            path_to_exp_models = os.path.join(PATH_TO_MODELS, 'experiment_{}'.format(exp_id))

            # if exp_id in [4016,5016,4046]:
            #     path_to_act_exp1 = os.path.join(PATH_TO_MODELS, 'experiment_{}'.format(exp_id))
            #     model_list1 = os.listdir(path_to_act_exp1)
            #     model_list_names = np.array([model.replace('_r_','_') for model in model_list1])
            #     ind_name = np.where(model_list_names == model_name)[0]
            #     model_name_tmp = np.array(model_list1)[ind_name][0]
            #     path_to_config_file = os.path.join(path_to_exp_models, '{}/config.yaml'.format(model_name_tmp))
            # else:
            path_to_config_file = os.path.join(path_to_exp_models, '{}/config.yaml'.format(model_name_config))
            with open(path_to_config_file, 'r') as myfile:
                model_config = yaml.safe_load(myfile)
                if exp_id in [4045, 4046, 10045]:
                    l_key = 'npplayers'
                    nlayers = model_config[l_key] + 1
                else:
                    l_key = 'nlayers'
                    nlayers = model_config[l_key]


            # ACTIVE
            try:
                if train_iter is not None:
                    ev_lm_te, ev_lm_tr = load_ev_scores_combined(exp_id_list=exp_id_list, monkey_name=monkey, session_date=session_date,
                                                model_name=model_name, ispassive=False, normalize=normalize,
                                                    params_dict=params_dict, nlayers=nlayers, train_iter=train_iter,path_to_results=result_path)
                else:
                    ev_lm_te, ev_lm_tr = load_ev_scores_combined(exp_id_list=exp_id_list, monkey_name=monkey, session_date=session_date,
                                                model_name=model_name, ispassive=False, normalize=normalize,
                                                    params_dict=params_dict, nlayers=nlayers,path_to_results=result_path)
                
                # model_name = model_name.replace('_r_','_')
                if neurons_to_add is None:
                    neurons_to_add = np.arange(ev_lm_te.shape[1])
                    #print('Loading {} neuron ids:{}'.format(len(neurons_to_add), neurons_to_add))

                # LOAD RESULTS AND FIELDS
                N_NEURONS = len(neurons_to_add)
                val_acc_all, test_acc_all, train_acc_all = get_model_accuracy_combined(exp_id_list, model_name)
                for l_idx in range(ev_lm_te.shape[0]):
                # for l_idx in range(int(nlayers)):
                    #print('Loading results layer', l_idx, 'model_name', model_name)
                    neuron_ids.extend(neurons_to_add)
                    exp_id_all.extend(N_NEURONS * [exp_id_name])
                    model_name_all.extend(N_NEURONS * [model_name])

                    if  'arch_type' in model_config.keys():
                        arch_type_all.extend(N_NEURONS * [model_config['arch_type']])
                    else:
                        arch_type_all.extend(N_NEURONS * [model_config['rec_blocktype']])

                    model_task_all.extend(N_NEURONS * [all_task_name])
                    model_val_acc_all.extend(N_NEURONS * [val_acc_all])
                    model_test_acc_all.extend(N_NEURONS * [test_acc_all])
                    model_train_acc_all.extend(N_NEURONS * [train_acc_all])
                    monkey_all.extend(N_NEURONS * [monkey])
                    session_all.extend(N_NEURONS * [session_date])
                    area = 'CN' if monkey in get_CN_monkeys() else 'S1'
                    area_all.extend(N_NEURONS * [area])
                    co_task_all.extend(N_NEURONS * ['active'])
                    model_layer_all.extend(N_NEURONS * [l_idx])

                    if 'npplayers' in model_config.keys():
                        n_tot_layer = model_config['npplayers'] + 1
                    elif model_config['arch_type'] == 'spatiotemporal':
                        n_tot_layer = model_config['nlayers']
                    else:
                        n_tot_layer = 2 * model_config['nlayers']
                    model_max_layer_all.extend(N_NEURONS * [int(n_tot_layer)])

                    layer_type = get_model_layer_type(l_idx,n_tot_layer,model_name)
                    model_type_layer_all.extend(N_NEURONS * [layer_type])

                    if 'npplayers' in model_config.keys():
                        model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'])])
                        model_t_stride_all.extend(N_NEURONS * [0])
                    elif isinstance(model_config['t_stride'], list):
                        if layer_type == 'temporal':
                            l_idx_stride = l_idx
                            if l_idx >= n_tot_layer/2:
                                l_idx_stride = int(l_idx - n_tot_layer/2)
                            model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'][l_idx_stride])])
                            model_s_stride_all.extend(N_NEURONS * [0])
                        elif layer_type == 'spatial':
                            l_idx_stride = l_idx
                            if l_idx >= n_tot_layer/2:
                                l_idx_stride = int(l_idx - n_tot_layer/2)
                            model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'][l_idx_stride])])
                            model_t_stride_all.extend(N_NEURONS * [0])
                        elif layer_type == 'spatiotemporal':
                            model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'][l_idx])])
                            model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'][l_idx])])
                    else:
                        if layer_type == 'temporal':
                            model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'])])
                            model_s_stride_all.extend(N_NEURONS * [0])
                        elif layer_type == 'spatial':
                            model_t_stride_all.extend(N_NEURONS * [0])
                            model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'])])
                        elif layer_type == 'spatiotemporal':
                            model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'])])
                            model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'])])

                    if 'npplayers' in model_config.keys():
                        model_t_kernelsize_all.extend(N_NEURONS * [0])
                        model_s_kernelsize_all.extend(N_NEURONS * [int(model_config['s_kernelsize'])])
                    else:
                        model_t_kernelsize_all.extend(N_NEURONS * [int(model_config['t_kernelsize'])])
                        model_s_kernelsize_all.extend(N_NEURONS * [int(model_config['s_kernelsize'])])
                    
                    if 'npplayers' in model_config.keys():
                        model_n_tkernels_all.extend(N_NEURONS * [0])
                        if l_idx < n_tot_layer -1:
                            model_n_skernels_all.extend(N_NEURONS * [int(model_config['nppfilters'][l_idx])])
                        else:
                            model_n_skernels_all.extend(N_NEURONS * [0])
                    else:
                        if layer_type == 'temporal':
                            l_idx_kernel = l_idx
                            if l_idx >= n_tot_layer/2:
                                l_idx_kernel = int(l_idx - n_tot_layer/2)
                            model_n_tkernels_all.extend(N_NEURONS * [int(model_config['n_tkernels'][l_idx_kernel])])
                            model_n_skernels_all.extend(N_NEURONS * [0])

                        if layer_type == 'spatial':
                            l_idx_kernel = l_idx
                            if l_idx >= n_tot_layer/2:
                                l_idx_kernel = int(l_idx - n_tot_layer/2)
                            model_n_skernels_all.extend(N_NEURONS * [int(model_config['n_skernels'][l_idx_kernel])])
                            model_n_tkernels_all.extend(N_NEURONS * [0])
                        
                        if layer_type == 'spatiotemporal':
                            model_n_skernels_all.extend(N_NEURONS * [int(model_config['n_skernels'][l_idx])])
                            model_n_tkernels_all.extend(N_NEURONS * [int(model_config['n_tkernels'][l_idx])])
                        

                    ev_train_all.extend(ev_lm_tr[l_idx, neurons_to_add])
                    ev_test_all.extend(ev_lm_te[l_idx, neurons_to_add])

            except FileNotFoundError as e:
                print('Results not found', e)
                continue

            # PASSIVE
            if load_passive == True:
                try:
                    ev_lm_te, ev_lm_tr = load_ev_scores_combined(exp_id_list=exp_id_list, monkey_name=monkey, session_date=session_date,
                                              model_name=model_name, ispassive=True, normalize=normalize,
                                              params_dict=params_dict, nlayers=nlayers,result_path=result_path)

                    model_name = model_name.replace('_r_','_')
                    # LOAD RESULTS AND FIELDS
                    N_NEURONS = len(neurons_to_add)
                    val_acc_all, test_acc_all, train_acc_all = get_model_accuracy_combined(exp_id_list, model_name)
                    for l_idx in range(ev_lm_te.shape[0]):
                    # for l_idx in range(int(nlayers)):
                        neuron_ids.extend(neurons_to_add)
                        exp_id_all.extend(N_NEURONS * [exp_id_name])
                        model_name_all.extend(N_NEURONS * [model_name])

                        if 'arch_type' in model_config.keys():
                            arch_type_all.extend(N_NEURONS * [model_config['arch_type']])
                        else:
                            arch_type_all.extend(N_NEURONS * [model_config['rec_blocktype']])

                        model_task_all.extend(N_NEURONS * [all_task_name])
                        model_val_acc_all.extend(N_NEURONS * [val_acc_all])
                        model_test_acc_all.extend(N_NEURONS * [test_acc_all])
                        model_train_acc_all.extend(N_NEURONS * [train_acc_all])
                        monkey_all.extend(N_NEURONS * [monkey])
                        session_all.extend(N_NEURONS * [session_date])
                        area = 'CN' if monkey in get_CN_monkeys() else 'S1'
                        area_all.extend(N_NEURONS * [area])
                        co_task_all.extend(N_NEURONS * ['passive'])
                        model_layer_all.extend(N_NEURONS * [l_idx])

                        if 'npplayers' in model_config.keys():
                            n_tot_layer = model_config['npplayers']
                        elif model_config['arch_type'] == 'spatiotemporal':
                            n_tot_layer = model_config['nlayers']
                        else:
                            n_tot_layer = 2 * model_config['nlayers']
                        model_max_layer_all.extend(N_NEURONS * [int(n_tot_layer)])

                        model_type_layer_all.extend(N_NEURONS * [get_model_layer_type(l_idx,n_tot_layer,model_name)])

                        if 'npplayers' in model_config.keys():
                            model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'])])
                            model_t_stride_all.extend(N_NEURONS * [0])
                        elif isinstance(model_config['t_stride'], list):
                            if layer_type == 'temporal':
                                l_idx_stride = l_idx
                                if l_idx >= n_tot_layer/2:
                                    l_idx_stride = int(l_idx - n_tot_layer/2)
                                model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'][l_idx_stride])])
                                model_s_stride_all.extend(N_NEURONS * [0])
                            elif layer_type == 'spatial':
                                l_idx_stride = l_idx
                                if l_idx >= n_tot_layer/2:
                                    l_idx_stride = int(l_idx - n_tot_layer/2)
                                model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'][l_idx_stride])])
                                model_t_stride_all.extend(N_NEURONS * [0])
                            elif layer_type == 'spatiotemporal':
                                model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'][l_idx])])
                                model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'][l_idx])])
                        else:
                            if layer_type == 'temporal':
                                model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'])])
                                model_s_stride_all.extend(N_NEURONS * [0])
                            elif layer_type == 'spatial':
                                model_t_stride_all.extend(N_NEURONS * [0])
                                model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'])])
                            elif layer_type == 'spatiotemporal':
                                model_t_stride_all.extend(N_NEURONS * [int(model_config['t_stride'])])
                                model_s_stride_all.extend(N_NEURONS * [int(model_config['s_stride'])])

                        if 'npplayers' in model_config.keys():
                            model_t_kernelsize_all.extend(N_NEURONS * [0])
                            model_s_kernelsize_all.extend(N_NEURONS * [int(model_config['s_kernelsize'])])
                        else:
                            model_t_kernelsize_all.extend(N_NEURONS * [int(model_config['t_kernelsize'])])
                            model_s_kernelsize_all.extend(N_NEURONS * [int(model_config['s_kernelsize'])])
                        
                        if 'npplayers' in model_config.keys():
                            model_n_tkernels_all.extend(N_NEURONS * [0])
                            if l_idx < n_tot_layer -1:
                                model_n_skernels_all.extend(N_NEURONS * [int(model_config['nppfilters'][l_idx])])
                            else:
                                model_n_skernels_all.extend(N_NEURONS * [0])
                        else:
                            if layer_type == 'temporal':
                                l_idx_kernel = l_idx
                                if l_idx >= n_tot_layer/2:
                                    l_idx_kernel = int(l_idx - n_tot_layer/2)
                                model_n_tkernels_all.extend(N_NEURONS * [int(model_config['n_tkernels'][l_idx_kernel])])
                                model_n_skernels_all.extend(N_NEURONS * [0])

                            if layer_type == 'spatial':
                                l_idx_kernel = l_idx
                                if l_idx >= n_tot_layer/2:
                                    l_idx_kernel = int(l_idx - n_tot_layer/2)
                                model_n_skernels_all.extend(N_NEURONS * [int(model_config['n_skernels'][l_idx_kernel])])
                                model_n_tkernels_all.extend(N_NEURONS * [0])
                            
                            if layer_type == 'spatiotemporal':
                                model_n_skernels_all.extend(N_NEURONS * [int(model_config['n_skernels'][l_idx])])
                                model_n_tkernels_all.extend(N_NEURONS * [int(model_config['n_tkernels'][l_idx])])

                        ev_train_all.extend(ev_lm_tr[l_idx, neurons_to_add])
                        ev_test_all.extend(ev_lm_te[l_idx, neurons_to_add])
                except FileNotFoundError as e:
                    print('Results not found', e)
                    continue

    # CREATE DATAFRAME
    seqs = [neuron_ids, exp_id_all, model_task_all, model_name_all, arch_type_all,
            model_val_acc_all, model_test_acc_all, model_train_acc_all,
            monkey_all, session_all, area_all, co_task_all, model_layer_all, model_type_layer_all, model_max_layer_all,
            model_t_stride_all, model_s_stride_all, model_t_kernelsize_all, model_s_kernelsize_all, model_n_tkernels_all, model_n_skernels_all,
            ev_train_all, ev_test_all]

    for col, seq in zip(res_df.columns, seqs):
        print('Adding col {} to DataFrame...'.format(col))
        res_df[col] = seq

    print('Results DataFrame done!')
    return res_df

def make_gsresults_df(results_dict):
    '''Turn grid search dict results into DataFrame.'''
    #Init. hyperparameter search results dataframe
    
    windows = [3, 5, 7, 8, 10, 12, 15] #Same as ones used to fit
    delays = [0, 2, 4, 6, 8, 10] 

    data_cols_init = ['Alpha', 'Window', 'Latency', 'Score', 'Input']
    n_neurons = np.asarray(results_dict['test']['scores']).shape[-1]
    res_df = pd.DataFrame(columns=data_cols_init)

    scores_te = np.asarray(results_dict['test']['scores'])
    n_layers = scores_te.shape[0]
    alphas = np.asarray(results_dict['alphas'])

    for layer_idx in range(n_layers):
        #Add for each neuron
        for neur_idx in range(n_neurons):
            max_ids = np.unravel_index(indices=np.argmax(scores_te[layer_idx,:,:,neur_idx]), 
                                       shape=scores_te[layer_idx,:,:,neur_idx].shape)

            #Enter best found paramters for that NEURON
            df = pd.DataFrame(data=[[alphas[layer_idx,max_ids[0],max_ids[1],neur_idx], 
                                    windows[max_ids[0]],
                                    delays[max_ids[1]],
                                    np.max(scores_te[layer_idx,:,:,neur_idx]),
                                    'SC'+str(layer_idx)]],
                              columns=data_cols_init)
            #Add row
            res_df = res_df.append(df, ignore_index=True)

    #Make category type for plotting
    res_df['Input'] = res_df['Input'].astype('category')
    res_df['Window'] = res_df['Window'].astype('float')
    res_df['Latency'] = res_df['Latency'].astype('float')
    return res_df

def get_model_layer_type(l_idx, n_tot_layer, model_name):
    '''Get model layer type.'''

    if 'spatial_temporal' in model_name:
        if l_idx < n_tot_layer/2:
            layer_type =  'spatial'
        else:
            layer_type = 'temporal'
    elif 'temporal_spatial' in model_name:
        if l_idx < n_tot_layer/2:
            layer_type =  'temporal'
        else:
            layer_type = 'spatial'
    elif 'spatiotemporal' in model_name:
        layer_type =  'spatiotemporal'
    elif 'lstm' in model_name:
        if l_idx < n_tot_layer -1:
            layer_type =  'spatial'
        else:
            layer_type =  'recurrent_out'

    return layer_type

def get_model_accuracy(exp_type, model_name):
    '''Get model validation accuracy from config file.'''

    PATH_TO_TRAINED_MODELS = '..' ## Change path here with the one with task performance, in case change the script to use the result dataframe
    if 'lstm' in model_name:
        file_name = 'all_rec'
    else:
        file_name = 'all_tcns'
    
    if exp_type == 'classification':
        file_name += '_class.p'
    elif exp_type == 'regression':
        file_name += '_regress.p'
    elif exp_type == 'bt':
        file_name += '_barlow.p'

    path_to_df = os.path.join(PATH_TO_TRAINED_MODELS,file_name)
    all_models = pickle.load(open(path_to_df, 'rb'))
    model_tmp = all_models[all_models.model_name == model_name]

    if (exp_type == 'classification') or (exp_type == 'regression'):
        val_acc = model_tmp['validation_accuracy']
        test_acc = model_tmp['test_accuracy']
        train_acc = model_tmp['train_accuracy']
    elif exp_type == 'bt':
        val_acc = model_tmp['validation_loss']
        test_acc = model_tmp['test_loss']
        train_acc = model_tmp['train_loss']

    return float(val_acc), float(test_acc), float(train_acc)

def get_model_accuracy_general(exp_id, model_name):
    '''Get model validation accuracy from config file.'''

    path_to_config = os.path.join(PATH_TO_MODELS, 'experiment_{}'.format(exp_id), model_name, 'config.yaml')
    with open(path_to_config) as file:
        model_config = yaml.load(file, Loader=yaml.FullLoader)
        if not 'train_accuracy' in model_config.keys():
            val_acc = model_config['validation_accuracy']
            test_acc = model_config['test_accuracy']
            train_acc = 0
        else:
            val_acc = model_config['validation_accuracy']
            test_acc = model_config['test_accuracy']
            train_acc = model_config['train_accuracy']
        # if exp_id in [10015,10030,10045]:
        #     val_acc = model_config['validation_loss'] #no accuracy: e.g. BT network
        #     test_acc = model_config['test_loss']
        # elif exp_id in list(range(10515,12015,100)) + list(range(10530,12030,100)) + list(range(10545,12045,100)) \
        #                 + list(range(11416,12016,100)) + list(range(11420,12020,100)) + list(range(11421,12021,100))\
        #                 + list(range(11431,12031,100)) + list(range(11446,12046,100)) + list(range(11447,12047,100))\
        #                 + list(range(13415,14015,100)) + list(range(13430,14030,100)) + list(range(13445,14045,100)):
        #     val_acc = 0
        #     test_acc = model_config['test_accuracy']
        #     train_acc = 0

    #print(val_acc, test_acc)
    return float(val_acc), float(test_acc), float(train_acc)

def get_model_accuracy_datadriven_pca(exp_id, model_name):
    '''Get model validation accuracy from config file.'''

    path_to_config = os.path.join(PATH_TO_MODELS, 'experiment_{}'.format(exp_id), model_name, 'config.yaml')
    with open(path_to_config) as file:
        model_config = yaml.load(file) #, Loader=yaml.FullLoader)
        if 'validation_accuracy' in model_config.keys():
            val_acc = model_config['validation_accuracy']
            test_acc = model_config['test_accuracy']
        if exp_id in [10015,10030,10045]:
            val_acc = model_config['validation_loss'] #no accuracy: e.g. BT network
            test_acc = model_config['test_loss']
        elif exp_id in list(range(10515,12015,100)) + list(range(10530,12030,100)) + list(range(10545,12045,100)) \
                        + list(range(11416,12016,100)) + list(range(11420,12020,100)) + list(range(11421,12021,100))\
                        + list(range(11431,12031,100)) + list(range(11446,12046,100)) + list(range(11447,12047,100))\
                        + list(range(13415,14015,100)) + list(range(13430,14030,100)) + list(range(13445,14045,100)):
            val_acc = 0
            test_acc = model_config['test_accuracy']
            train_acc = 0

    #print(val_acc, test_acc)
    return float(val_acc), float(test_acc), float(train_acc)

def get_model_accuracy_datadriven(exp_id, model_name):
    '''Get model validation accuracy from config file.'''

    path_to_config = os.path.join(PATH_TO_MODELS, 'experiment_{}'.format(exp_id), model_name, 'config.yaml')
    with open(path_to_config) as file:
        model_config = yaml.load(file) #, Loader=yaml.FullLoader)
        if 'validation_accuracy' in model_config.keys():
            val_acc = model_config['validation_accuracy']
            test_acc = model_config['test_accuracy']
        if exp_id in [10015,10030,10045]:
            val_acc = model_config['validation_loss'] #no accuracy: e.g. BT network
            test_acc = model_config['test_loss']
        elif exp_id in list(range(10515,12015,100)) + list(range(10530,12030,100)) + list(range(10545,12045,100)) \
                        + list(range(11416,12016,100)) + list(range(11420,12020,100)) + list(range(11421,12021,100))\
                        + list(range(11431,12031,100)) + list(range(11446,12046,100)) + list(range(11447,12047,100))\
                        + list(range(14415,15015,100)) + list(range(14416,15016,100)) + list(range(14420,15020,100)) + list(range(14421,15021,100))\
                        + list(range(14430,15030,100)) + list(range(14431,15031,100)) + list(range(14445,15045,100)) + list(range(14446,15046,100)) + list(range(14447,15047,100))\
                        + list(range(13415,14015,100)) + list(range(13430,14030,100)) + list(range(13445,14045,100)):
            val_acc = 0
            test_acc = model_config['test_accuracy']

    #print(val_acc, test_acc)
    return float(val_acc), float(test_acc)

def get_model_accuracy_combined(exp_id_list, model_name):
    '''Get model validation accuracy from config file.'''

    PATH_TO_TRAINED_MODELS = '..'

    val_acc_all = []
    test_acc_all = []
    train_acc_all = []
    for exp_id in exp_id_list:
        exp_type = get_PCR_exp_type(exp_id)

        if 'lstm' in model_name:
            file_name = 'all_rec'
        else:
            file_name = 'all_tcns'

        if exp_type == 'classification':
            file_name += '_class.p'
        elif exp_type == 'regression':
            file_name += '_regress.p'
        elif exp_type == 'bt':
            file_name += '_barlow.p'
        
        path_to_df = os.path.join(PATH_TO_TRAINED_MODELS,file_name)
        all_models = pickle.load(open(path_to_df, 'rb'))

        model_list_names = [model.replace('_r_','_') for model in np.array(all_models['model_name'])]
        all_models['model_name'] = model_list_names

        model_tmp = all_models[all_models.model_name == model_name]

        if (exp_type == 'classification') or (exp_type == 'regression'):
            val_acc = model_tmp['validation_accuracy']
            test_acc = model_tmp['test_accuracy']
            train_acc = model_tmp['train_accuracy']
        elif exp_type == 'bt':
            val_acc = model_tmp['validation_loss']
            test_acc = model_tmp['test_loss']
            train_acc = model_tmp['train_loss']

        val_acc_all.append(float(val_acc))
        test_acc_all.append(float(test_acc))
        train_acc_all.append(float(train_acc))

    return val_acc_all, test_acc_all, train_acc_all

###---- SCORING/EV FUNCTIONS ----###

def normalize_ceiling(score_array, monkey_name, session_date, ispassive=False):
    '''Normalize scores of (n_layers, n_neurons) by neuron ceiling.'''
    #Load ceiling scores
    path_to_data = '..'
    if ispassive: task='pas'
    else: task='act'
    file_name = 'neuron_ceilings_{}_{}_{}.npy'.format(task, monkey_name, str(session_date))
    ceilings = np.load(os.path.join(path_to_data, file_name))
    try:
        print('Normalizing')
        score_array_norm = np.divide(np.asarray(score_array), ceilings)
    except:
        print('Cannot normalize')
        score_array_norm = score_array
    return score_array_norm

def spearmanbrown_correct(rho):
    'Spearman-Brown-corrected trial split correlation score.'
    return (rho*2)/(1+rho)

def get_splithalves(var, ax=0):
    '''Get split halves of var array along axis ax (trials).'''
    
    #Random shuffle along specified axis
    np.random.seed(42)
    np.apply_along_axis(np.random.shuffle, ax, var)   
   
    #Split array in equal sizes
    split1, split2 = np.array_split(var, 2, axis=ax) 
    
    #Compute means of splits i.e. mean spike counts - not so necessary 
    split_mean1 = np.nanmean(split1, axis=ax) 
    split_mean2 = np.nanmean(split2, axis=ax) 
    return split1, split2, split_mean1, split_mean2

def spearmanbrown_correction(var):
    '''Correct correlation value with Spearman-Brown formula.'''
    spc_var = (2*var)/(1+var)
    return spc_var

def get_splithalf_corr(var, ax=0, type='spearman'): 
    ''' Compute split half correlation of neural spike count data.'''

    #Split halves and get means of halves
    s1, s2, split_mean1, split_mean2 = get_splithalves(var, ax=ax)
    
    #Compute correlation score, of type
    if (type == 'spearman'):
        split_half_correlation = stats.spearmanr(s1, s2) #get the Spearman Correlation
    else:
        split_half_correlation = stats.pearsonr(s1, s2) #get the Pearson Correlation

    corr_dict = {'split_half_corr':split_half_correlation[0],
            'p-value':split_half_correlation[1],
            'type':type}
    return corr_dict

def get_reliabilities(Y_counts, X_mean_acts, glm_instance):
    '''Compute observed/predicted counts correlations for split halves (sh) over trial
    Arguments:
    spike_counts - (array) spike data data set
    X_activations - (array) network layer activations data set
    glm_instance - (PoissonRegressor instance) Poisson GLM instance to get best found regularization parameters
    '''
    #Split in halves over ENTIRE spike count
    split_1, split_2, s_mean_1, s_mean_2 = get_splithalves(Y_counts, ax=0)
    split_1_acts, split_2_acts, _, _ = get_splithalves(X_mean_acts, ax=0) 
    
    print('SH neural', split_1.shape, split_2.shape)
    print('SH activations', split_1_acts.shape, split_2_acts.shape)
    
    #Compute split-half CORRELATION of spike count data (neural)
    sh_corr = stats.spearmanr(split_1, split_2)[0]
    print('Neural corr', sh_corr)
    #sh_corr = get_splithalf_corr(Y_counts, ax=1)
    
    #Compute split-half predictions of spike (model)
    #Train GLMs with best found parameters from glm_instance & predict and score from halves
    #Split 1
    poisson_glm = PoissonRegressor(alpha=glm_instance.alpha, fit_intercept=True, tol=1e-4,
                                   warm_start=True, max_iter=10000)
    poisson_glm.fit(split_1_acts, split_1) #match size
    pred_1 = poisson_glm.predict(split_1_acts)
    #Split 2
    poisson_glm = PoissonRegressor(alpha=glm_instance.alpha, fit_intercept=True, tol=1e-4,
                                   warm_start=True, max_iter=10000)
    poisson_glm.fit(split_2_acts, split_2)
    pred_2 = poisson_glm.predict(split_2_acts)
    
    #Compute Spearman-Brown corrected correlation of predictions
    model_sh_score = spearmanbrown_correction(stats.pearsonr(pred_1.T,pred_2.T)[0])
    #neural_sh_score = spearmanbrown_correction(sh_corr['split_half_corr'])
    neural_sh_score = spearmanbrown_correction(sh_corr)
    
    return model_sh_score, neural_sh_score
    
def compute_EV(p_r2, rho_xx, rho_yy):
    '''Compute exlained variance for a neural site.
    Arguments:
    x_counts - (array) observed spike counts [samples x trials]
    y_pred  - (array) predicted spike counts by the Poisson GLM [samples x trials]
    rho_xx - internal reliability of observed spike counts
    rho_yy - internal reliability of predicted spike counts
    Returns:
    ev - (float) explained variance score, between 0 and 1
    raw_score - (float) raw McFadden's pseudo-R2 score
    corrected_raw_score - (float) noise corrected McFadden's pseudo-R2 score
    '''   
    
    print('p-r2', p_r2, 'rho_xx', rho_xx, 'rho_yy', rho_yy)
    #Get McFadden's pseudo-R2
    numerator = p_r2
    #Compute normalization factor
    denominator = np.sqrt(np.multiply(rho_xx, rho_yy))
    #Raw p-r2 score
    raw_score = numerator
    #Correct for variability/noise
    corrected_raw_score = numerator/denominator
    #Compute explained variance
    ev = corrected_raw_score ** 2
    return ev, raw_score, corrected_raw_score
    
###---- PLOTTING FUNCTIONS ----###

def plot_parallel_coord(hyperparams_df):
    '''Plot parallel coordinates plot to show grid search results.'''
    
    def nan_ptp(a):
        return np.ptp(a[np.isfinite(a)])

    cols = ['Alpha', 'Window', 'Latency', 'Score']
    x = [i for i, _ in enumerate(cols)]
    colours = ['maroon', 'crimson', 'darkorange', 'gold']
    colours = ['darkorchid', 'mediumorchid', 'orchid', 'palevioletred']



    # create dict of categories: colours
    colours = {hyperparams_df['Input'].cat.categories[i]: colours[i] for i, _ in enumerate(hyperparams_df['Input'].cat.categories)}

    # Create (X-1) sublots along x axis
    fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(15,7), dpi=150)

    # Get min, max and range for each column
    # Normalize the data for each column
    min_max_range = {}
    for col in cols:
        range_ = nan_ptp(hyperparams_df[col])
        min_max_range[col] = [np.nanmin(hyperparams_df[col]), np.nanmax(hyperparams_df[col]), range_]
        #Normalize data for each column
        col_norm = col+'_norm'
        hyperparams_df[col_norm] = np.true_divide(hyperparams_df[col] - np.nanmin(hyperparams_df[col]), 
                                                 range_)

        cols_norm = [col+'_norm' for col in cols] #added!    
    # Plot each row
    for i, ax in enumerate(axes):
        for idx in hyperparams_df.index:
            input_category = hyperparams_df.loc[idx, 'Input']
            ax.plot(x, hyperparams_df.loc[idx, cols_norm], colours[input_category], alpha=0.5)
        ax.set_xlim([x[i], x[i+1]])

    # Set the tick positions and labels on y axis for each plot
    # Tick positions based on normalised data
    # Tick labels are based on original data
    def set_ticks_for_axis(dim, ax, ticks):
        min_val, max_val, val_range = min_max_range[cols[dim]]
        step = val_range / float(ticks-1)
        tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
        norm_min =  np.nanmin(hyperparams_df[cols[dim]+'_norm'])
        norm_range = nan_ptp(hyperparams_df[cols[dim]+'_norm'])
        norm_step = norm_range / float(ticks-1)
        tick_values = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
        ax.yaxis.set_ticks(tick_values)
        ax.set_yticklabels(tick_labels)


    for dim, ax in enumerate(axes):
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        set_ticks_for_axis(dim, ax, ticks=5)
        ax.set_xticklabels([cols[dim]])


    # Move the final axis' ticks to the right-hand side
    ax = plt.twinx(axes[-1])
    dim = len(axes)
    ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    set_ticks_for_axis(dim, ax, ticks=5)
    ax.set_xticklabels([cols[-2], cols[-1]])

    # Remove space between subplots
    plt.subplots_adjust(wspace=0)

    # Add legend to plot
    plt.legend(
        [plt.Line2D((0,1),(0,0), color=colours[cat]) for cat in hyperparams_df['Input'].cat.categories],
        hyperparams_df['Input'].cat.categories, 
        title='Spatial layers', fontsize='medium',
        bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)

    plt.show()
    return