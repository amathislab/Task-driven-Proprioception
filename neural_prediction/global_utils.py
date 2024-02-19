### IMPORTS
import os
import numpy as np
import yaml

####---- GENERAL ----####

def get_neuraldata_path():
    ''' Get path to neural data folder.'''
    return '..'

def get_deepdrawdata_path():
    ''' Get path to DeepDraw data folder.'''
    return '..'

####---- COLOR DEFINITIONS ----####

def get_colors():
    ''' Get color dictionary.'''
    color_dict = {'firing_rate':'crimson', 'monkey': 'darkmagenta', 'human':'deepforest'}
    return color_dict

####---- EXPERIMENT SORTING ----####


def is_CObump_session(session_date):
    '''To know if monkey session data has both passive and active tasks.'''

    non_CObump = [20170105, 20170203]
    if session_date in non_CObump:
        is_CObump = False
    else:
        is_CObump = True
    return is_CObump


def get_PCR_exp_type(exp_id):
    '''Get PCR task type.'''

    if exp_id in [2,4,6,8,11] + [4016] + [5016] + [16016] + list(range(6006, 6009)) + list(range(8000, 8011)) + [4046] + list(range(12616,13016,100)) + list(range(12631,13031,100)):
        return 'regression'
    elif exp_id in [15,115,45]: # + list(range(10000,10060,15)):
        return 'untrained'
    elif exp_id in [1,7,71] + [2000] + [4015,5015] + [16015] + [4060,4061] + list(range(4000,4011)) + [4045] + list(range(12615,13015,100)) + list(range(12630,13030,100)) + [16015]: # + list(range(10000,10060,15)):
        return 'classification'
    elif exp_id in list(range(10000,10060,15)) + list(range(12620,13020,100)) + list(range(12621,13021,100)) + [16020]:
        return 'bt'
    elif exp_id in list(range(10515,12015,100)) + list(range(10530,12030,100)) + list(range(10545,12045,100)) \
                    + list(range(14515,15015,100)) + list(range(14516,15016,100)) + list(range(14520,15020,100)) \
                    + list(range(14521,15021,100)) + list(range(14530,15030,100)) + list(range(14531,15031,100)) \
                    + list(range(14545,15045,100)) + list(range(14546,15046,100)) + list(range(14547,15047,100)) \
                   + list(range(13415,14015,100)) + list(range(13430,14030,100)) + list(range(13445,14045,100)):
        return 'data_driven'
    elif exp_id in [8015,8030,8045]:
        return 'torque'
    elif exp_id in [15016,15031]:
        return 'joints_input'
    elif exp_id in [17016,17031,17046]:
        return 'regress_joints_pos'
    elif exp_id in [17116,17131,17146]:
        return 'autoencoder'
    elif exp_id in [17216,17231,17246]:
        return 'regress_joints_vel'
    elif exp_id in [17316,17331,17346]:
        return 'regress_ee_vel'
    elif exp_id in [17416,17431,17446]:
        return 'regress_joints_pos_vel'
    elif exp_id in [17516,17531,17546]:
        return 'regress_ee_pos_vel'
    elif exp_id in [17816,17831,17846]:
        return 'regress_ee_pos_vel_acc'
    elif exp_id in [20016,20031,20046]:
        return 'regress_ee_elbow_pos'
    elif exp_id in [20416]:
        return 'regress_ee_pos_vel_acc'
    elif exp_id in [20516,20531,20546]:
        return 'regress_joints_pos_vel_acc'
    elif exp_id in [20616,20631,20646]:
        return 'regress_ee_elbow_pos_vel'
    elif exp_id in [20716,20717,20731,20732,20746]:
        return 'autoencoder'
    elif exp_id in [20816,20831,20846]:
        return 'regress_ee_elbow_pos_vel_acc'
    elif exp_id in [20916,20931,20946]:
        return 'regress_ee_elbow_vel'
    elif exp_id in list(range(40000,40099,1)):
        return 'regression'
    elif exp_id in list(range(40100,40199,1)):
        return 'regress_ee_pos_vel'
    elif exp_id in list(range(40200,40299,1)):
        return 'regress_ee_pos_vel_acc'
    elif exp_id in list(range(40300,40399,1)):
        return 'regress_ee_elbow_pos'
    elif exp_id in list(range(40400,40499,1)):
        return 'regress_ee_elbow_pos_vel'
    elif exp_id in list(range(40500,40599,1)):
        return 'regress_ee_elbow_pos_vel_acc'
    elif exp_id in list(range(40600,40699,1)):
        return 'regress_joints_pos'
    elif exp_id in list(range(40700,40799,1)):
        return 'regress_joints_pos_vel'
    elif exp_id in list(range(40800,40899,1)):
        return 'regress_joints_pos_vel_acc'
    elif exp_id in list(range(40900,40999,1)):
        return 'regress_ee_elbow_vel'
    elif exp_id in list(range(41000,41099,1)):
        return 'regress_ee_vel'
    elif exp_id in list(range(41100,41199,1)):
        return 'torque'
    elif exp_id in list(range(41200,41299,1)):
        return 'autoencoder'
    elif exp_id in list(range(41300,41399,1)):
        return 'regress_joints_vel'
    elif exp_id in list(range(50000,50099,1)):
        return 'regression'


def get_PCR_exp_type_combined(exp_id_list):
    '''Get PCR task type.'''
    
    all_task = []
    for exp_id in exp_id_list:
        if exp_id in [2,4,6,8,11] + [4016] + [5016] +list(range(6006, 6009)) + list(range(8000, 8011)) + [4046]:
            all_task.append('regression')
        elif exp_id in [1,7,71] + [2000] + [4015] + [5015] + list(range(4000,4011)) + [4045]:
            all_task.append('classification')
        elif exp_id in list(range(10000,10060,15)):
            all_task.append('bt')
        elif exp_id in list(range(10515,12015,100)) + list(range(10530,12030,100)) + list(range(10545,12045,100)):
            all_task.append('data_driven')
    all_task_name = '_'.join(map(str, all_task))
    return all_task_name

def get_network_family(exp_id):
    '''Get PCR task type.'''
    if exp_id in [71, 72, 1110, 4045, 4046]:
        return 'rec'
    else:
        return 'conv'

####---- DATA FORMATTING ----####

from json import JSONEncoder
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def load_model_config(path_to_config_file):
    '''Load model config given path, and makes correct keys are included and correct dtype.'''
    with open(path_to_config_file, 'r') as myfile:
        model_config = yaml.load(myfile, Loader=yaml.Loader)
        keys = [k for k in model_config.keys() if k not in ['experiment_id']]
        for key in keys:
            if key in ['n_tkernels', 'n_skernels']: #convert kernels numbers to int
                val = model_config[key]
                for i in range(len(val)):
                    val[i] = int(val[i])
            try:
                model_config[key] = float(model_config[key])# Make sure it's not strings
            except (TypeError, ValueError) as err:
                print('Error loading model configuration:', err)
                continue
    return model_config

def floatify_keys(model_config_dict):
    keys = [k for k in model_config_dict.keys() if k not in ['experiment_id']]
    for key in keys:
        try:
            model_config_dict[key] = float(model_config_dict[key])  # Make sure it's not strings
        except (TypeError, ValueError) as err:
            print('Skip config key:', key, err)
    return model_config_dict

def check_modelconfig_status(model_config_dict, session_date, monkey_name):
    ''' Check if model_config dict (any train iter) has activations for a session date and monkey name.'''

    print('Does this model have activations already?')
    model_name = model_config_dict['name']

    skip_model=False
    if session_date in model_config_dict.keys():
        if ((model_config_dict[session_date] == 1)
            or (model_config_dict[session_date] == '1.0')) \
                and (monkey_name != 'S1Lando'):

            print('Config:', model_config_dict[session_date],
                  'Activations done for', model_name, session_date,
                  ' - skipping model.')
            skip_model=True
        elif ((model_config_dict[session_date] == 2) or (model_config_dict[session_date] == '2.0')):
            print('Config:', model_config_dict[session_date],
                  'Predictions done for', model_name, session_date,
                  ' - skipping model.')
            skip_model=True

    return skip_model