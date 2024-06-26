'''
Script to make predictions of one monkey session dataset given a neural network model.
'''

# Imports
import sys, os
import numpy as np
import h5py
import yaml
import json
import argparse
import pickle
import pandas as pd

# Modules

from neural_utils import has_passive_task
from predict_utils import get_dilation_factor
from make_layer_predictions_cv_align_passive import make_layer_predictions

sys.path.append('../../code/')
from path_utils import MODELS_DIR, PATH_TO_NEURAL_DATA, PATH_TO_ACTIVATIONS, PATH_TO_PREDICTIONS

# Paths

def main(monkey_name, session_date, exp_id, train_iter, model_name, window, latency,
         active_start, active_length, align, control_dict, shuffle_flag, path_to_act=None, path_to_results=None):
    '''
    '''

    path_to_models_exp = os.path.join(MODELS_DIR, 'experiment_{}'.format(exp_id))
    path_to_act_exp = os.path.join(PATH_TO_ACTIVATIONS, 'passive', 'experiment_{}'.format(exp_id))
    path_to_res_exp = os.path.join(PATH_TO_PREDICTIONS, 'passive', 'experiment_{}'.format(exp_id))

    path_to_model_results = os.path.join(path_to_res_exp, model_name)
    param_suffix = 'w{}l{}'.format(window, latency)

    path_to_model_results_search = os.path.join(path_to_model_results, '_' + param_suffix)
    if not os.path.exists(path_to_model_results_search):
        os.makedirs(path_to_model_results_search)
        print('New predictions: folder created.')

    # Check whether passive task is in monkey-session data
    has_passive = has_passive_task(session_date)

    # LOAD MONKEY SESSION DATASETS
    # Get filename suffixes
    active_start_suff = '_' + str(active_start)
    if active_length == 0:
        active_length_suff = '_end'
    else:
        active_length_suff = '_' + str(active_length) + 'ms'
    align_suff = '_at' + str(align)
    permut_suff = ''
    if control_dict['permut_m'] or control_dict['permut_t']:
        permut_suff = '_'
    if control_dict['permut_m']:
        permut_suff += 'M'
    if control_dict['permut_t']:
        permut_suff += 'T'
    const_suff = ''
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

    if has_passive:
        file_name = '{}_{}_passive{}.hdf5'.format(monkey_name, session_date, file_name_suffix_passive)
        print('Loading passive neural dataset:', file_name)
        with h5py.File(os.path.join(PATH_TO_NEURAL_DATA, file_name), 'r') as f:
            spike_dataset_pas = f['spike_counts'][()]
            trial_durations_pas = f['trial_durations'][()]

    print('Neural dataset dimensions (active | passive):', spike_dataset_pas.shape)

    print('Applying timescale parameters: window and latency')

    # GET MODEL CONFIG AND ACTIVATIONS
    path_to_model_act = os.path.join(path_to_act_exp, model_name)

    # LOAD CONFIG FILE
    if train_iter is not None:
        try:
            path_to_model_config = os.path.join(path_to_models_exp, '{}/config_{}.yaml'.format(model_name, int(train_iter)))
        except FileNotFoundError as err:
            print('Error: config file at training checkpoint {} does not exist!'.format(train_iter), err)
            return
    else:
        path_to_model_config = os.path.join(path_to_models_exp, '{}/config.yaml'.format(model_name))

    ## Load config file
    try:
        with open(path_to_model_config, 'r') as myfile:
            model_config = yaml.safe_load(myfile)
            print('Model config:', model_config)
    except:
        print('config missing for: ',model_name)
        return

    ## Check if predictions are really done

    pred_model_files = os.listdir(path_to_model_results_search)
    
    if train_iter is not None:
        pred_m_files = [i for i in pred_model_files if monkey_name in i]
        pred_m_files = [i for i in pred_m_files if '_ckpt'+str(train_iter) in i ]
        pred_m_files = [i for i in pred_m_files if 'passive' in i ]
        if 'S1' not in monkey_name:
            pred_m_files = [i for i in pred_m_files if not 'S1' in i ]
    else:
        pred_m_files = [i for i in pred_model_files if monkey_name in i]
        pred_m_files = [i for i in pred_m_files if not '_ckpt' in i ]
        pred_m_files = [i for i in pred_m_files if 'passive' in i ]
        if 'S1' not in monkey_name:
            pred_m_files = [i for i in pred_m_files if not 'S1' in i ]
    
    try:
        if model_config['arch_type'] == 'spatiotemporal':
            n_files = int(float(model_config['nlayers']))
        else:
            n_files = int(float(model_config['nlayers']))*2
    except:
        if model_config['rec_blocktype'] == 'lstm':
            n_files = int(model_config['npplayers']) +1

    ## Decomment here!!
    if len(pred_m_files) == n_files:
        print(pred_m_files)
        print('Already done!!')
        return

    if (session_date in model_config.keys()) or (str(session_date) + '_ale' in model_config.keys()):
        print('redo only for now')
    else:
        print('No activations for that model yet. Skipping.')
        # return

    # Get max nlayers
    N_MAX_LAYERS = None
    if 'lstm' not in model_name:
        n_layers = float(model_config['nlayers'])
        if model_config['arch_type'] == 'spatiotemporal':
            N_MAX_LAYERS = n_layers
        else:
            N_MAX_LAYERS = 2 * n_layers
    elif 'lstm' in model_name:
        N_MAX_LAYERS = float(model_config['npplayers']) + 1

    # ITERATE OVER MODEL LAYERS

    # Pick checkpoint activations
    if train_iter is not None:
        file_name_suffix_active += '_ckpt{}'.format(int(train_iter))
        file_name_suffix_passive += '_ckpt{}'.format(int(train_iter))

    if monkey_name == 'Butter':
            bad_trials_path = os.path.join(PATH_TO_SPIKE_REGRESS_DATA,
                                'excludedtrials_{}_{}_passive.p'.format(monkey_name, session_date))
            excluded_trials = pd.read_pickle(bad_trials_path)
            all_bad_trials = np.array(excluded_trials['bad_trials'])
            spike_dataset_pas = np.delete(spike_dataset_pas, all_bad_trials, axis=0)
    
    for layer_n in range(int(N_MAX_LAYERS)):
        print('--- Layer', layer_n)

        temp_factor = 1 #define temporal dilation factor

        # Update layer temporal dilation
        l_t_stride = get_dilation_factor(model_config, layer_n)
        temp_factor = temp_factor * l_t_stride


        # PASSIVE BUMPS


        if has_passive:
            print('- Passive predictions -')
            layer_file = '{}_{}_passive_l{}{}.h5'.format(monkey_name, session_date, layer_n,
                                                            file_name_suffix_passive)

            try:
                datafile = h5py.File(os.path.join(path_to_model_act, layer_file), 'r')
                layer_activations = datafile['layer_pca'][()] #layer_pca
                # layer_activations = np.load(os.path.join(path_to_model_act, layer_file), allow_pickle=True)
            except FileNotFoundError as e:
                print(e, 'Activations missing! Skipping.')
                continue
            
            if monkey_name == 'Butter':
                layer_activations = np.delete(layer_activations, all_bad_trials, axis=0)

            assert len(layer_activations) == len(spike_dataset_pas)

            layer_result_passive = make_layer_predictions(layer_activations,
                                                              spike_dataset_pas,
                                                              trial_durations_pas,
                                                              monkey_name,
                                                              session_date,
                                                              model_config,
                                                              layer_n,
                                                              is_passive=True,
                                                              align=align,
                                                              current_temp_factor=temp_factor)

            result_file = 'passive_{}_{}_l{}_{}{}.h5'.format(monkey_name, session_date, layer_n, param_suffix, file_name_suffix_passive)
            shape_test_rates = np.array(layer_result_passive['lm']['test_rates']).shape
            shape_test_preds = np.array(layer_result_passive['lm']['test_preds']).shape
            shape_intercepts = np.array(layer_result_passive['lm']['intercepts']).shape
            shape_weights = np.array(layer_result_passive['lm']['weights']).shape
            shape_alphas = np.array(layer_result_passive['lm']['alphas']).shape
            shape_ev_train = np.array(layer_result_passive['lm']['ev_train']).shape
            shape_ev_test = np.array(layer_result_passive['lm']['ev_test']).shape

            ### DECOMMENT HERE
            with h5py.File(os.path.join(path_to_model_results_search, result_file), 'w') as file:
                file.create_dataset('test_rates', data=np.array(layer_result_passive['lm']['test_rates']), chunks=(1,shape_test_rates[1]), maxshape=(None,shape_test_rates[1]),compression="gzip", dtype='float32')
                file.create_dataset('test_preds', data=np.array(layer_result_passive['lm']['test_preds']), chunks=(1,shape_test_preds[1]), maxshape=(None,shape_test_preds[1]),compression="gzip", dtype='float32')
                file.create_dataset('intercepts', data=np.array(layer_result_passive['lm']['intercepts']), chunks=(1,), maxshape=(None,),compression="gzip", dtype='float32')
                file.create_dataset('weights', data=np.array(layer_result_passive['lm']['weights']), chunks=(1,shape_weights[1]), maxshape=(None,shape_weights[1]),compression="gzip", dtype='float32')
                file.create_dataset('alphas', data=np.array(layer_result_passive['lm']['alphas']), chunks=(1,), maxshape=(None,),compression="gzip", dtype='float32')
                file.create_dataset('ev_train', data=np.array(layer_result_passive['lm']['ev_train']), chunks=(1,), maxshape=(None,),compression="gzip", dtype='float32')
                file.create_dataset('ev_test', data=np.array(layer_result_passive['lm']['ev_test']), chunks=(1,), maxshape=(None,),compression="gzip", dtype='float32')
            print('Saved in :', os.path.join(path_to_model_results_search, result_file))

    print('Neural predictions for {} session {} with {} done!'.format(monkey_name, session_date, model_name))

    # UPDATE CONFIG FILE
    with open(path_to_model_config, 'w') as myfile:
        # model_config[session_date] = 2 #neural predictions done
        model_config[str(session_date) + '_ale'] = 2
        yaml.safe_dump(model_config, myfile)


    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predictivity script for one DNN model for one monkey session data.')

    parser.add_argument('--monkey', type=str, help='Which monkey?', required=True)
    parser.add_argument('--session', type=int, help='Which session data?', required=True)
    parser.add_argument('--exp_id', type=int, help='Which experiment ?', required=True)
    parser.add_argument('--train_iter', type=int, default=None, help='Which training checkpoint index?', required=False)
    parser.add_argument('--model', type=str, help='Which neural network model?', required=True)
    parser.add_argument('--window', type=int, default=5, help='Which window param?', required=False)
    parser.add_argument('--latency', type=int, default=0, help='Which latency param?', required=False)
    parser.add_argument('--active_start', type=str, help='Which active start index?', required=True)
    parser.add_argument('--active_length', type=int, default=0, help='Length after movement onset (1bin=10ms)? [None if passive/hold].', required=False)
    parser.add_argument('--align', type=int, default=0, help='Index of trial onset alignment.', required=False)

    args = parser.parse_args()

    main(args.monkey, args.session, args.exp_id, args.train_iter,
         args.model, args.window, args.latency,
         args.active_start, args.active_length, args.align)