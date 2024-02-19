import os
import random
import argparse
import time

import pandas
import pickle

import sys
sys.path.append('../code')
sys.path.append('./code')
from nn_rmodels_w_outputs import ConvRModel, RecurrentRModel, ConvRModel_new#, RecurrentRModel_new
from nn_train_rutils_spikes import *

from path_utils import PATH_TO_DATA_SPIKES, PATH_TO_OLD, PATH_TO_RESULTS_DATADRIVEN

import json
from json import JSONEncoder
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def load_df(model_type, arch_type, monkey, old_step=None):
    """ Function to load dataframe based on model type and architecture type.
    Outputs:
    - all_conv_models: dataframe containing all the networks to train
    - key_trained: key to set when train is finished
    """
    folder = monkey.lower()

    if 'han' in folder:
        folder = 'han11_22'

    if old_step is not None:
        suffix = '_' + str(old_step)
    else:
        suffix = ''

    if model_type == 'conv_new':
        all_conv_models = pickle.load(open(PATH_TO_OLD + '/' + folder + suffix + '/ALL_' + arch_type + '_newhyp_seed.p', 'rb'))
    elif model_type == 'conv':
        all_conv_models = pickle.load(open(PATH_TO_OLD + '/' + folder + suffix + '/ALL_' + arch_type + '_seed.p', 'rb'))
    elif model_type == 'rec':
        all_conv_models = pickle.load(open(PATH_TO_OLD + '/' + folder + suffix + '/ALL_' + arch_type + '_seed.p', 'rb'))
    elif model_type == 'rec_new':
        all_conv_models = pickle.load(open(PATH_TO_OLD + '/' + folder + suffix + '/ALL_' + arch_type + '_newhyp_seed.p', 'rb'))

    if (model_type == 'conv_new') or (model_type == 'conv'):
        best_models_arch = all_conv_models[all_conv_models['arch_type'] == arch_type] #.nlargest(1, 'test_accuracy')
        key_trained = 'is_trained'
    else:
        best_models_arch = all_conv_models[all_conv_models['rec_blocktype'] == arch_type]
        key_trained = 'is_training'
    return best_models_arch, key_trained

def save_df(best_models_arch, model_type, arch_type, monkey, old_step=None):
    """ Function to save the updated dataframe.
    """
    folder = monkey.lower()

    if 'han' in folder:
        folder = 'han11_22'

    if old_step is not None:
        suffix = '_' + str(old_step)
    else:
        suffix = ''

    if model_type == 'conv_new':
        best_models_arch.to_pickle(PATH_TO_OLD + '/' + folder + suffix + '/ALL_' + arch_type + '_newhyp_seed.p')
    elif model_type == 'conv':
        best_models_arch.to_pickle(PATH_TO_OLD + '/' + folder + suffix + '/ALL_' + arch_type + '_seed.p')
    elif model_type == 'rec':
        best_models_arch.to_pickle(PATH_TO_OLD + '/' + folder + suffix + '/ALL_' + arch_type + '_seed.p')
    elif model_type == 'rec_new':
        best_models_arch.to_pickle(PATH_TO_OLD + '/' + folder + suffix + '/ALL_' + arch_type + '_newhyp_seed.p')
    return 

def sel_exp_id(model_type):
    """Return the experiment ID for loading the same initialization

    Args:
        model_type (str): Define the model type

    Returns:
        int: Experiment ID of the corresponding untrained models
    """
    if model_type == 'conv_new':
        old_exp_id = 115
    elif model_type == 'conv':
        old_exp_id = 15
    elif model_type == 'rec':
        old_exp_id = 45
    elif model_type == 'rec_new':
        old_exp_id = 5045
    return old_exp_id

def main(args):
    # Load dataset
    exp_id = args.exp_id
    monkey = args.monkey
    session = str(args.session)
    lr = args.lr
    window = args.window
    latency = args.latency

    ### Path for dataset

    train_data_path = os.path.join(PATH_TO_DATA_SPIKES, 'dataset_train_' + monkey + '_' + session + '_center_out_spikes_datadriven.hdf5')
    latent_data_path = os.path.join(PATH_TO_DATA_SPIKES, 'latents_' + monkey + '_' + session + '_center_out_spikes_datadriven.p')
    train_data = RDataset(train_data_path, latent_data_path, 'train', key = 'spindle_info', target_key='spike_info')

    test_data_path = os.path.join(PATH_TO_DATA_SPIKES, 'dataset_test_' + monkey + '_' + session + '_center_out_spikes_datadriven.hdf5')
    test_data = RDataset(test_data_path, latent_data_path, dataset_type='test', key = 'spindle_info', target_key='spike_info')
    
    n_outputs = train_data.train_targets.shape[1]
    
    start_id = args.start_id
    end_id = args.end_id
    model_type = args.type
    arch_type = args.arch_type

    best_models_arch, key_trained = load_df(model_type, arch_type, monkey)

    old_exp_id = sel_exp_id(model_type)
    
    for i in range(start_id, end_id):
        print('---------------------------------')
        print('Training model: ', i)
        print('---------------------------------')

        ## Decomment this
        if not best_models_arch.iloc[i][key_trained]:
            latents = best_models_arch.iloc[i]
            
            if model_type == 'conv':
                # Create model
                mymodel = ConvRModel(
                    experiment_id=exp_id, #i
                    n_outputs=n_outputs,
                    arch_type=arch_type,
                    nlayers=latents['nlayers'],
                    n_skernels=latents['n_skernels'],
                    n_tkernels=latents['n_tkernels'],
                    s_kernelsize=latents['s_kernelsize'],
                    t_kernelsize=latents['t_kernelsize'],
                    s_stride=latents['s_stride'],
                    t_stride=latents['t_stride']) 

            elif model_type == 'conv_new':
                # Create model
                mymodel = ConvRModel_new(
                    experiment_id=exp_id, #i
                    n_outputs=n_outputs,
                    arch_type=arch_type,
                    nlayers=latents['nlayers'],
                    n_skernels=latents['n_skernels'],
                    n_tkernels=latents['n_tkernels'],
                    s_kernelsize=latents['s_kernelsize'],
                    t_kernelsize=latents['t_kernelsize'],
                    s_stride=latents['s_stride'],
                    t_stride=latents['t_stride']) #

            elif model_type == 'rec':
                mymodel = RecurrentRModel(
                    experiment_id=args.exp_id,
                    n_outputs=n_outputs,
                    rec_blocktype=arch_type,
                    n_recunits=latents['n_recunits'],
                    npplayers=latents['npplayers'],
                    nppfilters=latents['nppfilters'],
                    s_kernelsize=latents['s_kernelsize'],
                    s_stride=latents['s_stride'],
                    seed=latents['seed'])

            print(mymodel.__dict__)

            intime = time.time()

            ## Path for saving prediction result
            path_to_res_exp = os.path.join(PATH_TO_RESULTS_DATADRIVEN, 'experiment_{}'.format(args.exp_id))

            path_to_model_results = os.path.join(path_to_res_exp, mymodel.name)
            param_suffix = 'w{}l{}'.format(window, latency)

            path_to_model_results_search = os.path.join(path_to_model_results, '_' + param_suffix)
            if not os.path.exists(path_to_model_results_search):
                os.makedirs(path_to_model_results_search)
                print('New predictions: folder created.')

            # Create trainer and train!
            mytrainer = RTrainer(mymodel, train_data, test_data)
            if model_type == 'rec':
                result_prediction = mytrainer.train(num_epochs=70, learning_rate=1e-3, batch_size=16, 
                early_stopping_epochs=10, verbose=True, save_rand=True, val_steps=15, retrain_same_init=True, old_exp_dir = old_exp_id, window=window,latency=latency)
            else:
                result_prediction = mytrainer.train(num_epochs=60, batch_size = 16, learning_rate=lr, verbose=True, save_rand=True, val_steps=15, 
                                retrain_same_init=True, old_exp_dir = old_exp_id,window=window,latency=latency)
            outt = time.time()
            print(f'Successfully trained model {i+1} / {args.end_id - args.start_id} in {(outt-intime)/60} minutes.')

            best_models_arch, key_trained = load_df(model_type, arch_type, monkey)

            best_models_arch.at[i,key_trained] = True
            save_df(best_models_arch, model_type, arch_type, monkey)

            # Save result prediction
            result_file = 'active_{}_{}.json'.format(monkey, session)
            with open(os.path.join(path_to_model_results_search, result_file), 'w') as outfile:
                json.dump(result_prediction, outfile, cls=NumpyArrayEncoder)
                print('Saved in :', os.path.join(path_to_model_results_search, result_file))

    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training Convolutional Nets for PCR.')
    # parser.add_argument('--old_models', type=str, help='Name of old conv models',default='ALL_spatial_temporal')
    parser.add_argument('--monkey', type=str, help='Name of the monkey',default='Snap')  #Snap  Butter Chips
    parser.add_argument('--session', type=int, help='Monkey session',default=20190829)  #20190829  20180326 20170913
    parser.add_argument('--lr', type=float, help='Learning rate',default=0.001)  #0.005
    parser.add_argument('--type', type=str, help='Type of model',default='conv')
    parser.add_argument('--arch_type', type=str, help='Architecture of specific model',default='spatial_temporal')
    parser.add_argument('--exp_id', type=int, help='Experiment ID',default=13500)
    parser.add_argument('--start_id', type=int, help='Id of net to start',default=13) #15
    parser.add_argument('--end_id', type=int, help='Id of net to end',default=14)  #16
    parser.add_argument('--window', type=int, help='Which window param?', default=5)
    parser.add_argument('--latency', type=int, help='Which latency param?', default=0)
    main(parser.parse_args())