import os
import random
import argparse
import time

import pandas
import pickle

import sys
sys.path.append('../code')
sys.path.append('./code')
from nn_rmodels_torque import ConvRModel, RecurrentRModel, ConvRModel_new, RecurrentRModel_new
from nn_train_rutils_torque import *

from path_utils import PATH_TO_DATA_RL, PATH_TO_OLD

def load_df(model_type, arch_type):
    """ Function to load dataframe based on model type and architecture type.
    Outputs:
    - all_conv_models: dataframe containing all the networks to train
    - key_trained: key to set when train is finished
    """
    if model_type == 'conv_new':
        all_conv_models = pickle.load(open(PATH_TO_OLD + '/torque/ALL_' + arch_type + '_newhyp_seed.p', 'rb'))
    elif model_type == 'conv':
        all_conv_models = pickle.load(open(PATH_TO_OLD + '/torque/ALL_' + arch_type + '_seed.p', 'rb'))
    elif model_type == 'rec':
        all_conv_models = pickle.load(open(PATH_TO_OLD + '/torque/ALL_' + arch_type + '_seed.p', 'rb'))
    elif model_type == 'rec_new':
        all_conv_models = pickle.load(open(PATH_TO_OLD + '/torque/ALL_' + arch_type + '_newhyp_seed.p', 'rb'))

    if (model_type == 'conv_new') or (model_type == 'conv'):
        best_models_arch = all_conv_models[all_conv_models['arch_type'] == arch_type] #.nlargest(1, 'test_accuracy')
        key_trained = 'is_trained'
    else:
        best_models_arch = all_conv_models[all_conv_models['rec_blocktype'] == arch_type]
        key_trained = 'is_training'
    return best_models_arch, key_trained

def save_df(best_models_arch, model_type, arch_type):
    """ Function to save the updated dataframe.
    """
    if model_type == 'conv_new':
        best_models_arch.to_pickle(PATH_TO_OLD + '/torque/ALL_' + arch_type + '_newhyp_seed.p')
    elif model_type == 'conv':
        best_models_arch.to_pickle(PATH_TO_OLD + '/torque/ALL_' + arch_type + '_seed.p')
    elif model_type == 'rec':
        best_models_arch.to_pickle(PATH_TO_OLD + '/torque/ALL_' + arch_type + '_seed.p')
    elif model_type == 'rec_new':
        best_models_arch.to_pickle(PATH_TO_OLD + '/torque/ALL_' + arch_type + '_newhyp_seed.p')
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

    train_data_path = os.path.join(PATH_TO_DATA_RL, 'dataset_train_rl_torque.hdf5')
    val_data_path = os.path.join(PATH_TO_DATA_RL, 'dataset_val_rl_torque.hdf5')
    train_data = Dataset(train_data_path, val_data_path, 'train', key='spindle_info', target_key='torque_coords', reach_key = 'target_coords')

    test_data_path = os.path.join(PATH_TO_DATA_RL, 'dataset_test_rl_torque.hdf5')
    test_data = Dataset(test_data_path, dataset_type='test', key='spindle_info', target_key='torque_coords', reach_key = 'target_coords')
    
    n_outputs = train_data.train_targets.shape[1]
    
    start_id = args.start_id
    end_id = args.end_id
    model_type = args.type
    arch_type = args.arch_type
    n_layersfc = args.n_layersfc

    best_models_arch, key_trained = load_df(model_type, arch_type)

    old_exp_id = sel_exp_id(model_type)
    
    for i in range(start_id, end_id):
        print('---------------------------------')
        print('Training model: ', i)
        print('---------------------------------')

        if not best_models_arch.iloc[i][key_trained]:
            latents = best_models_arch.iloc[i]

            if model_type == 'conv':
                # Create model
                mymodel = ConvRModel(
                    experiment_id=exp_id, #i
                    arch_type=arch_type,
                    nlayers=latents['nlayers'],
                    n_skernels=latents['n_skernels'],
                    n_tkernels=latents['n_tkernels'],
                    s_kernelsize=latents['s_kernelsize'],
                    t_kernelsize=latents['t_kernelsize'],
                    s_stride=latents['s_stride'],
                    t_stride=latents['t_stride'],
                    n_outputs= n_outputs,
                    nlayers_fc=n_layersfc,
                    nunits= [args.proj_size for _ in range(n_layersfc)],
                    projector_rl=True) #,

            elif model_type == 'conv_new':
                # Create model
                mymodel = ConvRModel_new(
                    experiment_id=exp_id, #i
                    arch_type=arch_type,
                    nlayers=latents['nlayers'],
                    n_skernels=latents['n_skernels'],
                    n_tkernels=latents['n_tkernels'],
                    s_kernelsize=latents['s_kernelsize'],
                    t_kernelsize=latents['t_kernelsize'],
                    s_stride=latents['s_stride'],
                    t_stride=latents['t_stride'],
                    n_outputs= n_outputs,
                    nlayers_fc=n_layersfc,
                    nunits= [args.proj_size for _ in range(n_layersfc)],
                    projector_rl=True) #,

            elif model_type == 'rec':
                mymodel = RecurrentRModel(
                    experiment_id=args.exp_id,
                    rec_blocktype=arch_type,
                    n_recunits=latents['n_recunits'],
                    npplayers=latents['npplayers'],
                    nppfilters=latents['nppfilters'],
                    s_kernelsize=latents['s_kernelsize'],
                    s_stride=latents['s_stride'],
                    n_outputs= n_outputs,
                    nlayers_fc=n_layersfc,
                    nunits= [args.proj_size for _ in range(n_layersfc)],
                    seed=latents['seed'])

            elif model_type == 'rec_new':
                mymodel = RecurrentRModel_new(
                    experiment_id=args.exp_id,
                    rec_blocktype=arch_type,
                    n_reclayers=latents['n_reclayers'],
                    n_recunits=latents['n_recunits'],
                    npplayers=latents['npplayers'],
                    nppfilters=latents['nppfilters'],
                    s_kernelsize=latents['s_kernelsize'],
                    s_stride=latents['s_stride'],
                    seed=latents['seed'],
                    n_outputs= n_outputs,
                    nlayers_fc=n_layersfc,
                    nunits= [args.proj_size for _ in range(n_layersfc)],
                    projector_rl=True)

            print(mymodel.__dict__)

            intime = time.time()
            # Create trainer and train!
            mytrainer = Trainer(mymodel, train_data, test_data)
            if model_type == 'rec':
                mytrainer.train(num_epochs=70, learning_rate=1e-3, batch_size=512, 
                early_stopping_epochs=1, verbose=True, save_rand=True, retrain_same_init=True, old_exp_dir = old_exp_id)
            else:
                mytrainer.train(num_epochs=50, batch_size = 512, verbose=True, save_rand=True, retrain_same_init=True, old_exp_dir = old_exp_id)
            outt = time.time()
            print(f'Successfully trained model {i+1} / {args.end_id - args.start_id} in {(outt-intime)/60} minutes.')

            best_models_arch, key_trained = load_df(model_type, arch_type)

            best_models_arch.at[i,key_trained] = True
            save_df(best_models_arch, model_type, arch_type)

    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training Convolutional Nets for PCR.')
    # parser.add_argument('--old_models', type=str, help='Name of old conv models',default='ALL_spatial_temporal')
    parser.add_argument('--type', type=str, help='Type of model',default='conv')
    parser.add_argument('--arch_type', type=str, help='Architecture of specific model',default='spatial_temporal')
    parser.add_argument('--exp_id', type=int, help='Experiment ID',default=42)
    parser.add_argument('--proj_size', type=int, help='Size of the projector',default=128)
    parser.add_argument('--n_layersfc', type=int, help='Number of fully connected layers',default=1)
    parser.add_argument('--start_id', type=int, help='Id of net to start',default=0)
    parser.add_argument('--end_id', type=int, help='Id of net to end',default=1)
    main(parser.parse_args())