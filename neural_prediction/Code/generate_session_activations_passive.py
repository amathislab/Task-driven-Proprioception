'''
Script to generate activations using monkey passive and active kinematic datasets.
Generate for ALL models in an experiment_X.
'''

# Imports
import yaml, os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
import numpy as np
import pickle
import argparse
import sys
from sklearn.decomposition import PCA, IncrementalPCA
from collections import defaultdict

import h5py
sys.path.append('../../code')
sys.path.append('../../neural_prediction')
from nn_models import ConvModel, RecurrentModel, ConvModel_new
from nn_train_utils import *
from nn_rmodels_w_outputs import ConvRModel, RecurrentRModel, ConvRModel_new
from nn_train_rutils_multiple_outputs import *


# Modules
from predict_utils import load_monkey_datasets_align, load_monkey_datasets_align_new
from global_utils import is_CObump_session, get_PCR_exp_type, get_network_family, floatify_keys, check_modelconfig_status
from tensorflow.python import pywrap_tensorflow

sys.path.append('../../code/')
from path_utils import MODELS_DIR, PATH_TO_NEURAL_DATA


def get_tensors_in_checkpoint_file(file_name,all_tensors=True,tensor_name=None):
        varlist=[]
        var_value =[]
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        if all_tensors:
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in sorted(var_to_shape_map):
                varlist.append(key)
                var_value.append(reader.get_tensor(key))
        else:
            varlist.append(tensor_name)
            var_value.append(reader.get_tensor(tensor_name))
        return (varlist, var_value)
    
def build_tensors_in_checkpoint_file(model,loaded_tensors, tvars):
    full_var_list = list()
    # Loop all loaded tensors
    for i, tensor_name in enumerate(loaded_tensors[0]):
        # Extract tensor
        try:
            if (not 'BatchNorm' in tensor_name) and (not 'FC' in tensor_name) and (not 'Classifier' in tensor_name):
                for var in tvars:
                    if var.name == tensor_name+":0":
                        full_var_list.append(var)
        except:
            print('Not found: '+tensor_name)
    return full_var_list

def main(monkey_name, session_date, exp_id, train_iter, active_start='mvt', active_length=0, align=100,
         permut_m=False, permut_t=False, constant_input=False, n_pca=25, path_to_act='..'):
    ''' Generate all PCR (classification and regression) experiment model activations for a monkey dataset.
     Arguments:
     monkey_name (str) -- Name of the monkey
     session_date (int) -- Session date YYYYMMDD
     exp_id (int) -- PCR experiment id.
     '''

    PATH_TO_ACTIVATIONS = path_to_act

    # Check available kinematic files (active and/or passive)
    has_passive = is_CObump_session(session_date)
    print('Does monkey session have active and passive? ->', has_passive)

    PCR_task = get_PCR_exp_type(exp_id)
    print('PCR task type:', PCR_task)

    # LOAD SESSION KINEMATIC DATASETS
    control_dict = {'permut_m':permut_m, 'permut_t':permut_t, 'constant_input':constant_input}
    print('CONTROL DICT INFO', control_dict)
    datasets = load_monkey_datasets_align_new(PATH_TO_NEURAL_DATA,
                                    monkey_name,
                                    session_date,
                                    has_passive,
                                    active_start=active_start,
                                    active_length=active_length,
                                    align=align,
                                    control_dict=control_dict)

    task_datasets = [datasets['passive']]

    # Get list of network models in experiment
    path_to_exp_models = os.path.join(MODELS_DIR, 'experiment_{}'.format(exp_id))
    model_list = os.listdir(path_to_exp_models)

    #Path to experiment activations
    path_to_exp_act = os.path.join(PATH_TO_ACTIVATIONS, 'experiment_{}'.format(exp_id))

    # (FILTER SOME MODELS)
    m_counter = 0
    print('Number of trained networks in experiment_{}: {}'.format(exp_id, len(model_list)))

    ## This model doesn't converge for Chips
    if int(exp_id) in [20516]:
        model_list = [i for i in model_list if i not in ['spatiotemporal_r_1_64_5252']]

    for model_name in model_list:
        m_counter += 1

        print('--- Activations for model:', model_name)
        # Saving folders for activations
        path_to_model_act = os.path.join(path_to_exp_act, model_name)

        # LOAD CONFIG FILE
        if train_iter is not None:
            try:
                path_to_config_file = os.path.join(path_to_exp_models, '{}/config_{}.yaml'.format(model_name, int(train_iter)))
            except FileNotFoundError as err:
                print('Error: config file at training checkpoint {} does not exist!'.format(train_iter), err)
                return
        else:
            path_to_config_file = os.path.join(path_to_exp_models, '{}/config.yaml'.format(model_name))

        try:
            with open(path_to_config_file, 'r') as myfile:
                model_config = yaml.safe_load(myfile)
        except:
            continue

        model_config = floatify_keys(model_config)


        if not os.path.exists(path_to_model_act):
            os.makedirs(path_to_model_act)

        try:
            act_model_files = os.listdir(path_to_model_act)
        except FileNotFoundError as err:
            print('No model folder for {}'.format(model_name), ' - skipping.')
        
        if train_iter is not None:
            act_m_files = [i for i in act_model_files if i.startswith(monkey_name)]
            act_m_files = [i for i in act_m_files if '_ckpt'+str(train_iter) in i ]
        else:
            act_m_files = [i for i in act_model_files if i.startswith(monkey_name)]
            act_m_files = [i for i in act_m_files if not '_ckpt' in i ]

        print(len(act_m_files))

        try:
            if model_config['arch_type'] == 'spatiotemporal':
                n_files = int(float(model_config['nlayers']))
            else:
                n_files = int(float(model_config['nlayers']))*2
        except:
            if model_config['rec_blocktype'] == 'lstm':
                n_files = int(model_config['npplayers']) + 1


        ######## DECOMMENT HERE!! 
        if (len(act_m_files)//2 == n_files) or (len(act_m_files) == n_files):
            print(act_m_files)
            continue

        # RUN FOR EACH BEHAVIORAL MONKEY TASKS
        selected_tasks = ['passive']
        for task_data, task_type in zip(task_datasets, selected_tasks):

            # Get data input dimensions
            nsamples, ninputs, ntime, _ = task_data['muscle_coords'].shape
            if nsamples % 100 == 0:
                batch_size = 100
            elif nsamples % 50 == 0:
                batch_size = 50
            elif nsamples % 10 == 0:
                batch_size = 10
            elif nsamples % 5 == 0:
                batch_size = 5    
            elif nsamples % 3 == 0:
                batch_size = 3 
            elif nsamples % 2 == 0:
                batch_size = 2 
            else:
                batch_size = 1

            num_steps = nsamples // batch_size
            print('Number of batches for {} data :{}'.format(task_type, num_steps))

            # Initialize model with config file
            tf.reset_default_graph()

            regression_task_list = ['regression', 'torque', 'center_out', 'joints_input', \
                                    'regress_joints_pos', 'regress_joints_vel', 'regress_joints_pos_vel', 'regress_joints_pos_vel_acc', \
                                    'regress_ee_vel', 'regress_ee_pos_vel', 'regress_ee_pos_vel_acc', \
                                    'autoencoder', 'autoencoder_lin', \
                                    'regress_ee_elbow_pos', 'regress_ee_elbow_vel','regress_ee_elbow_pos_vel','regress_ee_elbow_pos_vel_acc',  \
                                    'regress_ee_pos_forward', 'regress_ee_pos_vel_forward', 'regress_ee_pos_vel_acc_forward', \
                                    'regress_joint_pos_forward', 'regress_joint_pos_vel_forward', 'regress_joint_pos_vel_acc_forward', \
                                    'regress_muscles_forward', 'regress_muscles_acc_forward']

            # REGRESSION MODELS
            if PCR_task in regression_task_list:
                if 'lstm' not in model_name:
                    if int(float(model_config['nlayers'])) < 5:
                        model = ConvRModel(model_config['experiment_id'], model_config['arch_type'],
                                           int(float(model_config['nlayers'])), model_config['n_skernels'], model_config['n_tkernels'],
                                           int(float(model_config['s_kernelsize'])), int(float(model_config['t_kernelsize'])),
                                           int(float(model_config['s_stride'])),
                                           int(float(model_config['t_stride'])))
                        model.model_path = path_to_model_act
                    elif int(float(model_config['nlayers'])) >= 5:
                        model = ConvRModel_new(model_config['experiment_id'], model_config['arch_type'],
                                           int(float(model_config['nlayers'])), model_config['n_skernels'],
                                           model_config['n_tkernels'],
                                           int(float(model_config['s_kernelsize'])),
                                           int(float(model_config['t_kernelsize'])),
                                           list(model_config['s_stride']),
                                           list(model_config['t_stride']))
                        model.model_path = path_to_model_act
                    else:
                        model = ConvRModel(model_config['experiment_id'], model_config['arch_type'],
                                           int(float(model_config['nlayers'])), model_config['n_skernels'], model_config['n_tkernels'],
                                           int(float(model_config['s_kernelsize'])), int(float(model_config['t_kernelsize'])),
                                           int(float(model_config['s_stride'])),
                                           int(float(model_config['t_stride'])))
                        model.model_path = path_to_model_act

                elif (get_network_family(exp_id) == 'rec') or ('lstm' in model_name):
                    model = RecurrentRModel(model_config['experiment_id'], model_config['rec_blocktype'],
                                            int(model_config['n_recunits']), int(model_config['npplayers']), model_config['nppfilters'],
                                            int(float(model_config['s_kernelsize'])), int(float(model_config['s_stride'])),
                                            int(model_config['seed']))
                    model.model_path = path_to_model_act

            # CLASSIFICATION MODELS
            elif PCR_task in ['classification', 'bt', 'untrained']:
                if 'lstm' not in model_name:
                    if int(float(model_config['nlayers'])) < 5:
                        model = ConvModel(model_config['experiment_id'], int(model_config['nclasses']), model_config['arch_type'],
                                          int(float(model_config['nlayers'])), model_config['n_skernels'], model_config['n_tkernels'],
                                          int(float(model_config['s_kernelsize'])), int(float(model_config['t_kernelsize'])),
                                          int(float(model_config['s_stride'])),
                                          int(float(model_config['t_stride'])))
                        model.model_path = path_to_model_act
                    
                    elif int(float(model_config['nlayers'])) >= 5:
                        model = ConvModel_new(model_config['experiment_id'], int(model_config['nclasses']), model_config['arch_type'],
                                           int(float(model_config['nlayers'])), model_config['n_skernels'],
                                           model_config['n_tkernels'],
                                           int(float(model_config['s_kernelsize'])),
                                           int(float(model_config['t_kernelsize'])),
                                           list(model_config['s_stride']),
                                           list(model_config['t_stride']))
                        model.model_path = path_to_model_act
                    else:
                        model = ConvModel(model_config['experiment_id'], int(model_config['nclasses']), model_config['arch_type'],
                                          int(float(model_config['nlayers'])), model_config['n_skernels'], model_config['n_tkernels'],
                                          int(float(model_config['s_kernelsize'])), int(float(model_config['t_kernelsize'])),
                                          int(float(model_config['s_stride'])),
                                          int(float(model_config['t_stride'])))
                        model.model_path = path_to_model_act


                elif (get_network_family(exp_id) == 'rec') or ('lstm' in model_name):
                    model = RecurrentModel(model_config['experiment_id'], int(model_config['nclasses']), model_config['rec_blocktype'],
                                    int(model_config['n_recunits']), int(model_config['npplayers']), model_config['nppfilters'],
                                    int(float(model_config['s_kernelsize'])), int(float(model_config['s_stride'])),
                                    int(model_config['seed']))

                    model.model_path = path_to_model_act


            ### INIT. GRAPH
            graph = tf.Graph()
            sess = tf.Session(graph=graph)
            with graph.as_default():
                ## Load model
                sess.run(tf.global_variables_initializer())

            ### Get the model checkpoint
            if train_iter is not None:
                try:
                    CHECKPOINT_FILE = os.path.join(os.path.join(path_to_exp_models,model_name), 'model_{}.ckpt'.format(int(train_iter)))
                except FileNotFoundError as err:
                    print('No model checkpoints for train iteration {}'.format(train_iter), err, ' - skipping.')
            else:
                CHECKPOINT_FILE = os.path.join(os.path.join(path_to_exp_models,model_name), 'model.ckpt')

            ### Build the graph
            with graph.as_default():
                X = tf.placeholder(tf.float32, shape=[batch_size, ninputs, ntime, 2], name="X")
                
                if PCR_task in regression_task_list:
                    scores, net = model.predict(X, is_training=False)
                elif PCR_task in ['classification', 'bt', 'untrained']:
                    scores, proba, net = model.predict(X, is_training=False)

                if PCR_task in regression_task_list + ['bt','untrained']:
                    all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
                    varlist = get_tensors_in_checkpoint_file(file_name=CHECKPOINT_FILE,all_tensors=True,tensor_name=None)
                    variables = build_tensors_in_checkpoint_file(model, varlist, all_variables)
                    loader = tf.train.Saver(variables)
                restorer = tf.train.Saver()
                init = tf.global_variables_initializer()

            ### Run the session
            myconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, gpu_options=gpu_options)
            with graph.as_default():
                with tf.Session(config=myconfig) as sess:
                    sess.run(tf.global_variables_initializer())
                    if PCR_task in regression_task_list + ['bt','untrained']:
                        loader.restore(sess, CHECKPOINT_FILE)
                    else:
                        restorer.restore(sess, CHECKPOINT_FILE)

                    # Forward predictions for each batch
                    layer_dict = defaultdict(list)
                    for step in range(num_steps):
                        # Next input batches
                        batch_x = task_data['muscle_coords'][batch_size * step:batch_size * (step + 1)]#*100
                        _ ,layer = sess.run([scores,net], feed_dict={X: batch_x})


                        for idx, key in enumerate(layer.keys()):
                            if key == 'score':
                                continue
                            # Combine all batches
                            layer_dict[key].extend(layer[key])
                        
                    ## Delete layer for memory
                    del layer, batch_x

                    # REFORMAT LAYER ACTIVATIONS
                    print('REDUCING ACTIVATION DIMENSIONALITY...')
                    for l_idx, l_key in enumerate(layer_dict.keys()):

                        #Format 4D: (trials, spatial, time, filters)
                        layer_data = np.asarray(layer_dict[l_key])
                        print('Model', model_name, ' - layer ({}) data shape'.format(l_key), layer_data.shape)

                        #Format 2D: (trials*time, spatial units<*filters>)
                        if l_key == 'recurrent_out': #Format last LSTM layers differently
                            n_trials, n_t_bins = layer_data.shape[0], layer_data.shape[1]
                            N_MAX_FEATURES = layer_data.shape[2]
                        else:
                            n_trials, n_t_bins = layer_data.shape[0], layer_data.shape[2]
                            N_MAX_FEATURES = layer_data.shape[1] * layer_data.shape[3]

                        #Reduce dimensions
                        print('Shape before PCA', layer_data.shape)
                        if N_MAX_FEATURES > n_pca:
                            n_components = n_pca
                        else:
                            n_components = N_MAX_FEATURES - 1
                        pca = IncrementalPCA(n_components=n_components)
                        # batch_size = 5
                        num_samples = layer_data.shape[0]

                        batch_size = 100

                        #First partial fit over batches
                        for i in range(0, num_samples // batch_size):
                            batch = layer_data[i*batch_size:(i+1)*batch_size]
                            if l_key == 'recurrent_out':
                                batch = batch.reshape((batch.shape[0] * n_t_bins, -1))
                            else:
                                batch = batch.swapaxes(1, 2).reshape((batch_size * n_t_bins, -1))
                            pca.partial_fit(batch)
                        batch = layer_data[(i+1)*batch_size:(i+2)*batch_size]
                        if l_key == 'recurrent_out':
                            batch = batch.reshape((batch.shape[0] * n_t_bins, -1))
                        else:
                            batch = batch.swapaxes(1, 2).reshape((batch.shape[0] * n_t_bins, -1))
                        pca.partial_fit(batch)
                        print('-- PCA batch partial fits done')

                        #Then apply transform over batches
                        data_pca = []
                        for i in range(0, num_samples // batch_size ):
                            batch = layer_data[i*batch_size:(i+1)*batch_size]
                            if l_key == 'recurrent_out':
                                batch = batch.reshape((batch_size * n_t_bins, -1))
                            else:
                                batch = batch.swapaxes(1, 2).reshape((batch_size * n_t_bins, -1))
                            batch_pca = pca.transform(batch)
                            data_pca.append(batch_pca.reshape(batch_size, n_t_bins, -1))
                        batch = layer_data[(i+1)*batch_size:(i+2)*batch_size]
                        last_batch_size = batch.shape[0]
                        if l_key == 'recurrent_out':
                            batch = batch.reshape((last_batch_size * n_t_bins, -1))
                        else:
                            batch = batch.swapaxes(1, 2).reshape((last_batch_size * n_t_bins, -1))
                        batch_pca = pca.transform(batch)
                        data_pca.append(batch_pca.reshape(last_batch_size, n_t_bins, -1))
                        print('-- PCA batch transforms done')

                        data_pca = np.vstack(data_pca).reshape(n_trials, n_t_bins, -1)
                        # data_pca = np.asarray(data_pca).reshape(n_trials, n_t_bins, -1)
                        print('Inc. PCA final shape', data_pca.shape)

                        # SAVE EACH LAYER SEPARATELY
                        active_start_suff = '_'+str(active_start)
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
                        if train_iter is not None:
                            file_name_suffix_active += '_ckpt'+str(train_iter)
                            file_name_suffix_passive += '_ckpt' + str(train_iter)


                        if task_type == 'active':
                            f_name = '{}_{}_{}_l{}{}.h5'.format(monkey_name, session_date, task_type, l_idx, file_name_suffix_active)
                        elif task_type == 'passive':
                            f_name = '{}_{}_{}_l{}{}.h5'.format(monkey_name, session_date, task_type, l_idx, file_name_suffix_passive)

                        file_path = os.path.join(path_to_model_act, f_name)
                        if not os.path.exists(path_to_model_act):
                            os.makedirs(path_to_model_act)

                        ### DECOMMENT HERE
                        shape_pca = data_pca.shape
                        with h5py.File(file_path, 'w') as file:
                            file.create_dataset('layer_pca', data=data_pca, chunks=(1,shape_pca[1],shape_pca[2]), maxshape=(None,shape_pca[1],shape_pca[2]),compression="gzip", dtype='float32')
                        
                        print('L{} {} saved: shape {}.'.format(l_idx, task_type, data_pca.shape))


            print('Activations for {} trials done.'.format(task_type))

        print('--- Activations for {} generated ({}/{} all exp models).'.format(model_name, m_counter, len(model_list)))

    print('------ Activations generated for {} networks in experiment_{}!'.format(len(model_list), exp_id))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate activations with pre-trained PCR experiment networks for a monkey session.')

    parser.add_argument('--monkey', type=str, help='Which monkey?', required=False, default = 'S1Lando')  #Snap
    parser.add_argument('--session', type=int, help='Which session data?', required=False, default = 20170917)  #20190829
    parser.add_argument('--exp_id', type=int, help='Which experiment id?', required=False, default = 4015)
    parser.add_argument('--n_pca', type=int, help='How many PCs?', required=False, default = 75)
    parser.add_argument('--path_to_act', type=str, help='Path where to save activations?', required=False, default = '..')
    parser.add_argument('--train_iter', type=int, default=None, help='Which training checkpoint index?', required=False)
    parser.add_argument('--active_start', type=str, default='mvt', help='Which active start index?', required=False)
    parser.add_argument('--active_length', type=int, default=0, help='Length after movement onset (1bin=10ms)? [None if passive].', required=False)
    parser.add_argument('--align', type=int, default=100, help='Index of trial onset alignment.', required=False)
    parser.add_argument('--permut_m', action='store_false', help='Permut muscles control?', required=False, default = False)
    parser.add_argument('--permut_t', action='store_false', help='Permut time control?', required=False, default = False)
    parser.add_argument('--constant_input', action='store_false', help='Constant input control?', required=False, default = False)

    args = parser.parse_args()

    main(args.monkey, args.session, args.exp_id, args.train_iter,
         args.active_start, args.active_length, args.align,
         args.permut_m, args.permut_t, args.constant_input, args.n_pca, args.path_to_act)