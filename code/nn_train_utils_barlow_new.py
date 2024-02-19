"""

"""

import os
import copy
import h5py
import yaml
import random

import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from barlow_augmentation_utils import *

from path_utils import MODELS_DIR
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

gpu_options = tf.GPUOptions(allow_growth=True) #per_process_gpu_memory_fraction=0.33, allow_growth=True) #per_process_gpu_memory_fraction=0.9)

class Dataset():
    """Defines a dataset object with simple routines to generate batches."""

    def __init__(self, path_to_data=None, path_to_val=None, data=None, path_to_ind = None, augm_type = None, dataset_type='train', key='spindle_firing', fraction=None):
        """Set up the `Dataset` object.

        Arguments
        ---------
        path_to_data : str, absolute location of the dataset file.
        dataset_type : {'train', 'test'} str, type of data that will be used along with the model.
        key : {'endeffector_coords', 'joint_coords', 'muscle_coords', 'spindle_firing'} str

        """
        self.path_to_data = path_to_data
        self.path_to_val = path_to_val
        self.dataset_type = dataset_type
        self.key = key
        self.train_data = self.train_labels = None
        self.val_data = self.val_labels = None
        self.test_data = self.test_data = None
        self.path_to_ind = path_to_ind
        self.ind_list = ind_list = None
        self.augm_type = augm_type
        self.make_data(data)

        print('The following augmentations are applied:', self.augm_type)
        

        # For when I want to use only a fraction of the dataset to train!
        if fraction is not None:
            random_idx = np.random.permutation(self.train_data.shape[0])
            subset_num = int(fraction * random_idx.size)
            self.train_data = self.train_data[random_idx[:subset_num]]
            self.train_labels = self.train_labels[random_idx[:subset_num]]

    def make_data(self, mydata):
        """Load train/val or test splits into the `Dataset` instance.

        Returns
        -------
        if dataset_type == 'train' : loads train and val splits.
        if dataset_type == 'test' : loads the test split.

        """
        # Load and shuffle dataset randomly before splitting
        if self.path_to_data is not None:
            datafile = h5py.File(self.path_to_data, 'r')
            if self.dataset_type == 'train':
                self.train_data = datafile[self.key]
                self.train_labels = datafile['label']
                self.train_data_mean = datafile['train_data_mean']
                datafile_val = h5py.File(self.path_to_val, 'r')
                self.val_data = datafile_val[self.key]
                self.val_labels = datafile_val['label']
            elif self.dataset_type == 'test':
                self.test_data = datafile[self.key]
                self.test_labels = datafile['label']
        else: 
            data = mydata['data']
            labels = mydata['labels'] - 1

        if self.path_to_ind is not None:
            self.ind_list = h5py.File(self.path_to_ind, 'r')

    def apply_augmentation(self, mybatch_data, data_type):
        """Apply augmentation to batch data for Barlow Twins:
        Inputs
        ------
        mybatch_data: batch data
        data_type: train/val/test to select the corresponding data

        Returns
        -------
        mybatch_data_1, mybatch_data_2: two augmentated batches

        """
        mybatch_data_1 = mybatch_data.copy()
        mybatch_data_2 = mybatch_data.copy()
        for augm_tmp in self.augm_type:
                if (augm_tmp == 'min') or (augm_tmp == 'max'):

                    ######## GET CLOSE TRAJECTORY
                    mybatch_ind = self.ind_list[data_type][augm_tmp][self.shuffle_idx[step]:self.shuffle_idx[step]+batch_size]

                    # mybatch_data_1 = mybatch_data
                    mybatch_data_2 = []
                    for ind_list_tmp in mybatch_ind:
                        sel_ind = random.sample(list(ind_list_tmp),1)[0]
                        if data_type == 'train':
                            mybatch_data_2.append(self.train_data[sel_ind])
                        elif data_type == 'val':
                            mybatch_data_2.append(self.val_data[sel_ind])
                        elif data_type == 'test':
                            mybatch_data_2.append(self.test_data[sel_ind])
                    mybatch_data_2 = np.stack(mybatch_data_2)

                if (augm_tmp == 'noise'):
                    ######## ADDING SPARSE NOISE
                    mybatch_data_1 = np.stack(list(map(random_noise, list(mybatch_data_1)))) #tf.map_fn(self.random_mask, mybatch_data)
                    mybatch_data_2 = np.stack(list(map(random_noise, list(mybatch_data_2)))) 

                if (augm_tmp == 'time'):
                    ####### MASKING TIME
                    mybatch_data_1 = np.stack(list(map(random_mask_time, list(mybatch_data_1)))) #tf.map_fn(self.random_mask, mybatch_data)
                    mybatch_data_2 = np.stack(list(map(random_mask_time, list(mybatch_data_2)))) 

                if (augm_tmp == 'muscle'):
                    ######## MASKING MUSCLES
                    mybatch_data_1 = np.stack(list(map(lambda x: random_mask(x), list(mybatch_data_1)))) #tf.map_fn(self.random_mask, mybatch_data)
                    mybatch_data_2 = np.stack(list(map(lambda x: random_mask(x), list(mybatch_data_2)))) # tf.map_fn(self.random_mask, mybatch_data)
        return mybatch_data_1, mybatch_data_2

    def next_trainbatch(self, batch_size, step=0, normalize = False):
        """Returns a new batch of training data.

        Arguments
        ---------
        batch_size : int, size of training batch.
        step : int, step index in the epoch.

        Returns
        -------
        2-tuple of batch of training data and correspondig labels.

        """
        if step == 0:
            steps_per_epoch = self.train_data.shape[0] // batch_size
            total_len = batch_size*steps_per_epoch
            poss_position = np.arange(0,total_len,batch_size)
            self.shuffle_idx = np.random.permutation(poss_position)
        mybatch_data = self.train_data[self.shuffle_idx[step]:self.shuffle_idx[step]+batch_size].astype('float32') 
        mybatch_data_1, mybatch_data_2 = self.apply_augmentation(mybatch_data, data_type = 'train')

        return (mybatch_data_1, mybatch_data_2) #(mybatch_data,mybatch_data) #mybatch_data_1, mybatch_data_2)

    def next_valbatch(self, batch_size, type='val', step=0, normalize = False):
        """Returns a new batch of validation or test data.

        Arguments
        ---------
        type : {'val', 'test'} str, type of data to return.

        """
        if type == 'val':
            mybatch_data = self.val_data[batch_size*step:batch_size*(step+1)].astype('float32') 
            mybatch_data_1, mybatch_data_2 = self.apply_augmentation(mybatch_data, data_type = 'val')

        elif type == 'test':
            mybatch_data = self.test_data[batch_size*step:batch_size*(step+1)].astype('float32') 
            mybatch_data_1, mybatch_data_2 = self.apply_augmentation(mybatch_data, data_type = 'test')

        return (mybatch_data_1, mybatch_data_2)


class Trainer:
    """Trains a `Model` object with the given `Dataset` object."""

    def __init__(self, model=None, dataset=None, test_dataset=None, global_step=None):
        """Set up the `Trainer`.

        Arguments
        ---------
        model : an instance of `ConvModel`, `AffineModel` or `RecurrentModel` to be trained.
        dataset : an instance of `Dataset`, containing the train/val data splits.

        """
        self.model = model
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.log_dir = model.model_path
        self.global_step = 0 if global_step is None else global_step
        self.session = None
        self.graph = None
        self.best_loss = 1e10
        self.validation_accuracy = 0

    def get_tensors_in_checkpoint_file(self, file_name,all_tensors=True,tensor_name=None):
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
    
    def build_tensors_in_checkpoint_file(self, loaded_tensors):
        full_var_list = list()
        # Loop all loaded tensors
        for i, tensor_name in enumerate(loaded_tensors[0]):
            # Extract tensor
            if not 'Classifier' in tensor_name:
                try:
                    tensor_aux = self.graph.get_tensor_by_name(tensor_name+":0")
                    full_var_list.append(tensor_aux)
                except:
                    print('Not found: '+tensor_name)
        return full_var_list

    def build_graph(self, **kwargs):
        """Build training graph using the `Model`s predict function and setting up an optimizer."""
        
        _, ninputs, ntime, _ = self.dataset.train_data.shape
        with tf.Graph().as_default() as self.graph:
            tf.set_random_seed(self.model.seed)
            # Placeholders
            self.learning_rate = tf.placeholder(tf.float32)
            self.X1 = tf.placeholder(tf.float32, shape=[self.batch_size, ninputs, ntime, 2], name="X1")
            self.X2 = tf.placeholder(tf.float32, shape=[self.batch_size, ninputs, ntime, 2], name="X2")

            # Set up optimizer, compute and apply gradients
            z_a, _, _ = self.model.predict(self.X1, is_training=True)
            z_b, _, _ = self.model.predict(self.X2, is_training=True)
            self.barlow_twins_loss = tf.reduce_mean(compute_loss(z_a, z_b, self.model.lambd), name="loss")
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.barlow_twins_loss)

            # Calculate metrics
            z_a, _, _ = self.model.predict(self.X1, is_training=False)
            z_b, _, _ = self.model.predict(self.X2, is_training=False)
            self.val_loss = tf.reduce_mean(compute_loss(z_a, z_b, self.model.lambd), name="val_loss")

            tf.summary.scalar('Train_Loss', self.barlow_twins_loss)
            self.train_summary_op = tf.summary.merge_all()
            
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

            if not len(kwargs) == 0:
                varlist = self.get_tensors_in_checkpoint_file(file_name=kwargs['log_dir'],all_tensors=True,tensor_name=None)
                variables = self.build_tensors_in_checkpoint_file(varlist)
                
                self.loader = tf.train.Saver(variables)

    def load(self):
        self.saver.restore(self.session, os.path.join(self.log_dir, 'model.ckpt'))
        
    def save(self):
        self.saver.save(self.session, os.path.join(self.log_dir, 'model.ckpt'))

    def save_step(self,step):
        self.saver.save(self.session, os.path.join(self.log_dir, 'model_' + str(step) + '.ckpt'))
    
    def load_step(self, step):
        self.saver.restore(self.session, os.path.join(self.log_dir, 'model_' + str(step) + '.ckpt'))

    def normalization(self):

        def repeat_batch(vector, batch_size, t_size):
            vector.resize((1,vector.shape[0],1))
            vector = np.repeat(vector, batch_size, axis = 0)
            vector = np.repeat(vector, t_size, axis = 2)
            return vector
            
        t_size = self.dataset.train_data.shape[2]
        self.minn_all_muscle = repeat_batch(self.minn_all_muscle, self.batch_size, t_size)
        self.maxx_all_muscle = repeat_batch(self.maxx_all_muscle, self.batch_size, t_size)
        self.minn_all_vel = repeat_batch(self.minn_all_vel, self.batch_size, t_size)
        self.maxx_all_vel = repeat_batch(self.maxx_all_vel, self.batch_size, t_size)

        divider_mus = (self.maxx_all_muscle - self.minn_all_muscle)
        divider_mus[divider_mus == 0] = 1

        divider_vel = (self.maxx_all_vel - self.minn_all_vel)
        divider_vel[divider_vel == 0] = 1
        return divider_mus, divider_vel

    def make_model_name(self):
        if (type(self.model).__name__ == 'BarlowTwinsModel'):
            # Make model name
            if self.model.arch_type == 'spatial_temporal':
                kernels = ('-'.join(str(i) for i in self.model.n_skernels)) + '_' + ('-'.join(str(i) for i in self.model.n_tkernels))
            elif self.model.arch_type == 'temporal_spatial':
                kernels = ('-'.join(str(i) for i in self.model.n_tkernels)) + '_' + ('-'.join(str(i) for i in self.model.n_skernels))
            else:
                kernels = ('-'.join(str(i) for i in self.model.n_skernels))

            parts_name = [self.model.arch_type, str(self.model.nlayers), kernels,
                        ''.join(str(i) for i in [self.model.s_kernelsize, self.model.s_stride, self.model.t_kernelsize, self.model.t_stride])]

            # Create model directory
            name = '_'.join(parts_name)
        elif (type(self.model).__name__ == 'BarlowTwinsModel_new'):

            max_tstride = self.model.t_stride.count(2)**2
            max_sstride = self.model.s_stride.count(2)**2
            # Make model name
            if self.model.arch_type == 'spatial_temporal':
                kernels = ('-'.join(str(i) for i in self.model.n_skernels)) + '_' + ('-'.join(str(i) for i in self.model.n_tkernels))
            elif self.model.arch_type == 'temporal_spatial':
                kernels = ('-'.join(str(i) for i in self.model.n_tkernels)) + '_' + ('-'.join(str(i) for i in self.model.n_skernels))
            else:
                kernels = ('-'.join(str(i) for i in self.model.n_skernels))

            parts_name = [self.model.arch_type, str(self.model.nlayers), kernels,
                        ''.join(str(i) for i in [self.model.s_kernelsize, max_sstride, self.model.t_kernelsize, max_tstride])]

            # Create model directory
            name = '_'.join(parts_name)
        elif (type(self.model).__name__ == 'BarlowTwinsModel_rec'):
            # Make model name
            units = ('-'.join(str(i) for i in self.model.nppfilters))
            parts_name = [self.model.rec_blocktype, str(self.model.npplayers), units, str(self.model.n_recunits)]

            # Create model directory
            name = '_'.join(parts_name)
            if self.model.seed is not None: name += '_' + str(self.model.seed)

        elif (type(self.model).__name__ == 'BarlowTwinsModel_rec_new'):
            max_sstride = self.s_stride.count(2)**2
        
            # Make model name
            units = ('-'.join(str(i) for i in nppfilters))
            parts_name = [rec_blocktype, str(n_reclayers), str(npplayers), units, str(n_recunits),
                        ''.join(str(i) for i in [s_kernelsize, max_sstride])]

            # Create model directory
            name = '_'.join(parts_name)
            if self.model.seed is not None: name += '_' + str(self.model.seed)
        return name

    def train(self,
            num_epochs=10,
            learning_rate=0.005,
            batch_size=256,
            val_steps=200,
            early_stopping_epochs=1,
            retrain=False,
            retrain_same_init=False,
            old_exp_dir = None,
            normalize=False,
            verbose=True, 
            save_rand=False):
        """Train the `Model` object.

        Arguments
        ---------
        num_epochs : int, number of epochs to train for.
        learning_rate : float, learning rate for Adam Optimizer.
        batch_size : int, size of batch to train on.
        val_steps : int, number of batches after which to perform validation.
        early_stopping_steps : int, number of steps for early stopping criterion.
        retrain : bool, train already existing model vs not.
        normalize : bool, whether to normalize training data or not.
        verbose : bool, print progress on screen.

        """
        steps_per_epoch = self.dataset.train_data.shape[0] // batch_size
        max_iter = num_epochs * steps_per_epoch
        early_stopping_steps = early_stopping_epochs * steps_per_epoch
        self.batch_size = batch_size
        self.normalize = normalize

        if self.normalize:
            # self.divider_mus, self.divider_vel = self.normalization()
            self.train_data_mean = float(self.dataset.train_data_mean)
            self.train_data_std = 0 #float(np.std(self.dataset.train_data))
        else:
            self.train_data_mean = self.train_data_std = 0
        train_params = {'train_mean': self.train_data_mean,
                        'train_std': self.train_data_std}
        val_params = {'validation_loss': 1e10}
        test_params = {'test_loss': 0}

        if retrain_same_init:
            old_exp_dir = os.path.join(MODELS_DIR,'experiment_' + str(old_exp_dir))
            name = self.make_model_name()
            log_dir = os.path.join(old_exp_dir, name)
            log_dir = os.path.join(log_dir, 'model_0.ckpt')
            self.build_graph(log_dir = log_dir)
        else:
            self.build_graph()
        self.session = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options))

        self.session.run(self.init)
        if retrain:
            self.load()
        
        if retrain_same_init:
            self.loader.restore(self.session, log_dir)

        if save_rand:
            self.save_step(0)
            self.model.is_training = False
            make_config_file(self.model, train_params, val_params, test_params, step = self.global_step) #, save_rand)

        # Create summaries
        self.train_summary = tf.summary.FileWriter(
            os.path.join(self.model.model_path, 'train'), graph=self.graph, flush_secs=30)
        self.val_summary = tf.summary.FileWriter(os.path.join(self.model.model_path, 'val'))

        # Define checkpoints
        try:
            if self.model.rec_blocktype == 'lstm':
                if batch_size == 512:
                    check1, check2, check3 = 1000, 3000, 6000
                else:
                    check1, check2, check3 = 2000, 6000, 12000
        except:
            if (self.model.arch_type == 'spatial_temporal') or (self.model.arch_type == 'temporal_spatial') or (self.model.arch_type == 'spatiotemporal'):
                if batch_size == 512:
                    check1, check2, check3 = 500, 1500, 3000
                else:
                    check1, check2, check3 = 1000, 3000, 6000

        not_improved = 0
        end_training = 0
        val_params = {}

        for self.global_step in range(max_iter):
            
            # Training step
            batch_X, batch_y = self.dataset.next_trainbatch(
                batch_size, self.global_step % steps_per_epoch)

            feed_dict = {self.X1: batch_X - self.train_data_mean,
                        self.X2: batch_y, 
                        self.learning_rate: learning_rate}
            _, train_loss = self.session.run([self.train_op, self.barlow_twins_loss], feed_dict)

            # Validate/save periodically
            if self.global_step % val_steps == 0:
                # Summarize, print progress
                loss_val = self.save_summary(feed_dict)
                if verbose:
                    print('Step : %4d, Train loss : %.2f' % (self.global_step, train_loss))
                    print('Step : %4d, Validation loss : %.2f' % (self.global_step, loss_val))
                    print('best_loss:', self.best_loss, 'loss:', loss_val)

                if loss_val < self.best_loss:
                    self.best_loss = loss_val
                    self.save()
                    val_params = {
                        'validation_loss': float(self.best_loss)}
                    not_improved = 0
                else:
                    not_improved += 1

                if not_improved >= early_stopping_steps/2:
                    learning_rate /= 4
                    print(learning_rate)
                    not_improved = 0
                    end_training += 1
                    self.load()

                if end_training == 2:
                    if self.global_step < 20*steps_per_epoch:
                        end_training = 1
                        not_improved = 0
                    else:
                        break

            if (self.global_step == check1) or (self.global_step == check2) or (self.global_step == check3):
                self.save_step(self.global_step)
                make_config_file(self.model, train_params, val_params, test_params, step = self.global_step)

        self.model.is_training = False

        ### Test the network
        test_loss = self.test_model()
        test_params = {'test_loss': float(test_loss)}

        make_config_file(self.model, train_params, val_params, test_params) #, False)  #train_params
        self.session.close()

    def save_summary(self, feed_dict):
        """Create and save summaries for training and validation."""
        train_summary = self.session.run(self.train_summary_op, feed_dict)
        self.train_summary.add_summary(train_summary, self.global_step)
        validation_loss = self.eval()
        validation_summary = tf.Summary()
        validation_summary.value.add(tag='Validation_Loss', simple_value=validation_loss)
        self.val_summary.add_summary(validation_summary, self.global_step)
        
        return validation_loss

    def eval(self):
        """Evaluate validation performance.
        
        Returns
        -------
        validation_loss : float, loss on the entire validation data
        validation_accuracy : float, accuracy on the validation data
        
        """
        num_iter = self.dataset.val_data.shape[0] // self.batch_size
        acc_val = np.zeros(num_iter)
        loss_val = np.zeros(num_iter)
        loss_val1 = np.zeros(self.batch_size)
        for i in range(num_iter):
            batch_X, batch_y = self.dataset.next_valbatch(self.batch_size, step=i)

            feed_dict = {self.X1: batch_X - self.train_data_mean, 
                        self.X2: batch_y}
            loss_val1 = self.session.run([self.val_loss], feed_dict)
            loss_val = np.array(loss_val1).mean()
        return loss_val.mean() #, acc_val.mean()
    
    def test_model(self):
        """Evaluate test performance.
        
        Returns
        -------
        test_accuracy : float, accuracy on the test data
        
        """
        num_iter = self.test_dataset.test_data.shape[0] // self.batch_size
        self.load()
        # acc_test = np.zeros(num_iter)
        acc_test = []
        for i in range(num_iter):
            batch_X, batch_y = self.test_dataset.next_valbatch(self.batch_size, 'test', step=i)

            feed_dict = {self.X1: batch_X - self.train_data_mean, 
                        self.X2: batch_y}
            acc = self.session.run([self.val_loss], feed_dict)
            acc_test.append(acc)
        return np.mean(acc_test)


def evaluate_model(model, dataset, batch_size=200):
    """Evaluation routine for trained models.

    Arguments
    ---------
    model : the `Conv`, `Affine` or `Recurrent` model to be evaluated. The test data is 
        assumed to be defined within the model.dataset object.
    dataset : the `Dataset` object on which the model is to be evaluated.

    Returns
    -------
    accuracy : float, Classification accuracy of the model on the given dataset.

    """

    # Data handling
    nsamples, ninputs, ntime, _ = dataset.test_data.shape
    num_steps = nsamples // batch_size

    # Retrieve training mean, if data was normalized
    path_to_config_file = os.path.join(model.model_path, 'config.yaml')
    with open(path_to_config_file, 'r') as myfile:
        model_config = yaml.load(myfile)
    train_mean = model_config['train_mean']

    mygraph = tf.Graph()
    with mygraph.as_default():
        X1 = tf.placeholder(tf.float32, shape=[batch_size, ninputs, ntime, 2], name="X1")
        X2 = tf.placeholder(tf.float32, shape=[batch_size, ninputs, ntime, 2], name="X2")

        # Set up optimizer, compute and apply gradients
        z_a, _, _ = model.predict(X1, is_training=False)
        z_b, _, _ = model.predict(X2, is_training=False)
        barlow_twins_loss = tf.reduce_mean(compute_loss(z_a, z_b, model.lambd), name="loss")

        # Test the `model`!
        restorer = tf.train.Saver()
        myconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, gpu_options = gpu_options)
        with tf.Session(config=myconfig) as sess:
            ckpt_filepath = os.path.join(model.model_path, 'model.ckpt')
            restorer.restore(sess, ckpt_filepath)

            test_loss = []
            for step in range(num_steps):         
                batch_x, batch_y = dataset.next_valbatch(batch_size, 'test', step)

                acc = sess.run([barlow_twins_loss], feed_dict={X1: batch_x - train_mean,
                                                    X2: batch_y})
                test_loss.append(acc)

    return np.mean(test_loss)


# Auxiliary Functions

def train_val_split(data, labels):
    num_train = int(0.9*data.shape[0])
    train_data, train_labels = data[:num_train], labels[:num_train]
    val_data, val_labels = data[num_train:], labels[num_train:]

    return (train_data, train_labels, val_data, val_labels)


def make_config_file(model, train_params, val_params, test_params, **kwargs): #, rand_flag = False): #, val_params):
    """Make a configuration file for the given model, created after training.

    Given a `ConvModel`, `AffineModel` or `RecurrentModel` instance, generates a 
    yaml file to save the configuration of the model.

    """
    mydict = copy.copy(model.__dict__)
    # Convert to python native types for better readability
    for (key, value) in mydict.items():
        if isinstance(value, np.generic):
            mydict[key] = float(value)
        elif isinstance(value, list) or isinstance(value, np.ndarray):
            mydict[key] = [int(item) for item in value]

    # Save yaml file in the model's path
    if len(kwargs) == 0:
        path_to_yaml_file = os.path.join(model.model_path, 'config.yaml')
    else:
        path_to_yaml_file = os.path.join(model.model_path, 'config_' + str(kwargs['step']) + '.yaml')

    with open(path_to_yaml_file, 'w') as myfile:
        yaml.dump(mydict, myfile, default_flow_style=False)
        yaml.dump(train_params, myfile, default_flow_style=False)
        yaml.dump(val_params, myfile, default_flow_style=False)
        yaml.dump(test_params, myfile, default_flow_style=False)

    return
