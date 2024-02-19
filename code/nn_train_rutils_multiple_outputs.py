"""

"""

import os
import copy
import h5py
import yaml

import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from kinematics_decoding import set_kin_dimensions

from path_utils import MODELS_DIR

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

gpu_options = tf.GPUOptions(allow_growth=True)

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33) #per_process_gpu_memory_fraction=0.9)

class RDataset():
    """Defines a dataset object for regression, with simple routines to generate batches."""

    def __init__(self, path_to_data=None, path_to_val=None, data=None, dataset_type='train', 
            key='spindle_info', fraction=None, target_key_list=None, n_out_time=400):
        """Set up the `Dataset` object.

        Arguments
        ---------
        path_to_data : str, absolute location of the dataset file.
        data: dict, optionally provide dataset directly as {'data': np.array, 'targets': np.array}
        dataset_type : {'train', 'test'} str, type of data that will be used along with the model.
        key : {'endeffector_coords', 'joint_coords', 'muscle_coords', 'spindle_firing'} str
        target_key: {'endeffector_coords', 'joint_coords'} str, specifies targets to regress

        """
        self.path_to_data = path_to_data
        self.path_to_val = path_to_val
        self.dataset_type = dataset_type
        self.key = key
        self.target_key_list = target_key_list
        self.train_data = self.train_targets = None
        self.val_data = self.val_targets = None
        self.test_data = self.test_targets = None
        self.n_out_time = n_out_time
        self.make_data(data)

        # For when I want to use only a fraction of the dataset to train!
        if fraction is not None:
            random_idx = np.random.permutation(self.train_data.shape[0])
            subset_num = int(fraction * random_idx.size)
            self.train_data = self.train_data[random_idx[:subset_num]]
            self.train_targets = self.train_targets[random_idx[:subset_num]]

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
                self.train_targets_list = [datafile[target_key] for target_key in self.target_key_list]
                self.train_data_mean = datafile['train_data_mean']
                datafile_val = h5py.File(self.path_to_val, 'r')
                self.val_data = datafile_val[self.key]
                self.val_targets_list = [datafile_val[target_key] for target_key in self.target_key_list]

                self.n_outputs = int(np.sum([train_target.shape[1] for train_target in self.train_targets_list]))
                # print(self.n_outputs)
            elif self.dataset_type == 'test':
                self.test_data = datafile[self.key]
                self.test_targets_list = [datafile[target_key] for target_key in self.target_key_list]

        else: 
            data = mydata['data']
            labels = mydata['targets']

    def next_trainbatch(self, batch_size, step=0):
        """Returns a new batch of training data.

        Arguments
        ---------
        batch_size : int, size of training batch.
        step : int, step index in the epoch.

        Returns
        -------
        2-tuple of batch of training data and correspondig targets.

        """
        if step == 0:
            steps_per_epoch = self.train_data.shape[0] // batch_size
            total_len = batch_size*steps_per_epoch
            poss_position = np.arange(0,total_len,batch_size)
            self.shuffle_idx = np.random.permutation(poss_position)
        mybatch_data = self.train_data[self.shuffle_idx[step]:self.shuffle_idx[step]+batch_size].astype('float32') 
        mybatch_targets = [train_targets[self.shuffle_idx[step]:self.shuffle_idx[step]+batch_size] for train_targets in self.train_targets_list]#batch_size*step:batch_size*(step+1)

        mybatch_targets = np.concatenate(mybatch_targets,axis=1)
        mybatch_targets = set_kin_dimensions(mybatch_targets, self.n_out_time)

        return (mybatch_data, mybatch_targets)

    def next_valbatch(self, batch_size, type='val', step=0):
        """Returns a new batch of validation or test data.

        Arguments
        ---------
        type : {'val', 'test'} str, type of data to return.

        """
        if type == 'val':
            mybatch_data = self.val_data[batch_size*step:batch_size*(step+1)]
            mybatch_targets = [val_targets[batch_size*step:batch_size*(step+1)] for val_targets in self.val_targets_list]
        elif type == 'test':
            mybatch_data = self.test_data[batch_size*step:batch_size*(step+1)]
            mybatch_targets = [test_targets[batch_size*step:batch_size*(step+1)] for test_targets in self.test_targets_list]
        
        mybatch_targets = np.concatenate(mybatch_targets,axis=1)
        
        mybatch_targets = set_kin_dimensions(mybatch_targets, self.n_out_time) 

        return (mybatch_data, mybatch_targets)
    
    def set_outtime(self, outtime):
        self.n_out_time = outtime


class RTrainer:
    """Trains a `RModel` object with the given `Dataset` object."""

    def __init__(self, model=None, dataset=None, test_dataset=None, global_step=None):
        """Set up the `Trainer`.

        Arguments
        ---------
        model : an instance of `ConvRModel` or `RecurrentRModel` to be trained.
        dataset : an instance of `Dataset`, containing the train/val data splits.

        """
        self.model = model
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.log_dir = model.model_path
        self.global_step = 0 if global_step == None else global_step
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
        ncoords = self.dataset.n_outputs
        # _, ncoords, _ = self.dataset.train_targets.shape   #ncoords was 3

        if (type(self.model).__name__ == 'ConvRModel'):
            layers = self.model.nlayers
            stride = self.model.t_stride
            nouttime = 400
            for i in range(layers):
                nouttime = int(np.ceil(nouttime/stride))
            self.dataset.set_outtime(nouttime)
            self.test_dataset.set_outtime(nouttime)
        elif (type(self.model).__name__ == 'ConvRModel_new'):
            layers = self.model.nlayers
            stride = self.model.t_stride
            nouttime = 400
            for i in range(layers):
                nouttime = int(np.ceil(nouttime/stride[i]))
            self.dataset.set_outtime(nouttime)
            self.test_dataset.set_outtime(nouttime)
        else:
            nouttime = 400
        
        with tf.Graph().as_default() as self.graph:
            tf.set_random_seed(self.model.seed)
            # Placeholders
            self.learning_rate = tf.placeholder(tf.float32)
            self.X = tf.placeholder(tf.float32, shape=[self.batch_size, ninputs, ntime, 2], name="X")
            self.y = tf.placeholder(tf.float32, shape=[self.batch_size, nouttime, ncoords], name="y") # ALWAYS endeffectors for now.

            # Set up optimizer, compute and apply gradients
            scores, _ = self.model.predict(self.X, is_training=True)
            regression_loss = tf.losses.mean_squared_error(labels=self.y, predictions=scores)
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(regression_loss)

            # Calculate metrics
            scores_eval, _ = self.model.predict(self.X, is_training=False)
            self.val_loss_op = tf.losses.mean_squared_error(labels=self.y, predictions=scores_eval)
            
            pred = tf.reshape(scores_eval, [-1, ncoords])  #3
            targ = tf.reshape(self.y, [-1, ncoords])   #3
            self.accuracy_op = tf.reduce_mean(tf.norm(tf.subtract(pred, targ), axis=1), name="accuracy")
            
            tf.summary.scalar('Train_Loss', regression_loss)
            tf.summary.scalar('Train_RMS', self.accuracy_op)
            self.train_summary_op = tf.summary.merge_all()
            
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            
            if not len(kwargs) == 0:
                varlist = self.get_tensors_in_checkpoint_file(file_name=kwargs['log_dir'],all_tensors=True,tensor_name=None)
                variables = self.build_tensors_in_checkpoint_file(varlist)
                
                self.loader = tf.train.Saver(variables)
            print('Built graph!')

    def load(self):
        self.saver.restore(self.session, os.path.join(self.log_dir, 'model.ckpt'))
        
    def save(self):
        self.saver.save(self.session, os.path.join(self.log_dir, 'model.ckpt'))

    def save_step(self,step):
        self.saver.save(self.session, os.path.join(self.log_dir, 'model_' + str(step) + '.ckpt'))
    
    def load_step(self, step, **kwargs):

        if len(kwargs) == 0:
            log_dir = self.log_dir
        else:
            log_dir = kwargs['log_dir']

        self.saver.restore(self.session, os.path.join(log_dir, 'model_' + str(step) + '.ckpt'))

    def make_model_name(self):
        if (type(self.model).__name__ == 'ConvRModel'):
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
        elif (type(self.model).__name__ == 'ConvRModel_new'):

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
        elif (type(self.model).__name__ == 'RecurrentRModel'):
            # Make model name
            units = ('-'.join(str(i) for i in self.model.nppfilters))
            parts_name = [self.model.rec_blocktype, str(self.model.npplayers), units, str(self.model.n_recunits)]

            # Create model directory
            name = '_'.join(parts_name)
            if self.model.seed is not None: name += '_' + str(self.model.seed)
        elif (type(self.model).__name__ == 'RecurrentRModel_new'):
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
            learning_rate= 0.0005, #0.0005,
            batch_size=256,
            val_steps=200,
            early_stopping_epochs=5,
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
        early_stopping_steps = early_stopping_epochs #* steps_per_epoch
        self.batch_size = batch_size

        if normalize:
            self.train_data_mean = float(self.dataset.train_data_mean)
            self.train_data_std = float(np.std(self.dataset.train_data))
        else:
            self.train_data_mean = self.train_data_std = 0
        train_params = {'train_mean': self.train_data_mean,
                        'train_std': self.train_data_std}
        val_params = {'validation_loss': 1e10, 'validation_accuracy': 0}
        test_params = {'test_accuracy': 0}

        if retrain_same_init:
            old_exp_dir = os.path.join(MODELS_DIR,'experiment_' + str(old_exp_dir))
            name = self.make_model_name()
            log_dir = os.path.join(old_exp_dir, name)
            log_dir = os.path.join(log_dir, 'model_0.ckpt')
#             self.load_step(0, log_dir = log_dir)
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
                    check1, check2, check3 = 5000, 10000, 15000
                else:
                    check1, check2, check3 = 10000, 20000, 30000
        except:
            if (self.model.arch_type == 'spatial_temporal') or (self.model.arch_type == 'temporal_spatial') or (self.model.arch_type == 'spatiotemporal'):
                if batch_size == 512:
                    check1, check2, check3 = 1000, 2500, 5000
                else:
                    check1, check2, check3 = 2000, 5000, 10000

        not_improved = 0
        end_training = 0
        val_params = {}

        for self.global_step in range(max_iter):
            
            # Training step
            batch_X, batch_y = self.dataset.next_trainbatch(
                batch_size, self.global_step % steps_per_epoch)
            feed_dict = {self.X: batch_X - self.train_data_mean,
                        self.y: batch_y, 
                        self.learning_rate: learning_rate}
            self.session.run(self.train_op, feed_dict)

            # Validate/save periodically
            if self.global_step % val_steps == 0:
                # Summarize, print progress

                loss_val, acc_val = self.save_summary(feed_dict)
                if verbose:
                    print('Step : %4d, Validation RMS : %.2f' % (self.global_step, acc_val))
                    print('best_loss:', self.best_loss, 'loss:', loss_val)

                if loss_val < self.best_loss:
                    self.best_loss = loss_val
                    self.validation_accuracy = acc_val
                    self.save()
                    val_params = {
                        'validation_loss': float(self.best_loss), 
                        'validation_accuracy': float(acc_val)}
                    not_improved = 0
                else:
                    not_improved += 1

                if not_improved >= early_stopping_steps:  #steps_per_epoch: #
                    learning_rate /= 4
                    print('lr:', learning_rate)
                    not_improved = 0
                    end_training += 1
                    self.load()

                if end_training == 2: #early_stopping_epochs:  #2
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
        test_accuracy = self.test_model()
        test_params = {'test_accuracy': float(test_accuracy)}

        make_config_file(self.model, train_params, val_params, test_params)
        self.session.close()

    def save_summary(self, feed_dict):
        """Create and save summaries for training and validation."""
        train_summary = self.session.run(self.train_summary_op, feed_dict)
        self.train_summary.add_summary(train_summary, self.global_step)
        validation_loss, validation_accuracy = self.eval()
        validation_summary = tf.Summary()
        validation_summary.value.add(tag='Validation_Loss', simple_value=validation_loss)
        validation_summary.value.add(tag='Validation_Acc', simple_value=validation_accuracy)
        self.val_summary.add_summary(validation_summary, self.global_step)
        
        return validation_loss, validation_accuracy

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
        for i in range(num_iter):
            batch_X, batch_y = self.dataset.next_valbatch(self.batch_size, step=i)
            feed_dict = {self.X: batch_X - self.train_data_mean, self.y: batch_y}
            loss_val[i], acc_val[i] = self.session.run([self.val_loss_op, self.accuracy_op], feed_dict)
        return loss_val.mean(), acc_val.mean()

    def test_model(self):
        """Evaluate test performance.
        
        Returns
        -------
        test_accuracy : float, accuracy on the test data
        
        """
        num_iter = self.test_dataset.test_data.shape[0] // self.batch_size
        self.load()
        acc_test = []
        for i in range(num_iter):
            batch_X, batch_y = self.test_dataset.next_valbatch(self.batch_size, 'test', step=i)

            feed_dict = {self.X: batch_X - self.train_data_mean, 
                        self.y: batch_y}
            acc = self.session.run([self.accuracy_op], feed_dict)
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
    _, ncoords, _ = dataset.test_targets.shape
    
    if type(model).__name__ == 'ConvRModel':
            layers = model.nlayers
            stride = model.t_stride
            nouttime = 400
            for i in range(layers):
                nouttime = int(np.ceil(nouttime/stride))
            dataset.set_outtime(nouttime)
    elif (type(model).__name__ == 'ConvRModel_new'):
        layers = model.nlayers
        stride = model.t_stride
        nouttime = 400
        for i in range(layers):
            nouttime = int(np.ceil(nouttime/stride[i]))
        dataset.set_outtime(nouttime)
    else:
        nouttime = 400
    
    num_steps = nsamples // batch_size

    # Retrieve training mean, if data was normalized
    path_to_config_file = os.path.join(model.model_path, 'config.yaml')
    with open(path_to_config_file, 'r') as myfile:
        model_config = yaml.load(myfile)
    train_mean = model_config['train_mean']

    mygraph = tf.Graph()
    with mygraph.as_default():
        # Declare placeholders for input data and labels
        X = tf.placeholder(tf.float32, shape=[batch_size, ninputs, ntime, 2], name="X")
        y = tf.placeholder(tf.float32, shape=[batch_size, nouttime, ncoords], name="y")

        # Compute scores and accuracy
        scores, _ = model.predict(X, is_training=False)
        pred = tf.reshape(scores, [-1, ncoords])
        targ = tf.reshape(y, [-1, ncoords])
        accuracy = tf.reduce_mean(tf.norm(tf.subtract(pred, targ), axis=1), name="accuracy")

        # Test the `model`!
        restorer = tf.train.Saver()
        myconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, gpu_options=gpu_options)
        with tf.Session(config=myconfig) as sess:
            ckpt_filepath = os.path.join(model.model_path, 'model.ckpt')
            restorer.restore(sess, ckpt_filepath)
            
            test_accuracy = []
            for step in range(num_steps):         
                batch_x, batch_y = dataset.next_valbatch(batch_size, 'test', step)
                acc = sess.run([accuracy], feed_dict={X: batch_x - train_mean, y: batch_y})
                test_accuracy.append(acc)

    return np.mean(test_accuracy)


# Auxiliary Functions

def train_val_split(data, labels):
    num_train = int(0.9*data.shape[0])
    train_data, train_labels = data[:num_train], labels[:num_train]
    val_data, val_labels = data[num_train:], labels[num_train:]

    return (train_data, train_labels, val_data, val_labels)


def make_config_file(model, train_params, val_params, test_params, **kwargs):
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

    # Save yaml file in the model's path
    # path_to_yaml_file = os.path.join(model.model_path, 'config.yaml')
    with open(path_to_yaml_file, 'w') as myfile:
        yaml.dump(mydict, myfile, default_flow_style=False)
        yaml.dump(train_params, myfile, default_flow_style=False)
        yaml.dump(val_params, myfile, default_flow_style=False)
        yaml.dump(test_params, myfile, default_flow_style=False)

    return
