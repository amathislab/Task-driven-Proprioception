'''
Class and forward pass definitions for various neural network models.

'''


import os
from collections import OrderedDict

import tensorflow as tf
slim = tf.contrib.slim
cudnn_rnn = tf.contrib.cudnn_rnn

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
# MODELS_DIR = os.path.join(os.path.dirname(CUR_DIR), '../nn-training/')
from path_utils import MODELS_DIR

class ConvModel():
    """Defines a convolutional neural network model of the proprioceptive system."""

    def __init__(self,
            experiment_id, nclasses, arch_type, nlayers, n_skernels, n_tkernels, s_kernelsize,
            t_kernelsize, s_stride, t_stride, seed=None, train=True):
        """Set up hyperparameters of the convolutional network.

        Arguments
        ---------
        experiment_id : int, identifier for model path
        nclasses : int, number of classes in the classification problem.
        arch_type : {'spatial_temporal', 'spatiotemporal', 'temporal_spatial'} str, defines the type
            of convolutional neural network model.
        nlayers : int, number of layers in the cnn model.
        n_skernels : list of ints, number of kernels for spatial processing.
        n_tkernels : list of ints, number of kernels for temporal processing.
        s_kernelsize : int, size of the spatial kernel.
        t_kernelsize : int, size of the temporal kernel.
        s_stride : int, stride along the spatial dimension.
        t_stride : int, stride along the temporal dimension.
        seed : int, for saving random initializations of networks.
        train : bool, is the network meant to be trained or not

        """
        # print(len(n_skernels))
        # print(len(n_tkernels))
        # print(nlayers)
        assert (len(n_skernels) == len(n_tkernels) == nlayers), \
            "Number of spatial and temporal processing layers must be equal!"
        if arch_type == 'spatiotemporal':
            n_tkernels = n_skernels
            t_kernelsize = s_kernelsize
            t_stride = s_stride

        self.experiment_id = experiment_id
        self.nclasses = nclasses
        self.arch_type = arch_type
        self.nlayers = nlayers
        self.n_tkernels = n_tkernels
        self.n_skernels = n_skernels
        self.t_kernelsize = t_kernelsize
        self.s_kernelsize = s_kernelsize
        self.t_stride = t_stride
        self.s_stride = s_stride
        self.seed = seed

        # Make model name
        if arch_type == 'spatial_temporal':
            kernels = ('-'.join(str(i) for i in n_skernels)) + '_' + ('-'.join(str(i) for i in n_tkernels))
        elif arch_type == 'temporal_spatial':
            kernels = ('-'.join(str(i) for i in n_tkernels)) + '_' + ('-'.join(str(i) for i in n_skernels))
        else:
            kernels = ('-'.join(str(i) for i in n_skernels))

        parts_name = [arch_type, str(nlayers), kernels,
                      ''.join(str(i) for i in [s_kernelsize, s_stride, t_kernelsize, t_stride])]

        # Create model directory
        self.name = '_'.join(parts_name)
        
        if seed is not None: self.name += '_' + str(self.seed)
        if not train: self.name += 'r'
            
        exp_dir = os.path.join(MODELS_DIR, f'experiment_{self.experiment_id}')
        self.model_path = os.path.join(exp_dir, self.name)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        # Additional useful parameters
        self.num_parameters = 0
        self.is_training = True

    def predict(self, X, is_training=True):
        """Computes the scores (forward pass) for the given network.

        Arguments
        ---------
        X : tf.tensor [batch_size, num_inputs, num_timesteps, 2], input tensor for which scores must
            be calculated.

        Returns
        -------
        score : tf.tensor [batch_size, nclasses], computed scores by passing X through the network.
        probabilities : tf.tensor [batch_size, nclasses], softmax probabilities.
        net : orderedDict, contains all layer representations.

        """
        net = OrderedDict([])

        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE):
            score = X
            batch_size = X.get_shape()[0]

            with slim.arg_scope([slim.conv2d], data_format='NHWC', normalizer_fn=slim.layer_norm):

                spatial_conv = lambda score, layer_id: slim.conv2d(
                    score, self.n_skernels[layer_id], [self.s_kernelsize, 1], [self.s_stride, 1],
                    scope=f'Spatial{layer_id}')
                temporal_conv = lambda score, layer_id: slim.conv2d(
                    score, self.n_tkernels[layer_id], [1, self.t_kernelsize], [1, self.t_stride],
                    scope=f'Temporal{layer_id}')
                spatiotemporal_conv = lambda score, layer_id: slim.conv2d(
                    score, self.n_skernels[layer_id], [self.s_kernelsize, self.t_kernelsize], 
                    [self.s_stride, self.t_stride], scope=f'Spatiotemporal{layer_id}')

                if self.arch_type == 'spatial_temporal':
                    for layer in range(self.nlayers):
                        score = spatial_conv(score, layer)
                        net[f'spatial{layer}'] = score
                    for layer in range(self.nlayers):
                        score = temporal_conv(score, layer)
                        net[f'temporal{layer}'] = score

                elif self.arch_type == 'temporal_spatial':
                    for layer in range(self.nlayers):
                        score = temporal_conv(score, layer)
                        net[f'temporal{layer}'] = score
                    for layer in range(self.nlayers):
                        score = spatial_conv(score, layer)
                        net[f'spatial{layer}'] = score

                elif self.arch_type == 'spatiotemporal':
                    for layer in range(self.nlayers):
                        score = spatiotemporal_conv(score, layer)
                        net[f'spatiotemporal{layer}'] = score

                
                score = tf.reshape(score, [batch_size, -1])
                score = slim.dropout(score, 0.7, is_training=is_training)
                score = slim.fully_connected(score, self.nclasses, None, scope='Classifier')

                net['score'] = score

                probabilities = tf.nn.softmax(score, name="Y_proba")

        return score, probabilities, net

class ConvModel_new():
    """Defines a convolutional neural network model of the proprioceptive system."""

    def __init__(self,
            experiment_id, nclasses, arch_type, nlayers, n_skernels, n_tkernels, s_kernelsize,
            t_kernelsize, s_stride, t_stride, seed=None, train=True):
        """Set up hyperparameters of the convolutional network.

        Arguments
        ---------
        experiment_id : int, identifier for model path
        nclasses : int, number of classes in the classification problem.
        arch_type : {'spatial_temporal', 'spatiotemporal', 'temporal_spatial'} str, defines the type
            of convolutional neural network model.
        nlayers : int, number of layers in the cnn model.
        n_skernels : list of ints, number of kernels for spatial processing.
        n_tkernels : list of ints, number of kernels for temporal processing.
        s_kernelsize : int, size of the spatial kernel.
        t_kernelsize : int, size of the temporal kernel.
        s_stride : int, stride along the spatial dimension.
        t_stride : int, stride along the temporal dimension.
        seed : int, for saving random initializations of networks.
        train : bool, is the network meant to be trained or not

        """
        # print(len(n_skernels))
        # print(len(n_tkernels))
        # print(nlayers)
        assert (len(n_skernels) == len(n_tkernels) == nlayers), \
            "Number of spatial and temporal processing layers must be equal!"
        if arch_type == 'spatiotemporal':
            n_tkernels = n_skernels
            t_kernelsize = s_kernelsize
            # t_stride = s_stride

        self.experiment_id = experiment_id
        self.nclasses = nclasses
        self.arch_type = arch_type
        self.nlayers = nlayers
        self.n_tkernels = n_tkernels
        self.n_skernels = n_skernels
        self.t_kernelsize = t_kernelsize
        self.s_kernelsize = s_kernelsize
        self.t_stride = t_stride
        self.s_stride = s_stride
        self.seed = seed

        max_tstride = self.t_stride.count(2)**2
        max_sstride = self.s_stride.count(2)**2

        # Make model name
        if arch_type == 'spatial_temporal':
            kernels = ('-'.join(str(i) for i in n_skernels)) + '_' + ('-'.join(str(i) for i in n_tkernels))
        elif arch_type == 'temporal_spatial':
            kernels = ('-'.join(str(i) for i in n_tkernels)) + '_' + ('-'.join(str(i) for i in n_skernels))
        else:
            kernels = ('-'.join(str(i) for i in n_skernels))

        parts_name = [arch_type, str(nlayers), kernels,
                      ''.join(str(i) for i in [s_kernelsize, max_sstride, t_kernelsize, max_tstride])]

        # Create model directory
        self.name = '_'.join(parts_name)
        
        if seed is not None: self.name += '_' + str(self.seed)
        if not train: self.name += 'r'
            
        exp_dir = os.path.join(MODELS_DIR, f'experiment_{self.experiment_id}')
        self.model_path = os.path.join(exp_dir, self.name)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        # Additional useful parameters
        self.num_parameters = 0
        self.is_training = True

    def predict(self, X, is_training=True):
        """Computes the scores (forward pass) for the given network.

        Arguments
        ---------
        X : tf.tensor [batch_size, num_inputs, num_timesteps, 2], input tensor for which scores must
            be calculated.

        Returns
        -------
        score : tf.tensor [batch_size, nclasses], computed scores by passing X through the network.
        probabilities : tf.tensor [batch_size, nclasses], softmax probabilities.
        net : orderedDict, contains all layer representations.

        """
        net = OrderedDict([])

        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE):
            score = X
            batch_size = X.get_shape()[0]

            with slim.arg_scope([slim.conv2d], data_format='NHWC', normalizer_fn=slim.layer_norm):

                spatial_conv = lambda score, layer_id: slim.conv2d(
                    score, self.n_skernels[layer_id], [self.s_kernelsize, 1], [self.s_stride[layer_id], 1],
                    scope=f'Spatial{layer_id}')
                temporal_conv = lambda score, layer_id: slim.conv2d(
                    score, self.n_tkernels[layer_id], [1, self.t_kernelsize], [1, self.t_stride[layer_id]],
                    scope=f'Temporal{layer_id}')
                spatiotemporal_conv = lambda score, layer_id: slim.conv2d(
                    score, self.n_skernels[layer_id], [self.s_kernelsize, self.t_kernelsize], 
                    [self.s_stride[layer_id], self.t_stride[layer_id]], scope=f'Spatiotemporal{layer_id}')

                if self.arch_type == 'spatial_temporal':
                    for layer in range(self.nlayers):
                        score = spatial_conv(score, layer)
                        net[f'spatial{layer}'] = score
                    for layer in range(self.nlayers):
                        score = temporal_conv(score, layer)
                        net[f'temporal{layer}'] = score

                elif self.arch_type == 'temporal_spatial':
                    for layer in range(self.nlayers):
                        score = temporal_conv(score, layer)
                        net[f'temporal{layer}'] = score
                    for layer in range(self.nlayers):
                        score = spatial_conv(score, layer)
                        net[f'spatial{layer}'] = score

                elif self.arch_type == 'spatiotemporal':
                    for layer in range(self.nlayers):
                        score = spatiotemporal_conv(score, layer)
                        net[f'spatiotemporal{layer}'] = score

                
                score = tf.reshape(score, [batch_size, -1])
                score = slim.dropout(score, 0.7, is_training=is_training)
                score = slim.fully_connected(score, self.nclasses, None, scope='Classifier')

                net['score'] = score

                probabilities = tf.nn.softmax(score, name="Y_proba")

        return score, probabilities, net

class BarlowTwinsModel():
    """Defines a convolutional neural network model of the proprioceptive system."""

    def __init__(self,
            experiment_id, nclasses, arch_type, nlayers, n_skernels, n_tkernels, s_kernelsize,
            t_kernelsize, s_stride, t_stride, nlayers_fc, nunits, seed=None, lambd=5e-3, 
            train=True, with_projector = True, task_transfer=False):
        """Set up hyperparameters of the convolutional network.

        Arguments
        ---------
        experiment_id : int, identifier for model path
        nclasses : int, number of classes in the classification problem.
        arch_type : {'spatial_temporal', 'spatiotemporal', 'temporal_spatial'} str, defines the type
            of convolutional neural network model.
        nlayers : int, number of layers in the cnn model.
        n_skernels : list of ints, number of kernels for spatial processing.
        n_tkernels : list of ints, number of kernels for temporal processing.
        s_kernelsize : int, size of the spatial kernel.
        t_kernelsize : int, size of the temporal kernel.
        s_stride : int, stride along the spatial dimension.
        t_stride : int, stride along the temporal dimension.
        seed : int, for saving random initializations of networks.
        train : bool, is the network meant to be trained or not
        with_projector : bool, if you want the forward pass to go through the projector.
        task_transfer : bool, if you want to add a readout on top.

        """
        # print(len(n_skernels))
        # print(len(n_tkernels))
        # print(nlayers)
        assert (len(n_skernels) == len(n_tkernels) == nlayers), \
            "Number of spatial and temporal processing layers must be equal!"
        if arch_type == 'spatiotemporal':
            n_tkernels = n_skernels
            t_kernelsize = s_kernelsize
            t_stride = s_stride

        self.experiment_id = experiment_id
        self.nclasses = nclasses
        self.arch_type = arch_type
        self.nlayers = nlayers
        self.n_tkernels = n_tkernels
        self.n_skernels = n_skernels
        self.t_kernelsize = t_kernelsize
        self.s_kernelsize = s_kernelsize
        self.t_stride = t_stride
        self.s_stride = s_stride
        self.nlayers_fc = nlayers_fc
        self.nunits = nunits
        self.seed = seed
        self.lambd = lambd
        self.task_transfer = task_transfer
        self.with_projector = with_projector

        # Make model name
        if arch_type == 'spatial_temporal':
            kernels = ('-'.join(str(i) for i in n_skernels)) + '_' + ('-'.join(str(i) for i in n_tkernels))
        elif arch_type == 'temporal_spatial':
            kernels = ('-'.join(str(i) for i in n_tkernels)) + '_' + ('-'.join(str(i) for i in n_skernels))
        else:
            kernels = ('-'.join(str(i) for i in n_skernels))

        parts_name = [arch_type, str(nlayers), kernels,
                      ''.join(str(i) for i in [s_kernelsize, s_stride, t_kernelsize, t_stride])]

        # Create model directory
        self.name = '_'.join(parts_name)
        
        if seed is not None: self.name += '_' + str(self.seed)
        if not train: self.name += 'r'
            
        exp_dir = os.path.join(MODELS_DIR, f'experiment_{self.experiment_id}')
        self.model_path = os.path.join(exp_dir, self.name)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        # Additional useful parameters
        self.num_parameters = 0
        self.is_training = True

    def predict(self, X, is_training=True):
        """Computes the scores (forward pass) for the given network.

        Arguments
        ---------
        X : tf.tensor [batch_size, num_inputs, num_timesteps, 2], input tensor for which scores must
            be calculated.

        Returns
        -------
        score : tf.tensor [batch_size, nclasses], computed scores by passing X through the network.
        probabilities : tf.tensor [batch_size, nclasses], softmax probabilities.
        net : orderedDict, contains all layer representations.

        """
        net = OrderedDict([])

        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE):
            score = X
            batch_size = X.get_shape()[0]

            with slim.arg_scope([slim.conv2d], data_format='NHWC', normalizer_fn=slim.layer_norm):

                spatial_conv = lambda score, layer_id: slim.conv2d(
                    score, self.n_skernels[layer_id], [self.s_kernelsize, 1], [self.s_stride, 1],
                    scope=f'Spatial{layer_id}')
                temporal_conv = lambda score, layer_id: slim.conv2d(
                    score, self.n_tkernels[layer_id], [1, self.t_kernelsize], [1, self.t_stride],
                    scope=f'Temporal{layer_id}')
                spatiotemporal_conv = lambda score, layer_id: slim.conv2d(
                    score, self.n_skernels[layer_id], [self.s_kernelsize, self.t_kernelsize], 
                    [self.s_stride, self.t_stride], scope=f'Spatiotemporal{layer_id}')
                ## Fully connected with batch norm and relu
                fully_connected = lambda score, layer_id: slim.fully_connected(
                    score, self.nunits[layer_id], activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, scope=f'FC{layer_id}')

                if self.arch_type == 'spatial_temporal':
                    for layer in range(self.nlayers):
                        score = spatial_conv(score, layer)
                        net[f'spatial{layer}'] = score
                    for layer in range(self.nlayers):
                        score = temporal_conv(score, layer)
                        net[f'temporal{layer}'] = score

                elif self.arch_type == 'temporal_spatial':
                    for layer in range(self.nlayers):
                        score = temporal_conv(score, layer)
                        net[f'temporal{layer}'] = score
                    for layer in range(self.nlayers):
                        score = spatial_conv(score, layer)
                        net[f'spatial{layer}'] = score

                elif self.arch_type == 'spatiotemporal':
                    for layer in range(self.nlayers):
                        score = spatiotemporal_conv(score, layer)
                        net[f'spatiotemporal{layer}'] = score

                
                score = tf.reshape(score, [batch_size, -1])
                
                if self.with_projector:
                    score = slim.dropout(score, 0.7, is_training=is_training)
                    
                    for layer in range(self.nlayers_fc -1):
                        score = fully_connected(score, layer)
                        net[f'projector{layer}'] = score

                        ### TO REMOVE THIS AFTER (remove also -1)
                        # score = slim.batch_norm(score)
                        # score = tf.nn.relu(score)
                        # net[f'projector{layer}'] = score
                    
                    ## AND ADD THIS
                    layer_id = layer+1
                    score = slim.fully_connected(score, self.nunits[layer_id], activation_fn=None, normalizer_fn=None, scope=f'FC{layer_id}')
                    # score = slim.fully_connected(score, self.nunits[layer_id], None, scope=f'FC{layer_id}')
                    net[f'projector{layer_id}'] = score

                # score = slim.fully_connected(score, self.nclasses, None, scope='Classifier')

                if self.task_transfer:
                    score = slim.fully_connected(score, self.nclasses, None, scope='Classifier')
                    net[f'classifier{layer}'] = score


                net['score'] = score

                probabilities = tf.nn.softmax(score, name="Y_proba")

        return score, probabilities, net

class BarlowTwinsModel_new():
    """Defines a convolutional neural network model of the proprioceptive system."""

    def __init__(self,
            experiment_id, nclasses, arch_type, nlayers, n_skernels, n_tkernels, s_kernelsize,
            t_kernelsize, s_stride, t_stride, nlayers_fc, nunits, seed=None, lambd=5e-3, 
            train=True, with_projector = True, task_transfer=False):
        """Set up hyperparameters of the convolutional network.

        Arguments
        ---------
        experiment_id : int, identifier for model path
        nclasses : int, number of classes in the classification problem.
        arch_type : {'spatial_temporal', 'spatiotemporal', 'temporal_spatial'} str, defines the type
            of convolutional neural network model.
        nlayers : int, number of layers in the cnn model.
        n_skernels : list of ints, number of kernels for spatial processing.
        n_tkernels : list of ints, number of kernels for temporal processing.
        s_kernelsize : int, size of the spatial kernel.
        t_kernelsize : int, size of the temporal kernel.
        s_stride : list of ints, stride along the spatial dimension per layer.
        t_stride : list of ints, stride along the temporal dimension per layer.
        seed : int, for saving random initializations of networks.
        train : bool, is the network meant to be trained or not
        with_projector : bool, if you want the forward pass to go through the projector.
        task_transfer : bool, if you want to add a readout on top.

        """
        # print(len(n_skernels))
        # print(len(n_tkernels))
        # print(nlayers)
        assert (len(n_skernels) == len(n_tkernels) == nlayers), \
            "Number of spatial and temporal processing layers must be equal!"
        if arch_type == 'spatiotemporal':
            n_tkernels = n_skernels
            t_kernelsize = s_kernelsize
            # t_stride = s_stride

        self.experiment_id = experiment_id
        self.nclasses = nclasses
        self.arch_type = arch_type
        self.nlayers = nlayers
        self.n_tkernels = n_tkernels
        self.n_skernels = n_skernels
        self.t_kernelsize = t_kernelsize
        self.s_kernelsize = s_kernelsize
        self.t_stride = t_stride
        self.s_stride = s_stride
        self.nlayers_fc = nlayers_fc
        self.nunits = nunits
        self.seed = seed
        self.lambd = lambd
        self.task_transfer = task_transfer
        self.with_projector = with_projector

        max_tstride = self.t_stride.count(2)**2
        max_sstride = self.s_stride.count(2)**2

        # Make model name
        if arch_type == 'spatial_temporal':
            kernels = ('-'.join(str(i) for i in n_skernels)) + '_' + ('-'.join(str(i) for i in n_tkernels))
        elif arch_type == 'temporal_spatial':
            kernels = ('-'.join(str(i) for i in n_tkernels)) + '_' + ('-'.join(str(i) for i in n_skernels))
        else:
            kernels = ('-'.join(str(i) for i in n_skernels))

        parts_name = [arch_type, str(nlayers), kernels,
                      ''.join(str(i) for i in [s_kernelsize, max_sstride, t_kernelsize, max_tstride])]

        # Create model directory
        self.name = '_'.join(parts_name)
        
        if seed is not None: self.name += '_' + str(self.seed)
        if not train: self.name += 'r'
            
        exp_dir = os.path.join(MODELS_DIR, f'experiment_{self.experiment_id}')
        self.model_path = os.path.join(exp_dir, self.name)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        # Additional useful parameters
        self.num_parameters = 0
        self.is_training = True

    def predict(self, X, is_training=True):
        """Computes the scores (forward pass) for the given network.

        Arguments
        ---------
        X : tf.tensor [batch_size, num_inputs, num_timesteps, 2], input tensor for which scores must
            be calculated.

        Returns
        -------
        score : tf.tensor [batch_size, nclasses], computed scores by passing X through the network.
        probabilities : tf.tensor [batch_size, nclasses], softmax probabilities.
        net : orderedDict, contains all layer representations.

        """
        net = OrderedDict([])

        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE):
            score = X
            batch_size = X.get_shape()[0]

            with slim.arg_scope([slim.conv2d], data_format='NHWC', normalizer_fn=slim.layer_norm):

                spatial_conv = lambda score, layer_id: slim.conv2d(
                    score, self.n_skernels[layer_id], [self.s_kernelsize, 1], [self.s_stride[layer_id], 1],
                    scope=f'Spatial{layer_id}')
                temporal_conv = lambda score, layer_id: slim.conv2d(
                    score, self.n_tkernels[layer_id], [1, self.t_kernelsize], [1, self.t_stride[layer_id]],
                    scope=f'Temporal{layer_id}')
                spatiotemporal_conv = lambda score, layer_id: slim.conv2d(
                    score, self.n_skernels[layer_id], [self.s_kernelsize, self.t_kernelsize], 
                    [self.s_stride[layer_id], self.t_stride[layer_id]], scope=f'Spatiotemporal{layer_id}')
                ## Fully connected with batch norm and relu
                fully_connected = lambda score, layer_id: slim.fully_connected(
                    score, self.nunits[layer_id], activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, scope=f'FC{layer_id}')

                if self.arch_type == 'spatial_temporal':
                    for layer in range(self.nlayers):
                        score = spatial_conv(score, layer)
                        net[f'spatial{layer}'] = score
                    for layer in range(self.nlayers):
                        score = temporal_conv(score, layer)
                        net[f'temporal{layer}'] = score

                elif self.arch_type == 'temporal_spatial':
                    for layer in range(self.nlayers):
                        score = temporal_conv(score, layer)
                        net[f'temporal{layer}'] = score
                    for layer in range(self.nlayers):
                        score = spatial_conv(score, layer)
                        net[f'spatial{layer}'] = score

                elif self.arch_type == 'spatiotemporal':
                    for layer in range(self.nlayers):
                        score = spatiotemporal_conv(score, layer)
                        net[f'spatiotemporal{layer}'] = score

                
                score = tf.reshape(score, [batch_size, -1])
                
                if self.with_projector:
                    score = slim.dropout(score, 0.7, is_training=is_training)
                    
                    for layer in range(self.nlayers_fc -1):
                        score = fully_connected(score, layer)
                        net[f'projector{layer}'] = score
                    layer_id = layer+1
                    score = slim.fully_connected(score, self.nunits[layer_id], activation_fn=None, normalizer_fn=None, scope=f'FC{layer_id}')
                    net[f'projector{layer_id}'] = score

                if self.task_transfer:
                    score = slim.fully_connected(score, self.nclasses, None, scope='Classifier')
                    net[f'classifier{layer}'] = score


                net['score'] = score

                probabilities = tf.nn.softmax(score, name="Y_proba")

        return score, probabilities, net

class BarlowTwinsModel_rec():
    """Defines a Barlow Twins model with a RNN backbone of the proprioceptive system."""

    def __init__(self,
            experiment_id, nclasses, rec_blocktype, n_recunits, npplayers, nppfilters, 
            s_kernelsize, s_stride, nlayers_fc, nunits, CPU=False, seed=None, lambd=5e-3, 
            train=True, with_projector = True, task_transfer=False):
        """Set up the hyperparameters of the recurrent model.

        Arguments
        ---------
        experiment_id : int, identifier for model path
        nclasses : int, number of classes in the classification problem.
        rec_blocktype: {'lstm', 'gru'} str, type of recurrent block.
        n_recunits : int, number of units in the recurrent block.
        npplayers : int, number of layers in the fully-connected module.
        nppfilters : list of ints, number of filters (spatial convolutions) for spatial processing.
        s_kernelsize : int, size of conv kernel
        s_stride : int, stride for conv kernel
        nlayers_fc : int, number of layers of the fully-connected projector.
        nunits : int, number of units in each layer of the projector.
        seed : int, for saving random initializations
        train : bool, whether to train the model or not (just save random initialization)
        with_projector : bool, if you want the forward pass to go through the projector.
        task_transfer : bool, if you want to add a readout on top.
        """
        assert len(nppfilters) == npplayers
        assert rec_blocktype in  ('lstm', 'gru')

        self.experiment_id = experiment_id
        self.nclasses = nclasses
        self.rec_blocktype = rec_blocktype
        self.n_recunits = n_recunits
        self.npplayers = npplayers
        self.nppfilters = nppfilters
        self.s_kernelsize = s_kernelsize
        self.s_stride = s_stride
        self.nlayers_fc = nlayers_fc
        self.nunits = nunits
        self.seed = seed
        self.CPU = CPU
        self.lambd = lambd
        self.task_transfer = task_transfer
        self.with_projector = with_projector

        # Make model name
        units = ('-'.join(str(i) for i in nppfilters))
        parts_name = [rec_blocktype, str(npplayers), units, str(n_recunits)]

        # Create model directory
        self.name = '_'.join(parts_name)
        if seed is not None: self.name += '_' + str(self.seed)
        if not train: self.name += 'r'

        exp_dir = os.path.join(MODELS_DIR, f'experiment_{self.experiment_id}')
        self.model_path = os.path.join(exp_dir, self.name)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # Additional useful parameters
        self.num_parameters = 0
        self.is_training = True

    def predict(self, X, is_training=True):
        """Computes the scores (forward pass) for the given network.

        Arguments
        ---------
        X : tf.tensor [batch_size, num_inputs, num_timesteps, 2], input tensor for which scores must
            be calculated.

        Returns
        -------
        score : tf.tensor [batch_size, nclasses], computed scores by passing X through the network.
        probabilities : tf.tensor [batch_size, nclasses], softmax probabilities.
        net : orderedDict, contains all layer representations.

        """
        net = OrderedDict()

        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE):
            score = X
            batch_size = X.get_shape()[0]
            t_size = X.get_shape()[2]

            spatial_conv = lambda score, layer_id: slim.conv2d(
                score, self.nppfilters[layer_id], [self.s_kernelsize, 1], [self.s_stride, 1],
                data_format='NHWC', normalizer_fn=slim.layer_norm, scope=f'Spatial{layer_id}')
            ## Fully connected with batch norm and relu
            fully_connected = lambda score, layer_id: slim.fully_connected(
                score, self.nunits[layer_id], activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, scope=f'FC{layer_id}')

            for layer in range(self.npplayers):
                score = spatial_conv(score, layer)
                net[f'spatial{layer}'] = score

            # `cudnn_rnn` requires the inputs to be of shape [timesteps, batch_size, num_inputs]
            score = tf.transpose(score, [0, 2, 1, 3])
            score = tf.reshape(score, [batch_size, t_size, -1])
            score = tf.transpose(score, [1, 0, 2])
            
            if self.rec_blocktype == 'lstm' and self.CPU == False:
                recurrent_cell = cudnn_rnn.CudnnLSTM(1, self.n_recunits, name='RecurrentBlock')
                score, _ = recurrent_cell.apply(score)
            elif self.rec_blocktype == 'gru' and self.CPU == False:
                recurrent_cell = cudnn_rnn.CudnnGRU(1, self.n_recunits, name='RecurrentBlock')
                score, _ = recurrent_cell.apply(score)
            elif self.rec_blocktype == 'lstm' and self.CPU:
                with tf.variable_scope('RecurrentBlock'):
                    rec_layer = lambda: cudnn_rnn.CudnnCompatibleLSTMCell(self.n_recunits)
                recurrent_cell = tf.nn.rnn_cell.MultiRNNCell([rec_layer() for _ in range(1)])
                score, _ = tf.nn.dynamic_rnn(recurrent_cell, score, dtype=tf.float32)

            score = tf.transpose(score, [1, 0, 2])
            net['recurrent_out'] = score
                
            score = tf.reshape(score, [batch_size, -1])

            if self.with_projector:
                score = slim.dropout(score, 0.7, is_training=is_training)
                
                for layer in range(self.nlayers_fc -1):
                    score = fully_connected(score, layer)
                    net[f'projector{layer}'] = score

                layer_id = layer+1
                score = slim.fully_connected(score, self.nunits[layer_id], activation_fn=None, normalizer_fn=None, scope=f'FC{layer_id}')
                net[f'projector{layer_id}'] = score


            if self.task_transfer:
                score = slim.fully_connected(score, self.nclasses, activation_fn=None, scope='Classifier')
                net[f'classifier{layer}'] = score

            net['score'] = score

            probabilities = tf.nn.softmax(score, name="Y_proba")
            # probabilities = tf.nn.softmax(score[:, -1, :], name="Y_proba")

        return score, probabilities, net

class BarlowTwinsModel_rec_new():
    """Defines a Barlow Twins model with a RNN backbone of the proprioceptive system."""

    def __init__(self,
            experiment_id, nclasses, rec_blocktype, n_recunits, npplayers, nppfilters, 
            s_kernelsize, s_stride, nlayers_fc, nunits, CPU=False, seed=None, lambd=5e-3, 
            train=True, with_projector = True, task_transfer=False):
        """Set up the hyperparameters of the recurrent model.

        Arguments
        ---------
        experiment_id : int, identifier for model path
        nclasses : int, number of classes in the classification problem.
        rec_blocktype: {'lstm', 'gru'} str, type of recurrent block.
        n_recunits : int, number of units in the recurrent block.
        npplayers : int, number of layers in the fully-connected module.
        nppfilters : list of ints, number of filters (spatial convolutions) for spatial processing.
        s_kernelsize : int, size of conv kernel
        s_stride : int, stride for conv kernel
        nlayers_fc : int, number of layers of the fully-connected projector.
        nunits : int, number of units in each layer of the projector.
        seed : int, for saving random initializations
        train : bool, whether to train the model or not (just save random initialization)
        with_projector : bool, if you want the forward pass to go through the projector.
        task_transfer : bool, if you want to add a readout on top.
        """
        assert len(nppfilters) == npplayers
        assert rec_blocktype in  ('lstm', 'gru')

        self.experiment_id = experiment_id
        self.nclasses = nclasses
        self.rec_blocktype = rec_blocktype
        self.n_recunits = n_recunits
        self.npplayers = npplayers
        self.nppfilters = nppfilters
        self.s_kernelsize = s_kernelsize
        self.s_stride = s_stride
        self.nlayers_fc = nlayers_fc
        self.nunits = nunits
        self.seed = seed
        self.CPU = CPU
        self.lambd = lambd
        self.task_transfer = task_transfer
        self.with_projector = with_projector

        max_sstride = self.s_stride.count(2)**2
        
        # Make model name
        units = ('-'.join(str(i) for i in nppfilters))
        parts_name = [rec_blocktype, str(n_reclayers), str(npplayers), units, str(n_recunits),
                     ''.join(str(i) for i in [s_kernelsize, max_sstride])]

        # Create model directory
        self.name = '_'.join(parts_name)
        if seed is not None: self.name += '_' + str(self.seed)
        if not train: self.name += 'r'

        exp_dir = os.path.join(MODELS_DIR, f'experiment_{self.experiment_id}')
        self.model_path = os.path.join(exp_dir, self.name)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # Additional useful parameters
        self.num_parameters = 0
        self.is_training = True

    def predict(self, X, is_training=True):
        """Computes the scores (forward pass) for the given network.

        Arguments
        ---------
        X : tf.tensor [batch_size, num_inputs, num_timesteps, 2], input tensor for which scores must
            be calculated.

        Returns
        -------
        score : tf.tensor [batch_size, nclasses], computed scores by passing X through the network.
        probabilities : tf.tensor [batch_size, nclasses], softmax probabilities.
        net : orderedDict, contains all layer representations.

        """
        net = OrderedDict()

        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE):
            score = X
            batch_size = X.get_shape()[0]
            t_size = X.get_shape()[2]

            spatial_conv = lambda score, layer_id: slim.conv2d(
                score, self.nppfilters[layer_id], [self.s_kernelsize, 1], [self.s_stride[layer_id], 1],
                data_format='NHWC', normalizer_fn=slim.layer_norm, scope=f'Spatial{layer_id}')
            ## Fully connected with batch norm and relu
            fully_connected = lambda score, layer_id: slim.fully_connected(
                score, self.nunits[layer_id], activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, scope=f'FC{layer_id}')

            for layer in range(self.npplayers):
                score = spatial_conv(score, layer)
                net[f'spatial{layer}'] = score

            # `cudnn_rnn` requires the inputs to be of shape [timesteps, batch_size, num_inputs]
            score = tf.transpose(score, [0, 2, 1, 3])
            score = tf.reshape(score, [batch_size, t_size, -1])
            score = tf.transpose(score, [1, 0, 2])
            
            if self.rec_blocktype == 'lstm' and self.CPU == False:
                recurrent_cell = cudnn_rnn.CudnnLSTM(self.n_reclayers, self.n_recunits, name='RecurrentBlock')
                score, _ = recurrent_cell.apply(score)
            elif self.rec_blocktype == 'gru' and self.CPU == False:
                recurrent_cell = cudnn_rnn.CudnnGRU(self.n_reclayers, self.n_recunits, name='RecurrentBlock')
                score, _ = recurrent_cell.apply(score)
            elif self.rec_blocktype == 'lstm' and self.CPU:
                with tf.variable_scope('RecurrentBlock'):
                    rec_layer = lambda: cudnn_rnn.CudnnCompatibleLSTMCell(self.n_recunits)
                recurrent_cell = tf.nn.rnn_cell.MultiRNNCell([rec_layer() for rec_layer_id in range(self.n_reclayers)])
                score, _ = tf.nn.dynamic_rnn(recurrent_cell, score, dtype=tf.float32)

            score = tf.transpose(score, [1, 0, 2])
            net['recurrent_out'] = score
                
            score = tf.reshape(score, [batch_size, -1])

            if self.with_projector:
                score = slim.dropout(score, 0.7, is_training=is_training)
                
                for layer in range(self.nlayers_fc -1):
                    score = fully_connected(score, layer)

                layer_id = layer+1
                score = slim.fully_connected(score, self.nunits[layer_id], activation_fn=None, normalizer_fn=None, scope=f'FC{layer_id}')
                net[f'projector{layer}'] = score


            if self.task_transfer:
                score = slim.fully_connected(score, self.nclasses, activation_fn=None, scope='Classifier')
                net[f'classifier{layer}'] = score

            net['score'] = score

            probabilities = tf.nn.softmax(score, name="Y_proba")
            # probabilities = tf.nn.softmax(score[:, -1, :], name="Y_proba")

        return score, probabilities, net

class RecurrentModel():
    """Defines a recurrent neural network model of the proprioceptive system."""

    def __init__(
            self, experiment_id, nclasses, rec_blocktype, n_recunits, npplayers, nppfilters, 
            s_kernelsize, s_stride, seed=None, train=True, CPU=False):
        """Set up the hyperparameters of the recurrent model.

        Arguments
        ---------
        experiment_id : int, identifier for model path
        nclasses : int, number of classes in the classification problem.
        rec_blocktype: {'lstm', 'gru'} str, type of recurrent block.
        n_recunits : int, number of units in the recurrent block.
        npplayers : int, number of layers in the fully-connected module.
        nppfilters : list of ints, number of filters (spatial convolutions) for spatial processing.
        s_kernelsize : int, size of conv kernel
        s_stride : int, stride for conv kernel
        seed : int, for saving random initializations
        train : bool, whether to train the model or not (just save random initialization)

        """

        assert len(nppfilters) == npplayers
        assert rec_blocktype in  ('lstm', 'gru')

        self.experiment_id = experiment_id
        self.nclasses = nclasses
        self.rec_blocktype = rec_blocktype
        self.n_recunits = n_recunits
        self.npplayers = npplayers
        self.nppfilters = nppfilters
        self.s_kernelsize = s_kernelsize
        self.s_stride = s_stride
        self.seed = seed
        self.CPU = CPU

        # Make model name
        units = ('-'.join(str(i) for i in nppfilters))
        parts_name = [rec_blocktype, str(npplayers), units, str(n_recunits)]

        # Create model directory
        self.name = '_'.join(parts_name)
        if seed is not None: self.name += '_' + str(self.seed)
        if not train: self.name += 'r'

        exp_dir = os.path.join(MODELS_DIR, f'experiment_{self.experiment_id}')
        self.model_path = os.path.join(exp_dir, self.name)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # Additional useful parameters
        self.num_parameters = 0
        self.is_training = True

    def predict(self, X, is_training=True):

        net = OrderedDict()

        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE):
            score = X
            batch_size = X.get_shape()[0]
            t_size = X.get_shape()[2]

            spatial_conv = lambda score, layer_id: slim.conv2d(
                score, self.nppfilters[layer_id], [self.s_kernelsize, 1], [self.s_stride, 1],
                data_format='NHWC', normalizer_fn=slim.layer_norm, scope=f'Spatial{layer_id}')

            for layer in range(self.npplayers):
                score = spatial_conv(score, layer)
                net[f'spatial{layer}'] = score

            # `cudnn_rnn` requires the inputs to be of shape [timesteps, batch_size, num_inputs]
            score = tf.transpose(score, [0, 2, 1, 3])
            score = tf.reshape(score, [batch_size, t_size, -1])
            score = tf.transpose(score, [1, 0, 2])
            
            if self.rec_blocktype == 'lstm' and self.CPU == False:
                recurrent_cell = cudnn_rnn.CudnnLSTM(1, self.n_recunits, name='RecurrentBlock')
                score, _ = recurrent_cell.apply(score)
            elif self.rec_blocktype == 'gru' and self.CPU == False:
                recurrent_cell = cudnn_rnn.CudnnGRU(1, self.n_recunits, name='RecurrentBlock')
                score, _ = recurrent_cell.apply(score)
            elif self.rec_blocktype == 'lstm' and self.CPU:
                with tf.variable_scope('RecurrentBlock'):
                    rec_layer = lambda: cudnn_rnn.CudnnCompatibleLSTMCell(self.n_recunits)
                recurrent_cell = tf.nn.rnn_cell.MultiRNNCell([rec_layer() for _ in range(1)])
                score, _ = tf.nn.dynamic_rnn(recurrent_cell, score, dtype=tf.float32)

            score = tf.transpose(score, [1, 0, 2])
            net['recurrent_out'] = score

            score = slim.dropout(score, 0.7, is_training=is_training)
            score = slim.fully_connected(score, self.nclasses, activation_fn=None, scope='Classifier')

            net['score'] = score

            probabilities = tf.nn.softmax(score[:, -1, :], name="Y_proba")

        return score[:, -1, :], probabilities, net

class RecurrentModel_new():
    """Defines a recurrent neural network model of the proprioceptive system."""

    def __init__(
            self, experiment_id, nclasses, rec_blocktype, n_reclayers, n_recunits, npplayers, nppfilters, 
            s_kernelsize, s_stride, seed=None, train=True, CPU=False):
        """Set up the hyperparameters of the recurrent model.

        Arguments
        ---------
        experiment_id : int, identifier for model path
        nclasses : int, number of classes in the classification problem.
        rec_blocktype: {'lstm', 'gru'} str, type of recurrent block.
        n_recunits : int, number of units in the recurrent block.
        npplayers : int, number of layers in the fully-connected module.
        nppfilters : list of ints, number of filters (spatial convolutions) for spatial processing.
        s_kernelsize : int, size of conv kernel
        s_stride : int, stride for conv kernel
        seed : int, for saving random initializations
        train : bool, whether to train the model or not (just save random initialization)

        """

        assert len(nppfilters) == npplayers
        assert rec_blocktype in  ('lstm', 'gru')

        self.experiment_id = experiment_id
        self.nclasses = nclasses
        self.rec_blocktype = rec_blocktype
        self.n_reclayers = n_reclayers
        self.n_recunits = n_recunits
        self.npplayers = npplayers
        self.nppfilters = nppfilters
        self.s_kernelsize = s_kernelsize
        self.s_stride = s_stride
        self.seed = seed
        self.CPU = CPU
        
        max_sstride = self.s_stride.count(2)**2
        
        # Make model name
        units = ('-'.join(str(i) for i in nppfilters))
        parts_name = [rec_blocktype, str(n_reclayers), str(npplayers), units, str(n_recunits),
                     ''.join(str(i) for i in [s_kernelsize, max_sstride])]

        # Create model directory
        self.name = '_'.join(parts_name)
        if seed is not None: self.name += '_' + str(self.seed)
        if not train: self.name += 'r'

        exp_dir = os.path.join(MODELS_DIR, f'experiment_{self.experiment_id}')
        self.model_path = os.path.join(exp_dir, self.name)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # Additional useful parameters
        self.num_parameters = 0
        self.is_training = True

    def predict(self, X, is_training=True):

        net = OrderedDict()

        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE):
            score = X
            batch_size = X.get_shape()[0]
            t_size = X.get_shape()[2]

            spatial_conv = lambda score, layer_id: slim.conv2d(
                score, self.nppfilters[layer_id], [self.s_kernelsize, 1], [self.s_stride[layer_id], 1],
                data_format='NHWC', normalizer_fn=slim.layer_norm, scope=f'Spatial{layer_id}')
            
            for layer in range(self.npplayers):
                score = spatial_conv(score, layer)
                net[f'spatial{layer}'] = score
            
            # `cudnn_rnn` requires the inputs to be of shape [timesteps, batch_size, num_inputs]
            score = tf.transpose(score, [0, 2, 1, 3])
            score = tf.reshape(score, [batch_size, t_size, -1])
            score = tf.transpose(score, [1, 0, 2])
            
            if self.rec_blocktype == 'lstm' and self.CPU == False:
                recurrent_cell = cudnn_rnn.CudnnLSTM(self.n_reclayers, self.n_recunits, name='RecurrentBlock')
                score, _ = recurrent_cell.apply(score)
            elif self.rec_blocktype == 'gru' and self.CPU == False:
                recurrent_cell = cudnn_rnn.CudnnGRU(self.n_reclayers, self.n_recunits, name='RecurrentBlock')
                score, _ = recurrent_cell.apply(score)
            elif self.rec_blocktype == 'lstm' and self.CPU:
                with tf.variable_scope('RecurrentBlock'):
                    rec_layer = lambda: cudnn_rnn.CudnnCompatibleLSTMCell(self.n_recunits)
                recurrent_cell = tf.nn.rnn_cell.MultiRNNCell([rec_layer() for rec_layer_id in range(self.n_reclayers)])
                score, _ = tf.nn.dynamic_rnn(recurrent_cell, score, dtype=tf.float32)

            score = tf.transpose(score, [1, 0, 2])
            net['recurrent_out'] = score

            score = slim.dropout(score, 0.7, is_training=is_training)
            score = slim.fully_connected(score, self.nclasses, activation_fn=None, scope='Classifier')

            net['score'] = score

            probabilities = tf.nn.softmax(score[:, -1, :], name="Y_proba")

        return score[:, -1, :], probabilities, net