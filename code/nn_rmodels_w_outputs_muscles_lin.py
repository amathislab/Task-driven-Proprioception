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

class ConvRModel():
    """Defines a convolutional neural network model of the proprioceptive system."""

    def __init__(self,
            experiment_id, arch_type, nlayers, n_skernels, n_tkernels, n_skernels_trans, n_tkernels_trans, s_kernelsize,
            t_kernelsize, s_stride, t_stride, n_outputs, seed=None, train=True,task_transfer=False,decoder=True,nlayers_fc=None,nunits=None):  #n_outputs,
        """Set up hyperparameters of the convolutional network.

        Arguments
        ---------
        experiment_id : int, identifier for model path
        arch_type : {'spatial_temporal', 'spatiotemporal', 'temporal_spatial'} str, defines the type
            of convolutional neural network model.
        nlayers : int, number of layers in the cnn model.
        n_skernels : list of ints, number of kernels for spatial processing.
        n_tkernels : list of ints, number of kernels for temporal processing.
        s_kernelsize : int, size of the spatial kernel.
        t_kernelsize : int, size of the temporal kernel.
        s_stride : int, stride along the spatial dimension.
        t_stride : int, stride along the temporal dimension.
        n_outputs: int, num of coordinates for the regression
        seed : int, for saving random initializations of networks.
        train : bool, is the network meant to be trained or not

        """

        assert (len(n_skernels) == len(n_tkernels) == nlayers), \
            "Number of spatial and temporal processing layers must be equal!"
        if arch_type == 'spatiotemporal':
            n_tkernels = n_skernels
            t_kernelsize = s_kernelsize
            t_stride = s_stride

        self.experiment_id = experiment_id
        self.arch_type = arch_type
        self.nlayers = nlayers
        self.nlayers_fc = nlayers_fc
        self.n_tkernels = n_tkernels
        self.n_skernels = n_skernels
        self.n_tkernels_trans = n_tkernels_trans
        self.n_skernels_trans = n_skernels_trans
        self.t_kernelsize = t_kernelsize
        self.s_kernelsize = s_kernelsize
        self.t_stride = t_stride
        self.s_stride = s_stride
        self.n_outputs = n_outputs
        self.nunits = nunits
        self.seed = seed
        self.task_transfer = task_transfer
        self.decoder = decoder

        # Make model name
        if arch_type == 'spatial_temporal':
            kernels = ('-'.join(str(i) for i in n_skernels)) + '_' + ('-'.join(str(i) for i in n_tkernels))
        elif arch_type == 'temporal_spatial':
            kernels = ('-'.join(str(i) for i in n_tkernels)) + '_' + ('-'.join(str(i) for i in n_skernels))
        else:
            kernels = ('-'.join(str(i) for i in n_skernels))

        parts_name = [arch_type, 'r', str(nlayers), kernels,
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
            batch_size,n_muscles,n_time,n_channels = X.get_shape() #[0]
            # n_muscles = X.get_shape()[1]
            # n_time = X.get_shape()[2]
            # n_channels = X.get_shape()[3]

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
                fully_connected = lambda score, layer_id: slim.fully_connected(
                    score, self.nunits[layer_id], activation_fn=tf.nn.relu, normalizer_fn=slim.layer_norm, scope=f'FC{layer_id}')

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

            # outtime = score.get_shape()[2]
            # score = tf.transpose(score, [0, 2, 1, 3])
            # score = tf.reshape(score, [batch_size, outtime, -1])
            
            # if self.decoder:
                
            #     ## Few FC of nonlinear processing
            #     for layer in range(self.nlayers_fc):
            #         score = fully_connected(score, layer)
            #     net[f'projector{layer}'] = score

            # score = slim.fully_connected(score, self.n_outputs, activation_fn=None, scope='Classifier')
            
            # net['score'] = score

            # print('middle',score.shape)
            if self.decoder:
                
                score = slim.dropout(score, 0.7, is_training=is_training)

                # with slim.arg_scope([slim.conv2d_transpose], data_format='NHWC', normalizer_fn=None): #slim.layer_norm):  #None):
                with slim.arg_scope([slim.conv2d_transpose], data_format='NHWC', normalizer_fn=slim.layer_norm):  #None):
                    ## Deconvolution
                    spatial_convTrans = lambda score, layer_id: slim.conv2d_transpose(
                        score, 
                        
                        self.n_skernels_trans[layer_id],
                        # 2,
                        [self.s_kernelsize, 1], [self.s_stride, 1],
                        scope=f'Spatial_trans{layer_id}')
                    temporal_convTrans = lambda score, layer_id: slim.conv2d_transpose(
                        score, 
                        self.n_tkernels_trans[layer_id], 
                        # 16,
                        [1, self.t_kernelsize], [1, self.t_stride],
                        # padding="VALID",
                        scope=f'Temporal_trans{layer_id}')
                    spatiotemporal_convTrans = lambda score, layer_id: slim.conv2d_transpose(
                        score, self.n_skernels[layer_id], [self.s_kernelsize, self.t_kernelsize], 
                        [self.s_stride, self.t_stride], scope=f'Spatiotemporal_trans{layer_id}')

                    # print(self.nlayers)
                    # print('n t kernel',self.n_tkernels)
                    # print('n s kernel',self.n_skernels)


                    # print('n t kernel_trans',self.n_tkernels_trans)
                    # print('n s kernel_trans',self.n_skernels_trans)

                    # if self.arch_type == 'spatial_temporal':
                    #     for layer in range(self.nlayers-1,-1,-1):
                    #         print('layer:', layer)
                    #         score = temporal_convTrans(score, layer)
                    #         net[f'temporalTrans{layer}'] = score
                    #     for layer in range(self.nlayers-1,-1,-1):
                    #         score = spatial_convTrans(score, layer)
                    #         net[f'spatialTrans{layer}'] = score

                    # elif self.arch_type == 'temporal_spatial':
                    #     for layer in range(self.nlayers-1,-1,-1):
                    #         score = spatial_convTrans(score, layer)
                    #         net[f'spatialTrans{layer}'] = score
                    #     for layer in range(self.nlayers-1,-1,-1):
                    #         score = temporal_convTrans(score, layer)
                    #         net[f'temporalTrans{layer}'] = score
                        

                    # elif self.arch_type == 'spatiotemporal':
                    #     for layer in range(self.nlayers-1,-1,-1):
                    #         score = spatiotemporal_convTrans(score, layer)
                    #         net[f'spatiotemporalTrans{layer}'] = score
                    if self.arch_type == 'spatial_temporal':
                        # for layer in range(self.nlayers-1,-1,-1):
                        for layer in range(self.nlayers):
                            score = temporal_convTrans(score, layer)
                            net[f'temporalTrans{layer}'] = score
                        # for layer in range(self.nlayers-1,1,-1):
                        for layer in range(self.nlayers):
                            score = spatial_convTrans(score, layer)
                            net[f'spatialTrans{layer}'] = score
                        
                        # layer = layer+1
                        # score = slim.conv2d_transpose(score, self.n_skernels_trans[layer] ,[self.s_kernelsize, 1], [1, 1], 
                        #                             activation_fn=None, scope=f'Spatial_trans{layer}') #2
                        # net[f'spatialTrans{layer}'] = score

                    elif self.arch_type == 'temporal_spatial':
                        # for layer in range(self.nlayers-1,-1,-1):
                        for layer in range(self.nlayers):
                            score = spatial_convTrans(score, layer)
                            net[f'spatialTrans{layer}'] = score
                        # for layer in range(self.nlayers-1,1,-1):
                        for layer in range(self.nlayers):
                            score = temporal_convTrans(score, layer)
                            net[f'temporalTrans{layer}'] = score
                        
                        # layer = layer+1
                        # score = slim.conv2d_transpose(score, self.n_tkernels_trans[layer], [1, self.t_kernelsize], [1, 1], #16, padding="VALID",
                        #                             activation_fn=None, scope=f'Temporal_trans{layer}')
                        # net[f'temporalTrans{layer}'] = score

                    elif self.arch_type == 'spatiotemporal':
                        # for layer in range(self.nlayers-1,1,-1):
                        for layer in range(self.nlayers):
                            score = spatiotemporal_convTrans(score, layer)
                            net[f'spatiotemporalTrans{layer}'] = score
                        
                        # layer = layer+1
                        # score = slim.conv2d_transpose(score, self.n_skernels_trans[layer], [self.s_kernelsize, self.t_kernelsize], [1,1],
                        #                             activation_fn=None, scope=f'Spatiotemporal_trans{layer}')
                        # net[f'Spatiotemporal_trans{layer}'] = score
            
            # # print('layer before last',layer)
            # # print('score shape',score.shape)
            # if self.arch_type == 'spatial_temporal':
            #     layer = self.nlayers-1
            #     # score = spatial_convTrans(score, layer)
            #     score = slim.conv2d_transpose(score, self.n_skernels_trans[layer] ,[self.s_kernelsize, 1], [self.s_stride, 1], activation_fn=None, scope=f'Spatial_trans{layer}') #2
            #     net[f'spatialTrans{layer}'] = score

            # elif self.arch_type == 'temporal_spatial':
            #     layer = self.nlayers-1
            #     score = slim.conv2d_transpose(score, self.n_tkernels_trans[layer], [1, self.t_kernelsize], [1, self.t_stride], #16, padding="VALID",
            #             activation_fn=None, scope=f'Temporal_trans{layer}')
            #     net[f'temporalTrans{layer}'] = score
            
            # elif self.arch_type == 'spatiotemporal':
            #     layer = self.nlayers-1
            #     score = slim.conv2d_transpose(score, self.n_skernels_trans[layer], [self.s_kernelsize, self.t_kernelsize], 
            #             [self.s_stride, self.t_stride], activation_fn=None, scope=f'Spatiotemporal_trans{layer}')
            #     net[f'Spatiotemporal_trans{layer}'] = score
                

            # elif self.arch_type == 'spatiotemporal':
            #     for layer in range(self.nlayers-1,-1,-1):
            #         score = spatiotemporal_convTrans(score, layer)
            #         net[f'spatiotemporalTrans{layer}'] = score

            score = score[:,:n_muscles,:n_time,:]

            outtime = score.get_shape()[2]
            score = tf.transpose(score, [0, 2, 1, 3])
            score = tf.reshape(score, [batch_size, outtime, -1])
            score = slim.fully_connected(score, self.n_outputs, activation_fn=None, scope='Classifier')


            # print('almost final shape',score.shape)

            # outtime = score.get_shape()[2]
            # score = tf.transpose(score, [0, 2, 1, 3])
            # score = tf.reshape(score, [batch_size, outtime, -1])
            # score = slim.fully_connected(score, self.n_outputs*2, activation_fn=None, scope='Classifier')

            # print('final shape',score.shape)

            net['score'] = score

        return score, net

class ConvRModel_new():
    """Defines a convolutional neural network model of the proprioceptive system."""

    def __init__(self,
            experiment_id, arch_type, nlayers, n_skernels, n_tkernels, n_skernels_trans, n_tkernels_trans, s_kernelsize,
            t_kernelsize, s_stride, t_stride, s_stride_trans, t_stride_trans, n_outputs, seed=None, decoder=True, train=True,task_transfer=False):  #n_outputs,
        """Set up hyperparameters of the convolutional network.

        Arguments
        ---------
        experiment_id : int, identifier for model path
        arch_type : {'spatial_temporal', 'spatiotemporal', 'temporal_spatial'} str, defines the type
            of convolutional neural network model.
        nlayers : int, number of layers in the cnn model.
        n_skernels : list of ints, number of kernels for spatial processing.
        n_tkernels : list of ints, number of kernels for temporal processing.
        s_kernelsize : int, size of the spatial kernel.
        t_kernelsize : int, size of the temporal kernel.
        s_stride : int, stride along the spatial dimension.
        t_stride : int, stride along the temporal dimension.
        n_outputs: int, num of coordinates for the regression
        seed : int, for saving random initializations of networks.
        train : bool, is the network meant to be trained or not

        """

        assert (len(n_skernels) == len(n_tkernels) == nlayers), \
            "Number of spatial and temporal processing layers must be equal!"
        if arch_type == 'spatiotemporal':
            n_tkernels = n_skernels
            t_kernelsize = s_kernelsize
            # t_stride = s_stride

        self.experiment_id = experiment_id
        self.arch_type = arch_type
        self.nlayers = nlayers
        self.n_tkernels = n_tkernels
        self.n_skernels = n_skernels
        self.t_kernelsize = t_kernelsize
        self.s_kernelsize = s_kernelsize
        self.t_stride = t_stride
        self.s_stride = s_stride
        self.n_outputs = n_outputs
        self.seed = seed
        self.task_transfer = task_transfer
        self.n_tkernels_trans = n_tkernels_trans
        self.n_skernels_trans = n_skernels_trans
        self.decoder = decoder
        self.s_stride_trans = s_stride_trans
        self.t_stride_trans = t_stride_trans

        max_tstride = self.t_stride.count(2)**2
        max_sstride = self.s_stride.count(2)**2

        # Make model name
        if arch_type == 'spatial_temporal':
            kernels = ('-'.join(str(i) for i in n_skernels)) + '_' + ('-'.join(str(i) for i in n_tkernels))
        elif arch_type == 'temporal_spatial':
            kernels = ('-'.join(str(i) for i in n_tkernels)) + '_' + ('-'.join(str(i) for i in n_skernels))
        else:
            kernels = ('-'.join(str(i) for i in n_skernels))

        parts_name = [arch_type, 'r', str(nlayers), kernels,
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
            # batch_size = X.get_shape()[0]
            batch_size,n_muscles,n_time,n_channels = X.get_shape() #[0]

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
            
            # print('middle',score.shape)
            if self.decoder:
                
                score = slim.dropout(score, 0.7, is_training=is_training)

                # with slim.arg_scope([slim.conv2d_transpose], data_format='NHWC', normalizer_fn=None): #slim.layer_norm):  #None):
                with slim.arg_scope([slim.conv2d_transpose], data_format='NHWC', normalizer_fn=slim.layer_norm):  #None):
                    ## Deconvolution
                    spatial_convTrans = lambda score, layer_id: slim.conv2d_transpose(
                        score, 
                        
                        self.n_skernels_trans[layer_id],
                        # 2,
                        [self.s_kernelsize, 1], [self.s_stride_trans[layer_id], 1],
                        scope=f'Spatial_trans{layer_id}')
                    temporal_convTrans = lambda score, layer_id: slim.conv2d_transpose(
                        score, 
                        self.n_tkernels_trans[layer_id], 
                        # 16,
                        [1, self.t_kernelsize], [1, self.t_stride_trans[layer_id]],
                        # padding="VALID",
                        scope=f'Temporal_trans{layer_id}')
                    spatiotemporal_convTrans = lambda score, layer_id: slim.conv2d_transpose(
                        score, self.n_skernels_trans[layer_id], [self.s_kernelsize, self.t_kernelsize], 
                        [self.s_stride_trans[layer_id], self.t_stride_trans[layer_id]], scope=f'Spatiotemporal_trans{layer_id}')

                    # print(self.nlayers)
                    # print('n t kernel',self.n_tkernels)
                    # print('n s kernel',self.n_skernels)


                    # print('n t kernel_trans',self.n_tkernels_trans)
                    # print('n s kernel_trans',self.n_skernels_trans)

                    # if self.arch_type == 'spatial_temporal':
                    #     for layer in range(self.nlayers-1,-1,-1):
                    #         print('layer:', layer)
                    #         score = temporal_convTrans(score, layer)
                    #         net[f'temporalTrans{layer}'] = score
                    #     for layer in range(self.nlayers-1,-1,-1):
                    #         score = spatial_convTrans(score, layer)
                    #         net[f'spatialTrans{layer}'] = score

                    # elif self.arch_type == 'temporal_spatial':
                    #     for layer in range(self.nlayers-1,-1,-1):
                    #         score = spatial_convTrans(score, layer)
                    #         net[f'spatialTrans{layer}'] = score
                    #     for layer in range(self.nlayers-1,-1,-1):
                    #         score = temporal_convTrans(score, layer)
                    #         net[f'temporalTrans{layer}'] = score
                        

                    # elif self.arch_type == 'spatiotemporal':
                    #     for layer in range(self.nlayers-1,-1,-1):
                    #         score = spatiotemporal_convTrans(score, layer)
                    #         net[f'spatiotemporalTrans{layer}'] = score
                    if self.arch_type == 'spatial_temporal':
                        # for layer in range(self.nlayers-1,-1,-1):
                        for layer in range(self.nlayers):
                            score = temporal_convTrans(score, layer)
                            net[f'temporalTrans{layer}'] = score
                        # for layer in range(self.nlayers-1,1,-1):
                        for layer in range(self.nlayers):
                            score = spatial_convTrans(score, layer)
                            net[f'spatialTrans{layer}'] = score

                        # print('layer before last',layer)
                        # print('score shape',score.shape)
                        # layer = layer+1
                        # score = slim.conv2d_transpose(score, self.n_skernels_trans[layer] ,[self.s_kernelsize, 1], [1, 1], 
                        #                             activation_fn=None, scope=f'Spatial_trans{layer}') #2
                        # net[f'spatialTrans{layer}'] = score

                        # print('layer last',layer)
                        # print('score shape',score.shape)

                    elif self.arch_type == 'temporal_spatial':
                        # for layer in range(self.nlayers-1,-1,-1):
                        for layer in range(self.nlayers):
                            score = spatial_convTrans(score, layer)
                            net[f'spatialTrans{layer}'] = score
                        # for layer in range(self.nlayers-1,1,-1):
                        for layer in range(self.nlayers):
                            score = temporal_convTrans(score, layer)
                            net[f'temporalTrans{layer}'] = score
                        
                        # layer = layer+1
                        # score = slim.conv2d_transpose(score, self.n_tkernels_trans[layer], [1, self.t_kernelsize], [1, 1], #16, padding="VALID",
                        #                             activation_fn=None, scope=f'Temporal_trans{layer}')
                        # net[f'temporalTrans{layer}'] = score
                        

                    elif self.arch_type == 'spatiotemporal':
                        # for layer in range(self.nlayers-1,1,-1):
                        for layer in range(self.nlayers):
                            score = spatiotemporal_convTrans(score, layer)
                            net[f'spatiotemporalTrans{layer}'] = score

                        # layer = layer+1
                        # score = slim.conv2d_transpose(score, self.n_skernels_trans[layer], [self.s_kernelsize, self.t_kernelsize], [1,1],
                        #                             activation_fn=None, scope=f'Spatiotemporal_trans{layer}')
                        # net[f'Spatiotemporal_trans{layer}'] = score
            
            # # print('layer before last',layer)
            # # print('score shape',score.shape)
            # if self.arch_type == 'spatial_temporal':
            #     layer = self.nlayers-1
            #     # layer_inv = 0
            #     # print(layer,layer_inv)
            #     # score = spatial_convTrans(score, layer)
            #     score = slim.conv2d_transpose(score, self.n_skernels_trans[layer] ,[self.s_kernelsize, 1], [self.s_stride_trans[layer], 1], activation_fn=None, scope=f'Spatial_trans{layer}') #2
            #     net[f'spatialTrans{layer}'] = score

            # elif self.arch_type == 'temporal_spatial':
            #     layer = self.nlayers-1
            #     layer_inv = 0
            #     score = slim.conv2d_transpose(score, self.n_tkernels_trans[layer], [1, self.t_kernelsize], [1, self.t_stride_trans[layer]], #16, padding="VALID",
            #             activation_fn=None, scope=f'Temporal_trans{layer}')
            #     net[f'temporalTrans{layer}'] = score

            # elif self.arch_type == 'spatiotemporal':
            #     layer = self.nlayers-1
            #     layer_inv = 0
            #     score = slim.conv2d_transpose(score, self.n_tkernels_trans[layer], [self.s_kernelsize, self.t_kernelsize], 
            #             [self.s_stride_trans[layer], self.t_stride_trans[layer]], activation_fn=None, scope=f'Spatiotemporal_trans{layer}')

            #     net[f'spatiotemporalTrans{layer}'] = score

            score = score[:,:n_muscles,:n_time,:]

            outtime = score.get_shape()[2]
            score = tf.transpose(score, [0, 2, 1, 3])
            score = tf.reshape(score, [batch_size, outtime, -1])
            score = slim.fully_connected(score, self.n_outputs, activation_fn=None, scope='Classifier')

            # outtime = score.get_shape()[2]
            # score = tf.transpose(score, [0, 2, 1, 3])
            # score = tf.reshape(score, [batch_size, outtime, -1])
            # score = slim.fully_connected(score, self.n_outputs, activation_fn=None, scope='Classifier')
            
            net['score'] = score

        return score, net

class BarlowTwinsRModel():
    """Defines a convolutional neural network model of the proprioceptive system."""

    def __init__(self,
            experiment_id, n_outputs, arch_type, nlayers, n_skernels, n_tkernels, s_kernelsize,
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
        self.n_outputs = n_outputs
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
        score : tf.tensor [batch_size, n_outputs], computed scores by passing X through the network.
        probabilities : tf.tensor [batch_size, n_outputs], softmax probabilities.
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

                
                outtime = score.get_shape()[2]
                score = tf.transpose(score, [0, 2, 1, 3])
                score = tf.reshape(score, [batch_size, outtime, -1])
                # score = tf.reshape(score, [batch_size, -1])
                
                if self.with_projector:
                    score = slim.dropout(score, 0.7, is_training=is_training)
                    
                    for layer in range(self.nlayers_fc -1):
                        score = fully_connected(score, layer)

                        ### TO REMOVE THIS AFTER (remove also -1)
                        # score = slim.batch_norm(score)
                        # score = tf.nn.relu(score)
                        # net[f'projector{layer}'] = score
                    
                    ## AND ADD THIS
                    layer_id = layer+1
                    score = slim.fully_connected(score, self.nunits[layer_id], activation_fn=None, normalizer_fn=None, scope=f'FC{layer_id}')
                    # score = slim.fully_connected(score, self.nunits[layer_id], None, scope=f'FC{layer_id}')
                    net[f'projector{layer}'] = score

                # score = slim.fully_connected(score, self.nclasses, None, scope='Classifier')

                if self.task_transfer:
                    score = slim.fully_connected(score, self.n_outputs, None, scope='Classifier')
                    net[f'classifier{layer}'] = score


                net['score'] = score

                probabilities = tf.nn.softmax(score, name="Y_proba")

        return score, probabilities, net

class RecurrentRModel():
    """Defines a recurrent neural network model of the proprioceptive system."""

    def __init__(
            self, experiment_id, rec_blocktype, n_recunits, npplayers, nppfilters, 
            s_kernelsize, s_stride, n_outputs, seed=None, train=True, task_transfer=False):  #n_outputs, 
        """Set up the hyperparameters of the recurrent model.

        Arguments
        ---------
        experiment_id : int, identifier for model path
        rec_blocktype: {'lstm', 'gru'} str, type of recurrent block.
        n_recunits : int, number of units in the recurrent block.
        npplayers : int, number of layers in the fully-connected module.
        nppfilters : list of ints, number of filters (spatial convolutions) for spatial processing.
        s_kernelsize : int, size of conv kernel
        s_stride : int, stride for conv kernel
        n_outputs: int, num of coordinates for the regression
        seed : int, for saving random initializations
        train : bool, whether to train the model or not (just save random initialization)

        """

        assert len(nppfilters) == npplayers
        assert rec_blocktype in  ('lstm', 'gru')

        self.experiment_id = experiment_id
        self.rec_blocktype = rec_blocktype
        self.n_recunits = n_recunits
        self.npplayers = npplayers
        self.nppfilters = nppfilters
        self.s_kernelsize = s_kernelsize
        self.s_stride = s_stride
        self.n_outputs = n_outputs  ## Added after to remove in case
        self.seed = seed
        self.task_transfer=task_transfer

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

            spatial_conv = lambda score, layer_id: slim.conv2d(
                score, self.nppfilters[layer_id], [self.s_kernelsize, 1], [self.s_stride, 1],
                data_format='NHWC', normalizer_fn=slim.layer_norm, scope=f'Spatial{layer_id}')

            for layer in range(self.npplayers):
                score = spatial_conv(score, layer)
                net[f'spatial{layer}'] = score

            # `cudnn_rnn` requires the inputs to be of shape [timesteps, batch_size, num_inputs]
            score = tf.transpose(score, [0, 2, 1, 3])
            score = tf.reshape(score, [batch_size, 400, -1])
            score = tf.transpose(score, [1, 0, 2])
            
            if self.rec_blocktype == 'lstm':
                recurrent_cell = cudnn_rnn.CudnnLSTM(1, self.n_recunits, name='RecurrentBlock')
                score, _ = recurrent_cell.apply(score)
            elif self.rec_blocktype == 'gru':
                recurrent_cell = cudnn_rnn.CudnnGRU(1, self.n_recunits, name='RecurrentBlock')
                score, _ = recurrent_cell.apply(score)

            score = tf.transpose(score, [1, 0, 2])
            net['recurrent_out'] = score
            
            score = slim.fully_connected(score, self.n_outputs, activation_fn=None, scope='Classifier')
            net['score'] = score

        return score, net

class RecurrentRModel_new():
    """Defines a recurrent neural network model of the proprioceptive system."""

    def __init__(
            self, experiment_id, rec_blocktype, n_reclayers, n_recunits, npplayers, nppfilters, 
            s_kernelsize, s_stride, n_outputs, seed=None, train=True):  #n_outputs, 
        """Set up the hyperparameters of the recurrent model.

        Arguments
        ---------
        experiment_id : int, identifier for model path
        rec_blocktype: {'lstm', 'gru'} str, type of recurrent block.
        n_recunits : int, number of units in the recurrent block.
        npplayers : int, number of layers in the fully-connected module.
        nppfilters : list of ints, number of filters (spatial convolutions) for spatial processing.
        s_kernelsize : int, size of conv kernel
        s_stride : int, stride for conv kernel
        n_outputs: int, num of coordinates for the regression
        seed : int, for saving random initializations
        train : bool, whether to train the model or not (just save random initialization)

        """

        assert len(nppfilters) == npplayers
        assert rec_blocktype in  ('lstm', 'gru')

        self.experiment_id = experiment_id
        self.rec_blocktype = rec_blocktype
        self.n_reclayers = n_reclayers
        self.n_recunits = n_recunits
        self.npplayers = npplayers
        self.nppfilters = nppfilters
        self.s_kernelsize = s_kernelsize
        self.s_stride = s_stride
        self.n_outputs = n_outputs  ## Added after to remove in case
        self.seed = seed

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

            spatial_conv = lambda score, layer_id: slim.conv2d(
                score, self.nppfilters[layer_id], [self.s_kernelsize, 1], [self.s_stride[layer_id], 1],
                data_format='NHWC', normalizer_fn=slim.layer_norm, scope=f'Spatial{layer_id}')

            for layer in range(self.npplayers):
                score = spatial_conv(score, layer)
                net[f'spatial{layer}'] = score

            # `cudnn_rnn` requires the inputs to be of shape [timesteps, batch_size, num_inputs]
            score = tf.transpose(score, [0, 2, 1, 3])
            score = tf.reshape(score, [batch_size, 400, -1])
            score = tf.transpose(score, [1, 0, 2])
            
            if self.rec_blocktype == 'lstm':
                recurrent_cell = cudnn_rnn.CudnnLSTM(self.n_reclayers, self.n_recunits, name='RecurrentBlock')
                score, _ = recurrent_cell.apply(score)
            elif self.rec_blocktype == 'gru':
                recurrent_cell = cudnn_rnn.CudnnGRU(self.n_reclayers, self.n_recunits, name='RecurrentBlock')
                score, _ = recurrent_cell.apply(score)

            score = tf.transpose(score, [1, 0, 2])
            net['recurrent_out'] = score
            
            score = slim.fully_connected(score, self.n_outputs, activation_fn=None, scope='Classifier')
            net['score'] = score

        return score, net


class BarlowTwinsRModel_new():
    """Defines a convolutional neural network model of the proprioceptive system."""

    def __init__(self,
            experiment_id, n_outputs, arch_type, nlayers, n_skernels, n_tkernels, s_kernelsize,
            t_kernelsize, s_stride, t_stride, nlayers_fc, nunits, seed=None, lambd=5e-3, 
            train=True, with_projector = True, task_transfer=False):
        """Set up hyperparameters of the convolutional network.

        Arguments
        ---------
        experiment_id : int, identifier for model path
        n_outputs : int, number of classes in the classification problem.
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
        self.n_outputs = n_outputs
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

        parts_name = [arch_type, 'r', str(nlayers), kernels,
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
        score : tf.tensor [batch_size, n_outputs], computed scores by passing X through the network.
        probabilities : tf.tensor [batch_size, n_outputs], softmax probabilities.
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

                
                outtime = score.get_shape()[2]
                score = tf.transpose(score, [0, 2, 1, 3])
                score = tf.reshape(score, [batch_size, outtime, -1])
                # score = tf.reshape(score, [batch_size, -1])
                
                if self.with_projector:
                    score = slim.dropout(score, 0.7, is_training=is_training)
                    
                    for layer in range(self.nlayers_fc -1):
                        score = fully_connected(score, layer)

                        ### TO REMOVE THIS AFTER (remove also -1)
                        # score = slim.batch_norm(score)
                        # score = tf.nn.relu(score)
                        # net[f'projector{layer}'] = score
                    
                    ## AND ADD THIS
                    layer_id = layer+1
                    score = slim.fully_connected(score, self.nunits[layer_id], activation_fn=None, normalizer_fn=None, scope=f'FC{layer_id}')
                    # score = slim.fully_connected(score, self.nunits[layer_id], None, scope=f'FC{layer_id}')
                    net[f'projector{layer}'] = score

                # score = slim.fully_connected(score, self.nclasses, None, scope='Classifier')

                if self.task_transfer:
                    score = slim.fully_connected(score, self.n_outputs, None, scope='Classifier')
                    net[f'classifier{layer}'] = score


                net['score'] = score

                probabilities = tf.nn.softmax(score, name="Y_proba")

        return score, probabilities, net


class BarlowTwinsRModel_rec():
    """Defines a recurrent neural network model of the proprioceptive system trained with Barlow Twins objective function."""

    def __init__(self, experiment_id, n_outputs, rec_blocktype, n_recunits, npplayers, nppfilters, 
                s_kernelsize, s_stride, nlayers_fc, nunits, seed=None, 
                lambd=5e-3, train=True, with_projector = True, task_transfer=False):  #n_outputs, 
        """Set up the hyperparameters of the recurrent model.

        Arguments
        ---------
        experiment_id : int, identifier for model path
        rec_blocktype: {'lstm', 'gru'} str, type of recurrent block.
        n_recunits : int, number of units in the recurrent block.
        npplayers : int, number of layers in the fully-connected module.
        nppfilters : list of ints, number of filters (spatial convolutions) for spatial processing.
        s_kernelsize : int, size of conv kernel
        s_stride : int, stride for conv kernel
        n_outputs: int, num of coordinates for the regression
        seed : int, for saving random initializations
        train : bool, whether to train the model or not (just save random initialization)

        """

        assert len(nppfilters) == npplayers
        assert rec_blocktype in  ('lstm', 'gru')

        self.experiment_id = experiment_id
        self.rec_blocktype = rec_blocktype
        self.n_recunits = n_recunits
        self.npplayers = npplayers
        self.nppfilters = nppfilters
        self.s_kernelsize = s_kernelsize
        self.s_stride = s_stride
        self.n_outputs = n_outputs  ## Added after to remove in case
        
        self.nlayers_fc = nlayers_fc
        self.nunits = nunits
        self.seed = seed
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

        net = OrderedDict()

        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE):
            score = X
            batch_size = X.get_shape()[0]

            spatial_conv = lambda score, layer_id: slim.conv2d(
                score, self.nppfilters[layer_id], [self.s_kernelsize, 1], [self.s_stride, 1],
                data_format='NHWC', normalizer_fn=slim.layer_norm, scope=f'Spatial{layer_id}')

            for layer in range(self.npplayers):
                score = spatial_conv(score, layer)
                net[f'spatial{layer}'] = score

            # `cudnn_rnn` requires the inputs to be of shape [timesteps, batch_size, num_inputs]
            score = tf.transpose(score, [0, 2, 1, 3])
            score = tf.reshape(score, [batch_size, 400, -1])
            score = tf.transpose(score, [1, 0, 2])
            
            if self.rec_blocktype == 'lstm':
                recurrent_cell = cudnn_rnn.CudnnLSTM(1, self.n_recunits, name='RecurrentBlock')
                score, _ = recurrent_cell.apply(score)
            elif self.rec_blocktype == 'gru':
                recurrent_cell = cudnn_rnn.CudnnGRU(1, self.n_recunits, name='RecurrentBlock')
                score, _ = recurrent_cell.apply(score)

            score = tf.transpose(score, [1, 0, 2])
            net['recurrent_out'] = score

            if self.with_projector:
                score = slim.dropout(score, 0.7, is_training=is_training)
                
                for layer in range(self.nlayers_fc -1):
                    score = fully_connected(score, layer)

                layer_id = layer+1
                score = slim.fully_connected(score, self.nunits[layer_id], activation_fn=None, normalizer_fn=None, scope=f'FC{layer_id}')
                net[f'projector{layer}'] = score


            if self.task_transfer:
                score = slim.fully_connected(score, self.n_outputs, activation_fn=None, scope='Classifier')
                net[f'classifier{layer}'] = score

            net['score'] = score
            probabilities = tf.nn.softmax(score, name="Y_proba")

        return score, probabilities, net