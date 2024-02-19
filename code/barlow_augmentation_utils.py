"""
UTILS FOR BARLOW TWINS - Augmentations
"""

import os
import copy
import h5py
import yaml
import random

import numpy as np
import tensorflow as tf

def random_noise(data):
    """Add adaptive gaussian noise randomly to 0 - 18 muscles (all timepoints = 400).

    # Arguments
        data: A datapoint [nmuscles, time, 2]

    # Returns
        The augmented data.
    """
    crop = data.copy()
    n_muscle = random.randint(0, 18)
    muscle_idx = random.sample(range(39), k=n_muscle)
    factor = random.sample(list(np.arange(0.1,0.5,0.1)), k=1)
    n_time = 400
    crop[muscle_idx,:] = add_noise(crop[muscle_idx,:],factor)
    return crop

def add_noise(mconf, factor):
    noisy_mconf = mconf + factor*mconf.std(axis=1)[:, None]*np.random.randn(*mconf.shape)
    return noisy_mconf

######## Adapted from https://github.com/sayakpaul/Barlow-Twins-TF ###################

def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)

def off_diagonal(x):
    n = tf.shape(x)[0]
    flattened = tf.reshape(x, [-1])[:-1]
    off_diagonals = tf.reshape(flattened, (n-1, n+1))[:, 1:]
    return tf.reshape(off_diagonals, [-1])

def normalize_repr(z):
    z_norm = (z - tf.reduce_mean(z, axis=0)) / reduce_var(z, axis = 0) #tf.math.reduce_std(z, axis=0)
    return z_norm

def compute_loss(z_a, z_b, lambd):
    # Get batch size and representation dimension.
    batch_size = tf.cast(tf.shape(z_a)[0], z_a.dtype)
    repr_dim = tf.shape(z_a)[1]

    # Normalize the representations along the batch dimension.
    z_a_norm = normalize_repr(z_a)
    z_b_norm = normalize_repr(z_b)

    # Cross-correlation matrix.
    c = tf.matmul(z_a_norm, z_b_norm, transpose_a=True) / batch_size

    # Loss.
    on_diag = tf.linalg.diag_part(c) + (-1)
    on_diag = tf.reduce_sum(tf.pow(on_diag, 2))
    off_diag = off_diagonal(c)
    off_diag = tf.reduce_sum(tf.pow(off_diag, 2))
    loss = on_diag + (lambd * off_diag)
    return loss

#####################################################################################

def random_mask(data, n_muscle=1, n_time=1):
    """Mask randomly from 0 to 18 muscles (all timepoints = 400).

    # Arguments
        data: A datapoint [nmuscles, time, 2]

    # Returns
        The augmented data.
    """
    crop = data.copy() #tf.identity(data) #.copy()
    n_muscle = random.randint(0, 18)
    muscle_idx = random.sample(range(39), k=n_muscle)
    n_time = 400 #tf.random.uniform(shape=(), minval=0, maxval=100, dtype=tf.int32)
    crop[muscle_idx,:] = 0
    return crop

def random_mask_time(data, len_mask=50):
    """Mask randomly 2 windows of 50 timepoints each (all muscles).

    # Arguments
        data: A datapoint [nmuscles, time, 2]

    # Returns
        The augmented data.
    """
    crop = data.copy()
    idx_time = random.randint(0,200)
    idx_time1 = random.randint(200,350)

    crop[:,idx_time:idx_time+len_mask] = 0
    crop[:,idx_time1:idx_time1+len_mask] = 0
    return crop