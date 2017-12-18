import tensorflow as tf
import numpy as np

def weight_bias_variable(input_shape,output_dim,name):
    """
    Function to return tf variables for kernel and bias
    :param input_shape: Shape of the input for convolution
    :param name: Name of the conv layer
    """
    n,h,w,c = input_shape
    W = tf.get_variable(dtype=tf.float32,
                    name=name+'_W',
                    shape=[h,w,c,output_dim],
                    initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b = tf.get_variable(dtype=tf.float32,
                    name=name+'_b',
                    shape=output_dim,
                    initializer=tf.contrib.layers.xavier_initializer())
    return W,b

def weight_bias_variable_linear(input_shape,output_dim,name):
    """
    Function to return a tf variables for weight and bias
    :param input_shape: Shape of the new variable to create
    :param name: Name of the new variable to create
    """
    a,input_dim = input_shape
    W = tf.get_variable(dtype=tf.float32,
                    name=name+'_W',
                    shape=[input_dim,output_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(dtype=tf.float32,
                    name=name+'_b',
                    shape=output_dim,
                    initializer=tf.contrib.layers.xavier_initializer())
    return W,b

def conv2d(input,k_size,output_dim,name,is_training=True):
    """
    Function to perform a convolution operation on 'input'.
    :param input: Input to perform conv on
    :param k_size: Size of conv kernel
    :param output_dim: Number of conv kernels to be used
    :param name: Name of the current conv layer
    :param is_training: Training flag for batch normalization
    """
    with tf.variable_scope(name):
        W,b = weight_bias_variable(input.get_shape().as_list(),output_dim,name)
        x = tf.nn.conv2d(input,W,strides=[1,1,1,1],padding='SAME')
        out = tf.nn.bias_add(x,b)
        if is_training:
            out = tf.contrib.layers.batch_norm(out,is_training=is_training)
        return tf.nn.relu(out)

def max_pool(input,name):
    """
    Function to perform a max pooling operation on 'input'.
    :param input: Input to perform max pooling on
    :param name: Name of the current max pool layer
    """
    with tf.variable_scope(name):
        out = tf.nn.max_pool(input,ksize=[1,2,2,1],padding='SAME',strides=[1,2,2,1])
        return out

def Linear(input,output_dim,name,dropout=False,dropout_prob=None,act='relu',is_training=True):
    """
    Function to perform a linear transformation of 'input'.
    :param input: Input to perform linear transformation on.
    :param output_dim: Dimension of output
    :param name: Name of the linear layer.
    """
    if len(input.get_shape())!=2:
        n = input.get_shape().as_list()[0]
        input = tf.reshape(input,[n,-1])
    with tf.variable_scope(name):
        W,b = weight_bias_variable_linear(input.get_shape().as_list(),output_dim,name)
        out = tf.nn.xw_plus_b(input,W,b)
        if is_training:
            out = tf.contrib.layers.batch_norm(out,is_training=is_training)
        if act == 'relu':
            out = tf.nn.relu(out)
        else:
            out = tf.nn.tanh(out)
        if dropout:
            out = tf.nn.dropout(out,dropout_prob)
        return out
