import tensorflow as tf
import numpy as np

def weight_bias_variable(input_shape,output_dim,name):
    """
    Function to return a tf Variable
    :param shape: Shape of the new variable to create
    :param name: Name of the new variable to create
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

def conv2d(input,k_size,output_dim,name):
    with tf.variable_scope(name):
        W,b = weight_bias_variable(input.get_shape().as_list(),output_dim,name)
        x = tf.nn.conv2d(input,W,strides=[1,1,1,1],padding='SAME')
        out = tf.nn.relu(tf.nn.bias_add(x,b))
        return out
