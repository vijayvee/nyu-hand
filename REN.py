import tensorflow as tf
import numpy as np
from tf_utils import *

class REN:
    def __init__(self,batch_size,nImgs,learning_rate=0.0001,input_size=128,l2_lambda=0.001):
        """
        Implementation of Region Ensemble Network('https://arxiv.org/pdf/1702.02447.pdf') for Hand pose estimation
        :param batch_size: Batch size of the REN model
        :param sequence: Sequence to load, eg. train/val/test
        :param Nmax: Maximum number of images to load
        :param input_size: Size of the input image to REN
        :param n_joints: Number of joints for the network to predict
        :param l2_lambda: Lambda for l2 regularization
        """
        self.batch_size = batch_size
        self.n_joints = 36
        self.batch_shape = [self.batch_size,input_size,input_size,1]
        self.l2_lambda = l2_lambda
        self.learning_rate = learning_rate

    def init_variables(self):
        """
        Function to initialize placeholders for input, ground truth and dropout_prob
        """
        self.imgs = tf.placeholder(tf.float32,self.batch_shape,name='input_imgs')
        self.gt_3d = tf.placeholder(tf.float32,[self.batch_size,self.n_joints*3],name='ground_truth_joints')
        self.dropout_prob = tf.placeholder(tf.float32,name='drouput_keep_probability')
        self.conv_layers,self.fc_layers = [],[]

    def build_model(self):
        """
        Function to build our REN network
        """
        x = self.imgs
        x = conv2d(input=x,k_size=3,output_dim=16,name='conv_1')
        self.conv_layers.append(x)
        x = conv2d(input=x,k_size=3,output_dim=16,name='conv_2')
        self.conv_layers.append(x)
        x = max_pool(input=x,name='max_pool_1')
        pool1 = conv2d(input=x,k_size=1,output_dim=32,name='residual_1')
        self.conv_layers.append(x)
        x = conv2d(input=x,k_size=3,output_dim=32,name='conv_3')
        self.conv_layers.append(x)
        x = conv2d(input=x,k_size=3,output_dim=32,name='conv_4') + pool1
        self.conv_layers.append(x)
        x = max_pool(input=x,name='max_pool2')
        pool2 = conv2d(input=x,k_size=1,output_dim=64,name='residual_2')
        self.conv_layers.append(x)
        x = conv2d(input=x,k_size=3,output_dim=64,name='conv_5')
        self.conv_layers.append(x)
        x = conv2d(input=x,k_size=3,output_dim=64,name='conv_6') + pool2
        self.conv_layers.append(x)
        x = max_pool(input=x,name='max_pool3')
        self.conv_layers.append(x)
        top_l,top_r,bot_l,bot_r = x[:,0:6,0:6,:],x[:,0:6,6:,:],x[:,6:,0:6,:],x[:,6:,6:,:]
        print type(top_l),top_l
        fc1 = Linear(top_l,2048,'fc1',True,self.dropout_prob)
        fc2 = Linear(top_l,2048,'fc2',True,self.dropout_prob)
        fc3 = Linear(top_l,2048,'fc3',True,self.dropout_prob)
        fc4 = Linear(top_l,2048,'fc4',True,self.dropout_prob)
        self.fc_layers.extend((fc1,fc2,fc3,fc4))
        fc_final_inp = tf.concat([fc1,fc2,fc3,fc4],axis=1)
        self.fc_final = Linear(fc_final_inp,self.n_joints*3,'output_regressor',True,self.dropout_prob,act='tanh')
        self.fc_layers.append(self.fc_final)
        print "Built model"
        return self.conv_layers,self.fc_layers

    def print_all_layers(self):
        """
        Function to print the name and shape of all layers in REN
        """
        for layer in self.conv_layers+self.fc_layers:
            print "Name: {}, Shape: {}".format(layer.name,layer.get_shape().as_list())

    def get_loss(self,fc_final):
        """
        Function to create the loss op for REN.
        """
        assert self.gt_3d.get_shape().as_list() == fc_final.get_shape().as_list()
        regr_loss = tf.reduce_mean(tf.nn.l2_loss(self.gt_3d-fc_final))
        l2_reg = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        self.loss = self.l2_lambda*l2_reg + (1-self.l2_lambda)*regr_loss
        print "Created regression loss for REN"
        return self.loss

    def get_optimizer(self):
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_step = opt.minimize(self.loss)
        return opt,train_step
