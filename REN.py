import tensorflow as tf
import numpy as np
from Importer import NYUImporter
from dataset import NYUDataset
from utils import *

class REN:
    def __init__(self,importer_path,useCache=True,sequence,Nmax,batch_size,input_size=128,cacheDir='/media/data_cifs/lu/cache'):
        """
        Implementation of Region Ensemble Network('https://arxiv.org/pdf/1702.02447.pdf') for Hand pose estimation
        :param importer_path: Path to the input dataset
        :param useCache: Flag to use cached data from a pickle file
        :param cacheDir: Directory where the cached pickle file is stored (Only if useCache is set)
        :param batch_size: Batch size of the REN model
        :param sequence: Sequence to load, eg. train/val/test
        :param Nmax: Maximum number of images to load
        :param input_size: Size of the input image to REN
        :param n_joints: Number of joints for the network to predict
        """
        self.batch_size = batch_size
        self.importer = NYUImporter(importer_path,useCache=useCache,cacheDir=cacheDir)
        self.sequence = sequence
        self.imgSeq = importer.loadSequence(sequence,Nmax=Nmax,allJoints=True)
        self.n_joints = 36
        self.dataset = NYUDataset(imgSeqs=[self.imgSeq])
        self.data, self.labels, seqconfig, train_com3D,train_M = dataset.imgStackDepthOnly(sequence)
        self.batch_shape = [self.batch_size,input_size,input_size,1]
        self.data = self.data.squeeze(1)
        self.nImgs = len(self.data)

    def fetch_data_batch(self):
        """
        Function to fetch a data batch of depth images and the groung truth (joint locations) in xyz
        """
        idxs = np.random.choice(self.nImgs,self.batch_size)
        curr_batch_dpt = self.data[idxs]
        curr_batch_gt = self.labels[idxs]
        return curr_batch_dpt,curr_batch_gt

    def init_variables(self):
        """
        Function to initialize placeholders for input, ground truth and dropout_prob
        """
        self.imgs = tf.placeholder(tf.float32,self.batch_shape,name='input_imgs')
        self.gt_3d = tf.placeholder(tf.float32,[self.batch_size,self.n_joints,3],name='ground_truth_joints')
        self.dropout_prob = tf.placeholder(tf.float32,name='drouput_keep_probability')

    def build_model(self):
        x = self.imgs
        x = conv2d(input=x,k_size=3,output_dim=16,name='conv_1')
        x = conv2d(input=x,k_size=3,output_dim=16,name='conv_2')
        x = pool2d(x)
        x = conv2d(input=x,k_size=3,output_dim=16,name='conv_')
        x = conv2d(input=x,k_size=3,output_dim=16,name='conv_1')
