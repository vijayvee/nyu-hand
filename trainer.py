import tensorflow as tf
import numpy as np
from REN import REN
from Importer import NYUImporter
from dataset import NYUDataset

#Script to train REN for hand pose estimation
dataset_path = '/media/data_cifs/lu/NYU/dataset/' #Path where dataset splits train, test and val are stored
cache_path = '/media/data_cifs/lu/cache' #Directory where cache files are stored while reading the dataset
useCache = True
Nmax = 40000
sequence = 'train'
print_loss_every = 100
save_model_every = 5000
imp = NYUImporter(dataset_path,useCache=useCache,cacheDir=cache_path)
trainSeq = imp.loadSequence(sequence,Nmax=Nmax,allJoints=True) #Sequence of training images
dataset = NYUDataset([trainSeq])
train_data, train_gt3D, seqconfig, train_com3D,train_M = dataset.imgStackDepthOnly(sequence)

def fetch_curr_batch(batch_size):
    """
    Function to yield a batch of depth images and labels
    :param batch_size: Yield batch_size number of images and labels
    """
    idxs = np.random.choice(Nmax,batch_size)
    yield train_data[idxs],train_gt3D[idxs]


def run(batch_size,Nmax,l2_lambda,n_epochs,learning_rate,init_vars=True):
        ren = REN(batch_size,Nmax,l2_lambda=l2_lambda)
        ren.init_variables()
        ren.build_model()
        ren.print_all_layers()
    loss = ren.get_loss(ren.fc_final)
    opt,train_step = ren.get_optimizer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    saver = tf.train.Saver()
    with tf.Session() as sess:
#        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        if init_vars:
            sess.run(tf.global_variables_initializer())
            print "Initialized variables"
        else:
            saver_ = tf.train.Saver()
            saver_.restore(sess,tf.train.latest_checkpoint('ckpt/'))
            print "Restored variables"
        nIter = int(Nmax*n_epochs/batch_size)
        for i in xrange(nIter):
            gen = fetch_curr_batch(batch_size)
            curr_imgs,curr_gt = gen.next()
            #import ipdb; ipdb.set_trace()
            curr_gt = curr_gt.reshape([batch_size,-1])
            curr_imgs = curr_imgs.transpose(0,2,3,1)
            curr_loss,_ = sess.run([loss,train_step],feed_dict={ren.imgs:curr_imgs,ren.gt_3d:curr_gt,ren.dropout_prob:0.6})
            if i%print_loss_every == 0:
                print "Iter-{} loss: {}".format(i,curr_loss)
            if i%save_model_every == 0:
                print "Saving iter-{} nmax- {} l2- {} epochs- {} lr- {}".format(i,Nmax,l2_lambda,n_epochs,learning_rate)
                saver.save(sess,'ckpt/NYU_Hand_{}_{}_{}_{}_{}.ckpt'.format(i,Nmax,l2_lambda,n_epochs,learning_rate),global_step=i)
            if i==nIter-1:
                print "Training complete..\nSaving iter-{} nmax- {} l2- {} epochs- {} lr- {}".format(i,Nmax,l2_lambda,n_epochs,learning_rate)
                saver.save(sess,'ckpt/NYU_Hand_{}_{}_{}_{}_Final.ckpt'.format(Nmax,l2_lambda,n_epochs,learning_rate))

def main():
    run(128,Nmax,0.0005,30,0.005,init_vars=True)

if __name__=="__main__":
    main()
