from Importer import NYUImporter
from dataset import NYUDataset
import numpy as np

class DataLoader:
    #Class file for using Importer and Dataset to fetch batches of data
    def __init__(self,importer_path,useCache=True,cacheDir='/media/data_cifs/lu/cache',sequence,Nmax,batch_size):
        """Constructor for the DataLoader class.
        :param importer_path: Path to the input dataset
        :param useCache: Flag to use cached data from a pickle file
        :param cacheDir: Directory where the cached pickle file is stored (Only if useCache is set)
        :param batch_size: Batch size to fetch data
        :param sequence: Sequence to load, eg. train/val/test
        :param Nmax: Maximum number of images to load
        """
        self.batch_size = batch_size
        self.importer = NYUImporter(importer_path,useCache=useCache,cacheDir=cacheDir)
        self.sequence = sequence
        self.imgSeq = importer.loadSequence(sequence,Nmax=Nmax,allJoints=True)
        self.dataset = NYUDataset(imgSeqs=[self.imgSeq])
        self.data, self.labels, seqconfig, train_com3D,train_M = dataset.imgStackDepthOnly(sequence)
        self.data = self.data.squeeze(1)
        self.nImgs = len(self.data)

    def fetch_data_batch(self):
        idxs = 
        return self.data[idxs],self.labels[idxs]
