"""Provides Dataset class for handling datasets.

Dataset provides interface for managing data, eg normalization, batch building.
ICVLDataset, NYUDataset, MSRADataset are specific instances of different datasets.

Copyright 2015 Markus Oberweger, ICG,
Graz University of Technology <oberweger@icg.tugraz.at>

This file is part of DeepPrior.

DeepPrior is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

DeepPrior is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with DeepPrior.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy
import transformations
from basetype import NamedImgSequence
from Importer import NYUImporter
from handdetector import HandDetector
from check_fun import showdepth,showImageLable,trans3DsToImg,showImagefromArray,showImageJoints



__author__ = "Paul Wohlhart <wohlhart@icg.tugraz.at>, Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Paul Wohlhart", "Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class Dataset(object):
    """
    Base class for managing data. Used to create training batches.
    """

    def __init__(self,imgSeqs=None,val_prop=None):
        """
        Constructor
        """
        if imgSeqs is None:
            self._imgSeqs = []
        else:
            self._imgSeqs = imgSeqs
        print('val_prop is {}'.format(val_prop))
        self._val_prop = val_prop
        self._imgStacks = {}
        self._labelStacks = {}
        if self._val_prop is not None:
            self._imgStacks_v = {}
            self._labelStacks_v = {}

    @property
    def imgSeqs(self):
        return self._imgSeqs

    def imgSeq(self, seqName):
        for seq in self._imgSeqs:
            if seq.name == seqName:
                return seq
        return []

    @imgSeqs.setter
    def imgSeqs(self, value):
        self._imgSeqs = value
        self._imgStacks = {}

    def load(self):
        objNames = self.lmi.getObjectNames()
        imgSeqs = self.lmi.loadSequences(objNames)
        raise NotImplementedError("Not implemented!")

    def check(self):
        print('val_prop: {}'.format(self._val_prop))

    def imgStackDepthOnly(self, seqName, normZeroOne=False):
        imgSeq = None
        for seq in self._imgSeqs:
            if seq.name == seqName:
                imgSeq = seq
                break
        if imgSeq is None:
            return []


        if self._val_prop == None:
            if seqName not in self._imgStacks:
                # compute the stack from the sequence
                numImgs = len(imgSeq.data)
                data0 = numpy.asarray(imgSeq.data[0].dpt, 'float32')
                label0 = numpy.asarray(imgSeq.data[0].gtorig, 'float32')
                h, w = data0.shape
                j, d = label0.shape
                imgStack = numpy.zeros((numImgs, 1, h, w), dtype='float32')  # num_imgs,stack_size,rows,cols
                labelStack = numpy.zeros((numImgs, j, d), dtype='float32')  # num_imgs,joints,dim
                com3DStack = numpy.zeros((numImgs,3),dtype='float32')
                MStack = numpy.zeros((numImgs,3,3),dtype='float32')
                for i in xrange(numImgs):
                    if normZeroOne:
                        imgD = numpy.asarray(imgSeq.data[i].dpt.copy(), 'float32')
                        imgD[imgD == 0] = imgSeq.data[i].com[2] + (imgSeq.config['cube'][2] / 2.)
                        imgD -= (imgSeq.data[i].com[2] - (imgSeq.config['cube'][2] / 2.))
                        imgD /= imgSeq.config['cube'][2]
                    else:
                        imgD = numpy.asarray(imgSeq.data[i].dpt.copy(), 'float32')
                        imgD[imgD == 0] = imgSeq.data[i].com[2] + (imgSeq.config['cube'][2] / 2.)
                        imgD -= imgSeq.data[i].com[2]
                        imgD /= (imgSeq.config['cube'][2] / 2.)

                    imgStack[i] = imgD

                    labelStack[i] = numpy.clip(numpy.asarray(imgSeq.data[i].gt3Dcrop, dtype='float32') / (imgSeq.config['cube'][2] / 2.), -1, 1)
                    com3DStack[i] = imgSeq.data[i].com
                    MStack[i] = imgSeq.data[i].T

                self._imgStacks[seqName] = imgStack
                self._labelStacks[seqName] = labelStack

            return self._imgStacks[seqName], self._labelStacks[seqName], imgSeq.config,com3DStack,MStack
        else:
            if seqName not in self._imgStacks:
                # compute the stack from the sequence
                numImgs = len(imgSeq.data)
                data0 = numpy.asarray(imgSeq.data[0].dpt, 'float32')
                label0 = numpy.asarray(imgSeq.data[0].gtorig, 'float32')
                h, w = data0.shape
                j, d = label0.shape

                # #check image and label (dpt, gtcrop)
                # showImagefromArray(imgSeq.data[56].dpt,imgSeq.data[56].gtcrop)
                # showImageJoints(imgSeq.data[2].dpt,imgSeq.data[2].gtcrop)
                # # check image and label (dpt, gt3Dcrop --> gtcroptest)
                # for n in range (numImgs):
                #     if (n >350) and (n<360):
                #         gtcrop_test = trans3DsToImg(imgSeq.data[n].gt3Dcrop, imgSeq.data[n].com, imgSeq.data[n].T)
                #         showImageLable(imgSeq.data[n].dpt, gtcrop_test)


                #split into training and validation
                train_cut = int((1 - self._val_prop) * numImgs)
                imgStack = numpy.zeros((train_cut, 1, h, w), dtype='float32')  # train_cut,stack_size,rows,cols
                labelStack = numpy.zeros((train_cut, j, d), dtype='float32')  # train_cut,joints,dim
                com3DStack = numpy.zeros((train_cut,3),dtype='float32')
                MStack = numpy.zeros((train_cut,3,3),dtype='float32')

                imgStack_v = numpy.zeros((numImgs - train_cut, 1, h, w), dtype='float32')
                lableStack_v =  numpy.zeros((numImgs-train_cut, j, d), dtype='float32')
                com3DStack_v = numpy.zeros((numImgs-train_cut,3),dtype='float32')
                MStack_v = numpy.zeros((numImgs-train_cut,3,3),dtype='float32')

                for i in xrange(numImgs):
                    if normZeroOne:
                        imgD = numpy.asarray(imgSeq.data[i].dpt.copy(), 'float32')
                        imgD[imgD == 0] = imgSeq.data[i].com[2] + (imgSeq.config['cube'][2] / 2.)
                        imgD -= (imgSeq.data[i].com[2] - (imgSeq.config['cube'][2] / 2.))
                        imgD /= imgSeq.config['cube'][2]
                    else:
                        imgD = numpy.asarray(imgSeq.data[i].dpt.copy(), 'float32')
                        imgD[imgD == 0] = imgSeq.data[i].com[2] + (imgSeq.config['cube'][2] / 2.)# assign the depth which is not available to be 1
                        imgD -= imgSeq.data[i].com[2]
                        imgD /= (imgSeq.config['cube'][2] / 2.)
                    #showImageLable(imgD,imgSeq.data[i].gtcrop)


                    if i < train_cut:
                        imgStack[i] = imgD
                        labelStack[i] = numpy.clip(numpy.asarray(imgSeq.data[i].gt3Dcrop, dtype='float32') / (imgSeq.config['cube'][2] / 2.), -1, 1)
                        com3DStack[i] = imgSeq.data[i].com
                        MStack[i] = imgSeq.data[i].T

                        # # check image and label (gt3Dcrop=labelstack*cube)
                        # if (i ==50):
                        #     print("cube_22:{}".format(imgSeq.config['cube'][2]/2.))
                        #     print("data gt3Dcrop:{}".format(imgSeq.data[i].gt3Dcrop))
                        #     print("labelStack:{}".format(labelStack[i]))
                        #     print("labelStack * cube22:{}".format(labelStack[i]*imgSeq.config['cube'][2]/2.))
                        #     gtcrop_test = trans3DsToImg(labelStack[i]*imgSeq.config['cube'][2]/2.,com3DStack[i],MStack[i])
                        #     showImageLable(imgD, gtcrop_test)

                    else:
                        imgStack_v[i-train_cut] = imgD
                        lableStack_v[i-train_cut] = numpy.clip(numpy.asarray(imgSeq.data[i].gt3Dcrop, dtype='float32') / (imgSeq.config['cube'][2] / 2.), -1, 1)
                        com3DStack_v[i-train_cut] = imgSeq.data[i].com
                        MStack_v[i-train_cut] = imgSeq.data[i].T

                self._imgStacks[seqName] = imgStack
                self._labelStacks[seqName] = labelStack
                self._imgStacks_v[seqName] = imgStack_v
                self._labelStacks_v[seqName] = lableStack_v

            return self._imgStacks[seqName], self._labelStacks[seqName], self._imgStacks_v[seqName], self._labelStacks_v[seqName],imgSeq.config,com3DStack,com3DStack_v,MStack,MStack_v






class NYUDataset(Dataset):
    def __init__(self, imgSeqs=None,basepath=None,val_prop=None):
        """
        constructor
        """
        super(NYUDataset, self).__init__(imgSeqs,val_prop)
        if basepath is None:
            basepath = '../../data/NYU/'

        self.lmi = NYUImporter(basepath)

