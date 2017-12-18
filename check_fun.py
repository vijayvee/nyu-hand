import numpy as np
import  matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def joint3DToImg(sample):
    fx = 588.03
    fy = 587.07
    ux = 320.
    uy = 240.
    ret = np.zeros((3,),np.float32)
    # convert to metric using f, see Thomson et.al.
    if sample[2] == 0.:
        ret[0] = ux
        ret[1] = uy
        return ret
    ret[0] = sample[0] / sample[2] * fx + ux
    ret[1] = uy - sample[1] / sample[2] * fy
    ret[2] = sample[2]
    return ret

def joints3DToImg(sample):
    ret = np.zeros((sample.shape[0],3),np.float32)
    for i in range(sample.shape[0]):
        ret[i] = joint3DToImg(sample[i])
    return  ret

def transformPoint2D(pt, M):

    pt2 = np.asmatrix(M.reshape((3, 3))) * np.matrix([pt[0], pt[1], 1]).T
    return np.array([pt2[0] / pt2[2], pt2[1] / pt2[2]])

def trans3DToImg(gt3Dcrop,com3D,M):
    #transform gt3Dcrop[i] to gtcrop[i]
    #print gt3Dcrop[0]
    gt3Dcrop_trans = joint3DToImg(gt3Dcrop + com3D)
    t_test = transformPoint2D(gt3Dcrop_trans,M)
    gtcrop_test = np.zeros(3,np.float32)
    gtcrop_test[0]=t_test[0]
    gtcrop_test[1]=t_test[1]
    gtcrop_test[2]=gt3Dcrop_trans[2]
    return gtcrop_test # = gtcrop


def trans3DsToImg(gt3Dcrop,com3D,M):
    ret = np.zeros((gt3Dcrop.shape[0],3),np.float32)
    for i in range(gt3Dcrop.shape[0]):
        ret[i] = trans3DToImg(gt3Dcrop[i],com3D,M)
    return ret


def showAnnotatedDepth(frame):
    """
    Show the depth image
    :param frame: image to show
    :return:
    """
    print("img min {}, max {}".format(frame.dpt.min(), frame.dpt.max()))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(frame.dpt, cmap=matplotlib.cm.jet, interpolation='nearest')
    ax.scatter(frame.gtcrop[:, 0], frame.gtcrop[:, 1])

    ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[1::-1, 0])),
            np.hstack((frame.gtcrop[13, 1], frame.gtcrop[1::-1, 1])), c='y')
    ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[3:1:-1, 0])),
            np.hstack((frame.gtcrop[13, 1], frame.gtcrop[3:1:-1, 1])), c='y')
    ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[5:3:-1, 0])),
            np.hstack((frame.gtcrop[13, 1], frame.gtcrop[5:3:-1, 1])), c='y')
    ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[7:5:-1, 0])),
            np.hstack((frame.gtcrop[13, 1], frame.gtcrop[7:5:-1, 1])), c='y')
    ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[10:7:-1, 0])),
            np.hstack((frame.gtcrop[13, 1], frame.gtcrop[10:7:-1, 1])), c='y')
    ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[11, 0])),
            np.hstack((frame.gtcrop[13, 1], frame.gtcrop[11, 1])), c='y')
    ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[12, 0])),
            np.hstack((frame.gtcrop[13, 1], frame.gtcrop[12, 1])), c='y')

    def format_coord(x, y):
        numrows, numcols = frame.dpt.shape
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = frame.dpt[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)

    ax.format_coord = format_coord
    plt.show()

def showImageLable(dpt,lable):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print("img min {}, max {}".format(dpt.min(), dpt.max()))
    ax.imshow(dpt, cmap=matplotlib.cm.jet, interpolation='nearest')
    ax.scatter(lable[:, 0], lable[:, 1])

    # ax.plot(np.hstack((lable[13, 0], lable[1::-1, 0])),
    #         np.hstack((lable[13, 1], lable[1::-1, 1])), c='y')
    # ax.plot(np.hstack((lable[13, 0], lable[3:1:-1, 0])),
    #         np.hstack((lable[13, 1], lable[3:1:-1, 1])), c='y')
    # ax.plot(np.hstack((lable[13, 0], lable[5:3:-1, 0])),
    #         np.hstack((lable[13, 1], lable[5:3:-1, 1])), c='y')
    # ax.plot(np.hstack((lable[13, 0], lable[7:5:-1, 0])),
    #         np.hstack((lable[13, 1], lable[7:5:-1, 1])), c='y')
    # ax.plot(np.hstack((lable[13, 0], lable[10:7:-1, 0])),
    #         np.hstack((lable[13, 1], lable[10:7:-1, 1])), c='y')
    # ax.plot(np.hstack((lable[13, 0], lable[11, 0])),
    #         np.hstack((lable[13, 1], lable[11, 1])), c='y')
    # ax.plot(np.hstack((lable[13, 0], lable[12, 0])),
    #         np.hstack((lable[13, 1], lable[12, 1])), c='y')

    def format_coord(x, y):
        numrows, numcols = dpt.shape
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = dpt[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)

    ax.format_coord = format_coord
    plt.show()


def showdepth(depth):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(depth, cmap=matplotlib.cm.jet, interpolation='nearest')

def showImagefromArray(img,labels):
    #plt.imshow(img,aspect="auto")
    #plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img,aspect="auto")
    ax.scatter(labels[:, 0], labels[:, 1])
    plt.show()


def showImageLableCom(dpt,lable,com):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print("img min {}, max {}".format(dpt.min(), dpt.max()))
    ax.imshow(dpt, cmap=matplotlib.cm.jet, interpolation='nearest')
    ax.scatter(lable[:, 0], lable[:, 1])
    ax.scatter(com[0], com[1],s= 100 ,c='g' )

    # ax.plot(np.hstack((lable[13, 0], lable[1::-1, 0])),
    #         np.hstack((lable[13, 1], lable[1::-1, 1])), c='y')
    # ax.plot(np.hstack((lable[13, 0], lable[3:1:-1, 0])),
    #         np.hstack((lable[13, 1], lable[3:1:-1, 1])), c='y')
    # ax.plot(np.hstack((lable[13, 0], lable[5:3:-1, 0])),
    #         np.hstack((lable[13, 1], lable[5:3:-1, 1])), c='y')
    # ax.plot(np.hstack((lable[13, 0], lable[7:5:-1, 0])),
    #         np.hstack((lable[13, 1], lable[7:5:-1, 1])), c='y')
    # ax.plot(np.hstack((lable[13, 0], lable[10:7:-1, 0])),
    #         np.hstack((lable[13, 1], lable[10:7:-1, 1])), c='y')
    # ax.plot(np.hstack((lable[13, 0], lable[11, 0])),
    #         np.hstack((lable[13, 1], lable[11, 1])), c='y')
    # ax.plot(np.hstack((lable[13, 0], lable[12, 0])),
    #         np.hstack((lable[13, 1], lable[12, 1])), c='y')

    def format_coord(x, y):
        numrows, numcols = dpt.shape
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = dpt[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)

    ax.format_coord = format_coord
    plt.show()


def showImageJoints(dpt,lable,save=False,imagename=None,allJoints=False,line=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #print("img min {}, max {}".format(dpt.min(), dpt.max()))
    ax.imshow(dpt, cmap=matplotlib.cm.jet, interpolation='nearest')
    ax.scatter(lable[:, 0], lable[:, 1],c='r')

    if line==True:
        if allJoints:
            ax.plot([lable[0,0],lable[1,0]],[lable[0,1],lable[1,1]],c='r')
            ax.plot([lable[1, 0], lable[2, 0]], [lable[1, 1], lable[2, 1]], c='r')
            ax.plot([lable[2, 0], lable[3, 0]], [lable[2, 1], lable[3, 1]], c='r')
            ax.plot([lable[3, 0], lable[4, 0]], [lable[3, 1], lable[4, 1]], c='r')
            ax.plot([lable[6, 0], lable[7, 0]], [lable[6, 1], lable[7, 1]], c='g')
            ax.plot([lable[7, 0], lable[8, 0]], [lable[7, 1], lable[8, 1]], c='g')
            ax.plot([lable[8, 0], lable[9, 0]], [lable[8, 1], lable[9, 1]], c='g')
            ax.plot([lable[9, 0], lable[10, 0]], [lable[9, 1], lable[10, 1]], c='g')
            ax.plot([lable[12, 0], lable[13, 0]], [lable[12, 1], lable[13, 1]], c='k')
            ax.plot([lable[13, 0], lable[14, 0]], [lable[13, 1], lable[14, 1]], c='k')
            ax.plot([lable[14, 0], lable[15, 0]], [lable[14, 1], lable[15, 1]], c='k')
            ax.plot([lable[15, 0], lable[16, 0]], [lable[15, 1], lable[16, 1]], c='k')
            ax.plot([lable[18, 0], lable[19, 0]], [lable[18, 1], lable[19, 1]], c='w')
            ax.plot([lable[19, 0], lable[20, 0]], [lable[19, 1], lable[20, 1]], c='w')
            ax.plot([lable[20, 0], lable[21, 0]], [lable[20, 1], lable[21, 1]], c='w')
            ax.plot([lable[21, 0], lable[22, 0]], [lable[21, 1], lable[22, 1]], c='w')
            ax.plot([lable[4, 0], lable[5, 0]], [lable[4, 1], lable[5, 1]], c='r')
            ax.plot([lable[10, 0], lable[11, 0]], [lable[10, 1], lable[11, 1]], c='g')
            ax.plot([lable[16, 0], lable[17, 0]], [lable[16, 1], lable[17, 1]], c='k')
            ax.plot([lable[22, 0], lable[23, 0]], [lable[22, 1], lable[23, 1]], c='w')
            ax.plot([lable[5, 0], lable[32, 0]], [lable[5, 1], lable[32, 1]], c='y')
            ax.plot([lable[11, 0], lable[32, 0]], [lable[11, 1], lable[32, 1]], c='y')
            ax.plot([lable[17, 0], lable[32, 0]], [lable[17, 1], lable[32, 1]], c='y')
            ax.plot([lable[23, 0], lable[32, 0]], [lable[23, 1], lable[32, 1]], c='y')
            ax.plot([lable[32, 0], lable[30, 0]], [lable[32, 1], lable[30, 1]], c='y')
            ax.plot([lable[32, 0], lable[31, 0]], [lable[32, 1], lable[31, 1]], c='y')
            ax.plot([lable[32, 0], lable[28, 0]], [lable[32, 1], lable[28, 1]], c='y')
            ax.plot([lable[28, 0], lable[27, 0]], [lable[28, 1], lable[27, 1]], c='m')
            ax.plot([lable[27, 0], lable[26, 0]], [lable[27, 1], lable[26, 1]], c='m')
            ax.plot([lable[26, 0], lable[25, 0]], [lable[26, 1], lable[25, 1]], c='m')
            ax.plot([lable[25, 0], lable[24, 0]], [lable[25, 1], lable[24, 1]], c='m')
        else:

            ax.plot(np.hstack((lable[13, 0], lable[1::-1, 0])),
                    np.hstack((lable[13, 1], lable[1::-1, 1])), c='y')
            ax.plot(np.hstack((lable[13, 0], lable[3:1:-1, 0])),
                    np.hstack((lable[13, 1], lable[3:1:-1, 1])), c='y')
            ax.plot(np.hstack((lable[13, 0], lable[5:3:-1, 0])),
                    np.hstack((lable[13, 1], lable[5:3:-1, 1])), c='y')
            ax.plot(np.hstack((lable[13, 0], lable[7:5:-1, 0])),
                    np.hstack((lable[13, 1], lable[7:5:-1, 1])), c='y')
            ax.plot(np.hstack((lable[13, 0], lable[10:7:-1, 0])),
                    np.hstack((lable[13, 1], lable[10:7:-1, 1])), c='y')
            ax.plot(np.hstack((lable[13, 0], lable[11, 0])),
                    np.hstack((lable[13, 1], lable[11, 1])), c='y')
            ax.plot(np.hstack((lable[13, 0], lable[12, 0])),
                    np.hstack((lable[13, 1], lable[12, 1])), c='y')

    def format_coord(x, y):
        numrows, numcols = dpt.shape
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = dpt[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)

    ax.format_coord = format_coord
    if save == False:
        plt.show()
    else:
        fig.savefig(imagename)
        plt.close(fig)


def showImageJointsandResults(dpt,lable,result,save=False,imagename=None,allJoints=False,line=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #print("img min {}, max {}".format(dpt.min(), dpt.max()))
    ax.imshow(dpt, cmap=matplotlib.cm.jet, interpolation='nearest')
    ax.scatter(lable[:, 0], lable[:, 1],color='y')
    ax.scatter(result[:, 0], result[:, 1], color='r')
    if line:
        if allJoints:
            ax.plot([lable[0,0],lable[1,0]],[lable[0,1],lable[1,1]],c='y')
            ax.plot([lable[1, 0], lable[2, 0]], [lable[1, 1], lable[2, 1]], c='y')
            ax.plot([lable[2, 0], lable[3, 0]], [lable[2, 1], lable[3, 1]], c='y')
            ax.plot([lable[3, 0], lable[4, 0]], [lable[3, 1], lable[4, 1]], c='y')
            ax.plot([lable[6, 0], lable[7, 0]], [lable[6, 1], lable[7, 1]], c='y')
            ax.plot([lable[7, 0], lable[8, 0]], [lable[7, 1], lable[8, 1]], c='y')
            ax.plot([lable[8, 0], lable[9, 0]], [lable[8, 1], lable[9, 1]], c='y')
            ax.plot([lable[9, 0], lable[10, 0]], [lable[9, 1], lable[10, 1]], c='y')
            ax.plot([lable[12, 0], lable[13, 0]], [lable[12, 1], lable[13, 1]], c='y')
            ax.plot([lable[13, 0], lable[14, 0]], [lable[13, 1], lable[14, 1]], c='y')
            ax.plot([lable[14, 0], lable[15, 0]], [lable[14, 1], lable[15, 1]], c='y')
            ax.plot([lable[15, 0], lable[16, 0]], [lable[15, 1], lable[16, 1]], c='y')
            ax.plot([lable[18, 0], lable[19, 0]], [lable[18, 1], lable[19, 1]], c='y')
            ax.plot([lable[19, 0], lable[20, 0]], [lable[19, 1], lable[20, 1]], c='y')
            ax.plot([lable[20, 0], lable[21, 0]], [lable[20, 1], lable[21, 1]], c='y')
            ax.plot([lable[21, 0], lable[22, 0]], [lable[21, 1], lable[22, 1]], c='y')
            ax.plot([lable[4, 0], lable[5, 0]], [lable[4, 1], lable[5, 1]], c='y')
            ax.plot([lable[10, 0], lable[11, 0]], [lable[10, 1], lable[11, 1]], c='y')
            ax.plot([lable[16, 0], lable[17, 0]], [lable[16, 1], lable[17, 1]], c='y')
            ax.plot([lable[22, 0], lable[23, 0]], [lable[22, 1], lable[23, 1]], c='y')
            ax.plot([lable[5, 0], lable[32, 0]], [lable[5, 1], lable[32, 1]], c='y')
            ax.plot([lable[11, 0], lable[32, 0]], [lable[11, 1], lable[32, 1]], c='y')
            ax.plot([lable[17, 0], lable[17, 0]], [lable[17, 1], lable[32, 1]], c='y')
            ax.plot([lable[23, 0], lable[32, 0]], [lable[23, 1], lable[32, 1]], c='y')
            ax.plot([lable[32, 0], lable[30, 0]], [lable[32, 1], lable[30, 1]], c='y')
            ax.plot([lable[32, 0], lable[31, 0]], [lable[32, 1], lable[31, 1]], c='y')
            ax.plot([lable[32, 0], lable[28, 0]], [lable[32, 1], lable[28, 1]], c='y')
            ax.plot([lable[28, 0], lable[27, 0]], [lable[28, 1], lable[27, 1]], c='y')
            ax.plot([lable[27, 0], lable[26, 0]], [lable[27, 1], lable[26, 1]], c='y')
            ax.plot([lable[26, 0], lable[25, 0]], [lable[26, 1], lable[25, 1]], c='y')
            ax.plot([lable[25, 0], lable[24, 0]], [lable[25, 1], lable[24, 1]], c='y')

            ax.plot([result[0,0],result[1,0]],[result[0,1],result[1,1]],c='r')
            ax.plot([result[1, 0], result[2, 0]], [result[1, 1], result[2, 1]], c='r')
            ax.plot([result[2, 0], result[3, 0]], [result[2, 1], result[3, 1]], c='r')
            ax.plot([result[3, 0], result[4, 0]], [result[3, 1], result[4, 1]], c='r')
            ax.plot([result[6, 0], result[7, 0]], [result[6, 1], result[7, 1]], c='r')
            ax.plot([result[7, 0], result[8, 0]], [result[7, 1], result[8, 1]], c='r')
            ax.plot([result[8, 0], result[9, 0]], [result[8, 1], result[9, 1]], c='r')
            ax.plot([result[9, 0], result[10, 0]], [result[9, 1], result[10, 1]], c='r')
            ax.plot([result[12, 0], result[13, 0]], [result[12, 1], result[13, 1]], c='r')
            ax.plot([result[13, 0], result[14, 0]], [result[13, 1], result[14, 1]], c='r')
            ax.plot([result[14, 0], result[15, 0]], [result[14, 1], result[15, 1]], c='r')
            ax.plot([result[15, 0], result[16, 0]], [result[15, 1], result[16, 1]], c='r')
            ax.plot([result[18, 0], result[19, 0]], [result[18, 1], result[19, 1]], c='r')
            ax.plot([result[19, 0], result[20, 0]], [result[19, 1], result[20, 1]], c='r')
            ax.plot([result[20, 0], result[21, 0]], [result[20, 1], result[21, 1]], c='r')
            ax.plot([result[21, 0], result[22, 0]], [result[21, 1], result[22, 1]], c='r')
            ax.plot([result[4, 0], result[5, 0]], [result[4, 1], result[5, 1]], c='r')
            ax.plot([result[10, 0], result[11, 0]], [result[10, 1], result[11, 1]], c='r')
            ax.plot([result[16, 0], result[17, 0]], [result[16, 1], result[17, 1]], c='r')
            ax.plot([result[22, 0], result[23, 0]], [result[22, 1], result[23, 1]], c='r')
            ax.plot([result[5, 0], result[32, 0]], [result[5, 1], result[32, 1]], c='r')
            ax.plot([result[11, 0], result[32, 0]], [result[11, 1], result[32, 1]], c='r')
            ax.plot([result[17, 0], result[17, 0]], [result[17, 1], result[32, 1]], c='r')
            ax.plot([result[23, 0], result[32, 0]], [result[23, 1], result[32, 1]], c='r')
            ax.plot([result[32, 0], result[30, 0]], [result[32, 1], result[30, 1]], c='r')
            ax.plot([result[32, 0], result[31, 0]], [result[32, 1], result[31, 1]], c='r')
            ax.plot([result[32, 0], result[28, 0]], [result[32, 1], result[28, 1]], c='r')
            ax.plot([result[28, 0], result[27, 0]], [result[28, 1], result[27, 1]], c='r')
            ax.plot([result[26, 0], result[25, 0]], [result[26, 1], result[25, 1]], c='r')
            ax.plot([result[27, 0], result[26, 0]], [result[27, 1], result[26, 1]], c='r')
            ax.plot([result[25, 0], result[24, 0]], [result[25, 1], result[24, 1]], c='r')
        else:
            ax.plot(np.hstack((lable[13, 0], lable[1::-1, 0])),
                    np.hstack((lable[13, 1], lable[1::-1, 1])), c='y')
            ax.plot(np.hstack((lable[13, 0], lable[3:1:-1, 0])),
                    np.hstack((lable[13, 1], lable[3:1:-1, 1])), c='y')
            ax.plot(np.hstack((lable[13, 0], lable[5:3:-1, 0])),
                    np.hstack((lable[13, 1], lable[5:3:-1, 1])), c='y')
            ax.plot(np.hstack((lable[13, 0], lable[7:5:-1, 0])),
                    np.hstack((lable[13, 1], lable[7:5:-1, 1])), c='y')
            ax.plot(np.hstack((lable[13, 0], lable[10:7:-1, 0])),
                    np.hstack((lable[13, 1], lable[10:7:-1, 1])), c='y')
            ax.plot(np.hstack((lable[13, 0], lable[11, 0])),
                    np.hstack((lable[13, 1], lable[11, 1])), c='y')
            ax.plot(np.hstack((lable[13, 0], lable[12, 0])),
                    np.hstack((lable[13, 1], lable[12, 1])), c='y')

            ax.plot(np.hstack((result[13, 0], result[1::-1, 0])),
                    np.hstack((result[13, 1], result[1::-1, 1])), c='r')
            ax.plot(np.hstack((result[13, 0], result[3:1:-1, 0])),
                    np.hstack((result[13, 1], result[3:1:-1, 1])), c='r')
            ax.plot(np.hstack((result[13, 0], result[5:3:-1, 0])),
                    np.hstack((result[13, 1], result[5:3:-1, 1])), c='r')
            ax.plot(np.hstack((result[13, 0], result[7:5:-1, 0])),
                    np.hstack((result[13, 1], result[7:5:-1, 1])), c='r')
            ax.plot(np.hstack((result[13, 0], result[10:7:-1, 0])),
                    np.hstack((result[13, 1], result[10:7:-1, 1])), c='r')
            ax.plot(np.hstack((result[13, 0], result[11, 0])),
                    np.hstack((result[13, 1], result[11, 1])), c='r')
            ax.plot(np.hstack((result[13, 0], result[12, 0])),
                    np.hstack((result[13, 1], result[12, 1])), c='r')

    def format_coord(x, y):
        numrows, numcols = dpt.shape
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = dpt[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)

    ax.format_coord = format_coord
    plt.axis('off')

    if save == False:
        plt.show()
    else:
        fig.savefig(imagename,bbox_inches='tight')
        plt.close(fig)


def showDepthLabelSeg_Syn(dpt,labels,topleftx,toplefty,Segarry):
    fig=plt.figure()
    ax=fig.add_subplot(121)
    ax.imshow(dpt)
    ax.scatter(labels[:,0]-topleftx,labels[:,1]-toplefty,c='y')
    bx=fig.add_subplot(122)
    seg_clu = Image.fromarray(Segarry, 'RGB')
    bx.imshow(seg_clu)
    plt.show()

def showDepthLabelResult_Syn(dpt,labels,results,topleftx,toplefty):
    fig=plt.figure()
    ax=fig.add_subplot(121)
    ax.imshow(dpt)
    ax.scatter(labels[:,0]-topleftx,labels[:,1]-toplefty,c='y')
    bx=fig.add_subplot(122)
    bx.imshow(dpt)
    bx.scatter(results[:,0]-topleftx,results[:,1]-toplefty,c='y')
    plt.show()

def showSeg(segarry):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    seg_image=Image.fromarray(segarry,'RGB')
    ax.imshow(seg_image)
    plt.show()

def trans3dt2d_syn(bone3d,view_M,proj_M,originrow,origincol):
    num_bone=bone3d.shape[0]
    bone_trans = np.zeros((num_bone, 3))
    for i in range(num_bone):
        cur_bone=bone3d[i]
        cur_bone_pos = np.append(cur_bone, 1)
        cur_bone_pos=cur_bone_pos.transpose()
        world=np.dot(view_M,cur_bone_pos)
        bone_trans[i,2]=world[2]
        hcpos = np.dot(proj_M, world)
        hcpos[0]=hcpos[0]/hcpos[3]
        hcpos[1]=hcpos[1]/hcpos[3]
        bone_trans[i,0] = origincol/2 + origincol/2*hcpos[0]
        bone_trans[i,1] = originrow/2 -originrow/2*hcpos[1]
    return bone_trans

def translabel2seg(label,colors):
    seg_arr=np.zeros((label.shape[0],label.shape[1],3),dtype='uint8')
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            seg_arr[i,j,:]=colors[int(label[i,j])]
    return seg_arr

def transonehot2label(onehot):
    label=np.zeros((onehot.shape[0],onehot.shape[1],1),dtype='uint8')
    for i in range(onehot.shape[0]):
        for j in range(onehot.shape[1]):
            label[i,j]=np.nonzero(onehot[i,j,:])[0]
    return label