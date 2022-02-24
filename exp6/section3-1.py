import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import Interface
import cv2
from PCApro import PCA

def GetGreyArray(img,iy,ix,selsize):
    imgsel = cv2.resize(img[iy:iy + selsize, ix:ix + selsize, :], [20, 20])
    img_r = np.array(imgsel[:, :, 0], dtype=float)
    img_g = np.array(imgsel[:, :, 0], dtype=float)
    img_b = np.array(imgsel[:, :, 0], dtype=float)
    img_grey = 0.2989 * img_r + 0.5870 * img_g + 0.1140 * img_b
    return img_grey

if __name__ == '__main__' :
    K = 1
    eigreserve = 64
    classnum = 417
    selsize = 100
    path = './faces-ids-n6680-m417-20x20.mat'
    data = scio.loadmat(path)
    ids = data['ids'].T[0]
    faces = data['faces']
    attributenum = faces.shape[1]
    samplenum = faces.shape[0]
    # fit the PCA model
    P = PCA(eigreserve)
    P.FitModel(faces)
    transmatrix = P.transmatrix

    img = cv2.imread('./groupimage.png')
    Sel = Interface.Selector(img, selsize)
    Sel.Interface()
    img_grey = GetGreyArray(img, Sel.iy, Sel.ix, selsize)
    img_recover = P.GegradationVec(img_grey.reshape([1, 20 * 20]).T)
    img_recover = img_recover.reshape([20, 20])

    plt.figure()
    plt.imshow(Image.fromarray(img_grey))
    plt.figure()
    plt.imshow(Image.fromarray(img_recover))
    plt.show()

