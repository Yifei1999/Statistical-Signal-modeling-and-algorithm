import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from PCApro import PCA

def GetGreyArray(img,iy,ix,selsize):
    imgsel = cv2.resize(img[iy:iy + selsize, ix:ix + selsize, :], [20, 20])
    img_r = np.array(imgsel[:, :, 0], dtype=float)
    img_g = np.array(imgsel[:, :, 1], dtype=float)
    img_b = np.array(imgsel[:, :, 2], dtype=float)
    img_grey = 0.299 * img_r + 0.587 * img_g + 0.114 * img_b
    img_grey = img_grey.astype('uint8')
    return img_grey

def ImgCompensate(img, blocksize, step):
    # Refine the boundary of img
    # step is (blocksize/footstep), footstep is measured in pixel in one direction (x or y)
    footstep = int(blocksize/step)
    compensatemaskx = np.ones(img.shape)*step
    for i in range(step-1):
        compensatemaskx[0+i*footstep:footstep+i*footstep,:] -= step - i -1
    temp = np.flipud(compensatemaskx/step)
    compensatemaskx = compensatemaskx/step * temp
    compensatemaskx = 1/ compensatemaskx

    compensatemasky = np.ones(img.shape)*step
    for i in range(step - 1):
        compensatemasky[:,0 + i * footstep:footstep + i * footstep] -= step - i -1
    temp = np.fliplr(compensatemasky/step)
    compensatemasky = compensatemasky/step * temp
    compensatemasky = 1 / compensatemasky

    return compensatemaskx*compensatemasky

if __name__ == '__main__' :
    eigreserve = 64
    selsize = 20
    step = 4
    footstep = int(selsize / step)
    classnum = 417

    print(f'parameter:')
    print(f' - eig:{eigreserve}')
    print(f' - window size:{selsize}')
    print(f' - footstep:{footstep}')
    path = './faces-ids-n6680-m417-20x20.mat'
    data = scio.loadmat(path)
    ids = data['ids'].T[0]
    faces = data['faces']
    attributenum = faces.shape[1]
    samplenum = faces.shape[0]
    # fit the PCA model
    P = PCA(eigreserve)
    P.FitModel(faces)

    img = cv2.imread('./groupimage.png')
    imgsizex = img.shape[1]
    imgsizey = img.shape[0]

    blocknumx = int(np.floor( (imgsizex - (selsize - footstep)) / footstep) )
    blocknumy = int(np.floor((imgsizey - (selsize - footstep)) / footstep) )
    img_recover = np.zeros([ int( (selsize - footstep) + footstep*blocknumy),int( (selsize - footstep) + footstep*blocknumx)])
    for i in range(blocknumy):
        for j in range(blocknumx):
            block = GetGreyArray(img, i*footstep, j*footstep, selsize)
            block = P.GegradationVec(block.reshape([1, 20 * 20]).T)
            block = block.reshape([20, 20])
            block = cv2.resize(block, [selsize, selsize])
            img_recover[i*footstep:i*footstep+selsize,j*footstep:j*footstep+selsize] += 1 / step ** 2 * block
    compensate = ImgCompensate(img_recover, selsize, step)
    # get the grey layer of original image
    img_r = np.array(img[:, :, 0], dtype=float)
    img_g = np.array(img[:, :, 1], dtype=float)
    img_b = np.array(img[:, :, 2], dtype=float)
    img_grey = 0.299 * img_r + 0.587 * img_g + 0.114 * img_b
    img_diff = compensate*img_recover - img_grey[:img_recover.shape[0],:img_recover.shape[1]]
    img_diff = abs(img_diff*2)

    plt.figure()
    plt.imshow(Image.fromarray(compensate*img_recover))
    plt.figure()
    plt.imshow(Image.fromarray(img_diff))
    error = (img_diff ** 2).sum()/ img_diff.shape[0] /img_diff.shape[1]
    print(f' - estimate error: {error} ')
    plt.show()

