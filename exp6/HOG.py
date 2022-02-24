import scipy.io as scio
import numpy as np
import cv2


def GetImgCharacter(imgarr, blocksize, cellsize):
    # get the gradient character for an image
    blocknum = int(imgarr.shape[0] / cellsize) - (blocksize - 1)
    charlist = []
    for i in range(blocknum):
        for j in range(blocknum):
            imgblock = imgarr[cellsize*i:cellsize*(i+blocksize),:]
            imgblock = imgblock[:, cellsize*j:cellsize*(j+blocksize)]
            charlist = charlist + GetBlockCharacter(imgblock, cellsize)
    return charlist

def GetBlockCharacter(imgarr, cellsize):
    # get the gradient character for an block
    # the character is nomoralized
    cellnum = int(imgarr.shape[0] / cellsize)
    charlist = []
    for i in range(cellnum):
        for j in range(cellnum):
            imgcell = imgarr[cellsize*i:cellsize*(i+1),:]
            imgcell = imgcell[:,cellsize*j:cellsize*(j+1)]
            charlist = charlist + GetCellCharacter(imgcell)

    charlist = np.array(charlist)
    charlist = charlist / ( (charlist ** 2).sum() ** 0.5 )
    return list(charlist)

def GetCellCharacter(imgarr):
    # get the gradient character for an cell
    [mag,ang] = GradientGet(imgarr)
    charlist = []
    for i in range(9):
        (index_x, index_y) = np.where( (np.pi/9* i <= ang) & ( ang < np.pi/9 * (i+1) ) )
        magsel = mag[index_x,:]
        magsel = magsel[:,index_y]
        magsum = magsel.sum()
        charlist.append(magsum)
    return charlist

def GradientGet(imgarr):
    diff_x = cv2.Sobel(imgarr, cv2.CV_64F, 1, 0, ksize=3)
    diff_y = cv2.Sobel(imgarr, cv2.CV_64F, 0, 1, ksize=3)
    mag = (diff_x ** 2 + diff_y ** 2) ** 0.5
    ang = np.arctan( diff_y / (diff_x + 0.001) )
    ang = ang + np.pi/2
    return [mag,ang]

if __name__ == '__main__':
    # test module
    path = './faces-ids-n6680-m417-20x20.mat'
    data = scio.loadmat(path)
    ids = np.array(data['ids'])
    faces = np.array(data['faces'])
    face1 = faces[0,:].reshape([20,20]).T

    imgarr = face1
    GetImgCharacter(imgarr, 2,4) # block , cell


