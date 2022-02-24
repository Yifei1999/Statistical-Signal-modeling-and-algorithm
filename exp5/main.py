# written by Liu Yifei, Sept 2, 2021
# exp5, course: statistic signal analysis and modeling
# this program is the appliance of GrabCut
import numpy as np
import cv2 as cv
import maxflow as mf
import Kmeans_init
import GMMmodel
import interface
from tqdm import tqdm
import greydiffstat

def GaussianCal( attr, means, covar):
    # calculate the probability of a M dim gaussian distribution
    # parameter:
    # - attr: 3 array, the target compute point
    # - means: M array
    # - covar: M*M array
    return np.exp(-0.5 * (((attr - means) @ np.linalg.inv(covar)) * (attr - means)).sum()) / np.sqrt(
        ((2 * np.pi) ** 3) * np.linalg.det(covar))

def MixGaussianCal(attr ,mean ,covar,pro):
    # calculate the probability of a mixture gaussian distribution
    # parameter:
    # - attr: 3 array, the target compute point
    # - tag: 1 for class A model, 2 for class B model
    Probility = 0
    for i in range(mean.__len__()):
        Probility = Probility + GaussianCal(attr, mean[i], covar[i]) * pro[i]
    return Probility

# img: 输入图像
# mask: 对应大小0，1矩阵
def SingleCut(img, mean, covar, pro ,index_gamma = 1.0,index_beta = 0.01):
    # network seperate function
    g = mf.GraphInt()
    nodeids = g.add_grid_nodes(img.shape[0:2])
    # setup the weight between the pixels in the network
    structureX = np.array([[0, 0, 0],  # build grid along x - axis, for eg. : between (x,y) and (x,y+1)
                          [0 ,0, 1],
                          [0 ,0, 0]])
    structureY = np.array([[0, 0, 0],    # build grid along y - axis
                           [0, 0, 0],
                           [0, 1, 0]])
    weightsX = np.zeros([img.shape[0] ,img.shape[1]])    # the weight along x - axis
    weightsY = np.zeros([img.shape[0], img.shape[1]])    # the weight along y - axis
    for i in range(img.shape[0]-1):    # calculate the weight
        for j in range(img.shape[1]-1):
            weightsX[i][j] = np.exp( - index_beta * sum( (img[i][j] - img[i][j+1]) ** 2) ) * index_gamma
            weightsY[i][j] = np.exp( - index_beta * sum( (img[i][j] - img[i+1][j]) ** 2) ) * index_gamma
    g.add_grid_edges(nodeids, weightsX, structureX, symmetric=True)
    g.add_grid_edges(nodeids, weightsY, structureY, symmetric=True)

    # setup the weights between pixels and source/end vertex
    sourcecaps = np.zeros([img.shape[0] ,img.shape[1]])
    sinkcaps = np.zeros([img.shape[0], img.shape[1]])
    for i in tqdm(range(img.shape[0])):
        for j in range(img.shape[1]):
            sourcecaps[i][j] = - np.log10(MixGaussianCal(img[i][j], mean[0], covar[0], pro[0]) )
            sinkcaps[i][j] = - np.log10(MixGaussianCal(img[i][j], mean[1], covar[1], pro[1]) )
    g.add_grid_tedges(nodeids, sourcecaps, sinkcaps)

    # seperate the network and receive the mask
    g.maxflow()
    mask = g.get_grid_segments(nodeids) # false: 源
    return mask

def PixelClassifier(mask,img,datasetb,datasetf):
    # classifying the pixel into 2 group, according to the mask
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if mask[i][j] == True:
                datasetf.append(img[i][j])
            else:
                datasetb.append(img[i][j])


if __name__ == '__main__':
    index_gamma = 3
    epoch = 3
    loadpath = './baboon.jpg'
    print(f' - set parameter: gamma = {index_gamma}')
    print(f' - set parameter: epoch : {epoch}')
    img = cv.imread(loadpath)
    print(f'loading {loadpath}')

    # select an initial background and foreground
    print('select an initial background and foreground, to confirm press ESC')
    Sel = interface.Selector(img)
    Sel.Interface()
    print(' - foreground selected')
    rect = Sel.rect

    # initial the mask and dataset of bg and fg
    mask = np.zeros([img.shape[0] ,img.shape[1]])
    mask[rect[0]:rect[0]+rect[2],rect[1]:rect[1]+rect[3]] = 1
    Datasetb = []
    Datasetf = []
    # set up GMMmodel for bg and fg
    C = GMMmodel.GMMmodel()
    index_beta = greydiffstat.GreyStat(img)
    for i in range(epoch):
        savepath = f'./{index_gamma}_{i+1}.jpg'
        # procedure:
        # - 1. classifying the pixel according to the mask
        # - 2. update GMM parameters
        # - 3. seperate the network and update the mask
        print(f" ======== epoch: round {i+1} start ======== ")
        PixelClassifier(mask,img,Datasetb,Datasetf)

        Kf = Kmeans_init.KmeansCluster()
        Kf.Initalise(Datasetf, 5, 3)
        Kf.TrainModel()
        Kb = Kmeans_init.KmeansCluster()
        Kb.Initalise(Datasetb, 5, 3)
        Kb.TrainModel()
        C.Initalise([Kf.EstimateMeans(), Kb.EstimateMeans()], [Kf.EstimateCov(), Kb.EstimateCov()],
                    [Kf.EstimatePro(), Kb.EstimatePro()],5)
        print(' - GMM update finished')

        mask = SingleCut(img,C.Means, C.Covar,C.Pro,index_gamma,index_beta)
        print(' - network seperated')

        # generate the mask array to display the img
        img_mask = np.zeros(img.shape[0:2],dtype=np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if mask[i][j] == True:
                    img_mask[i][j] = 255
        image_cut = cv.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=img_mask)
        cv.imshow('result', image_cut)

        # print('press any key to exit and save the image')
        # cv.waitKey(0)
        cv.imwrite(savepath, image_cut)
        # print('finished, display the result')