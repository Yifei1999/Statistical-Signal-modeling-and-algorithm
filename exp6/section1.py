import scipy.io as scio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import HOG
from tqdm import tqdm
import GenDataSet
from sklearn.neighbors import KNeighborsClassifier
import time

# parameter:
#  - K: the parameter for the KNN
#  - blocksize: the number of cell in a row for a block, blocksize = 2 means block is 2*2 cell
#  - cellsize: the size of a cell in pixel
#  - classnum: the number of classes in dataset provided
if __name__ == '__main__':
    K = 1
    blocksize = 2
    cellsize = 4
    classnum = 417
    print(f'parameter: \n blocksize:{blocksize}  cellsize {cellsize}  K:{K} ')
    path = './faces-ids-n6680-m417-20x20.mat'
    data = scio.loadmat(path)
    ids = data['ids'].T[0]
    faces = data['faces']
    # distinguish the train set and test set
    [traindataset,traindatatag,testdataset,testdatatag] = GenDataSet.GenDataSet(faces, ids, classnum, 4, 2)

    # get HOG
    trainfacecharacter = []
    print(' - analyzing image character on TRAIN set (HOG)')
    for i in tqdm(range(traindataset.shape[0]) ):
        img_charater = HOG.GetImgCharacter(traindataset[i,:].reshape([20,20]),blocksize,cellsize)
        trainfacecharacter.append(img_charater)
    trainfacecharacter = np.array(trainfacecharacter)
    print(f' - finish: get {(20/cellsize - blocksize + 1)**2 * (blocksize)**2 * 9} character per image')
    time.sleep(1)

    testfacecharacter = []
    print(' - analyzing image character on TEST set (HOG)')
    for i in tqdm(range(testdataset.shape[0]) ):
        img_charater = HOG.GetImgCharacter(testdataset[i,:].reshape([20,20]),blocksize,cellsize)
        testfacecharacter.append(img_charater)
    testfacecharacter = np.array(testfacecharacter)
    print(' - finish')

    KNN = KNeighborsClassifier(n_neighbors=K)
    KNN.fit(trainfacecharacter, traindatatag)
    print(' - KNN model built')
    print(' - prediting on KNN model ')
    predicttag = KNN.predict(testfacecharacter)

    # stat result
    print(' - analyizing result')
    correctnum = 0
    confusionmatrix = np.zeros([classnum,classnum])
    for i in range(len(predicttag)):
        confusionmatrix[testdatatag[i]-1][predicttag[i]-1] += 1
        if testdatatag[i] == predicttag[i] :
            correctnum += 1
    print(f' - result: correct rate {float(correctnum)/len(predicttag)}')

    face2 = Image.fromarray( 127*confusionmatrix )
    plt.imshow(face2)
    plt.show()
