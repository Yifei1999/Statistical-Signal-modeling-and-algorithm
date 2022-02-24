import scipy.io as scio
import numpy as np
import GenDataSet
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from PIL import Image
from PCApro import PCA

# vistualize the base vector of PCA
def VistualVec(transmatrix):
    vec = transmatrix[0:16, :].reshape([320, 20])
    outmatrix = np.zeros([80,80])
    for i in range(4):
        for j in range(4):
            outmatrix[i*20:i*20+20,j*20:j*20+20] = vec[(i*4+j)*20:(i*4+j)*20+20,:]
    return outmatrix

if __name__ == '__main__':
    K=1
    eigreserve = 64
    classnum = 417
    print(f'eigen value resered: {eigreserve}')
    path = './faces-ids-n6680-m417-20x20.mat'
    data = scio.loadmat(path)
    ids = data['ids'].T[0]
    faces = data['faces']
    attributenum = faces.shape[1]
    # generate transmit matrix
    P = PCA(eigreserve)
    P.FitModel(faces)
    transmatrix = P.transmatrix
    # extract character, the number of character is (eigreserve)
    character_res =( transmatrix @ faces.T ).T
    [traindataset,traindatatag,testdataset,testdatatag] = GenDataSet.GenDataSet(character_res, ids, classnum, 4, 2)  # face: 6680*400

    KNN = KNeighborsClassifier(n_neighbors=K)
    KNN.fit(traindataset, traindatatag)
    print(' - KNN model built')
    print(' - prediting on KNN model ')
    predicttag = KNN.predict(testdataset)

    # calculate predict correct rate
    print(' - analyizing result')
    correctnum = 0
    confusionmatrix = np.zeros([classnum, classnum])
    for i in range(len(predicttag)):
        confusionmatrix[testdatatag[i] - 1][predicttag[i] - 1] += 1
        if testdatatag[i] == predicttag[i]:
            correctnum += 1
    print(f' - result: correct rate {float(correctnum) / len(predicttag)}')


    confusiondisplay = Image.fromarray(127 * confusionmatrix)
    plt.figure()
    plt.imshow(Image.fromarray( 1000* VistualVec(transmatrix).T + 127 ) )    # vistualize the base vector
    plt.figure()
    plt.imshow(confusiondisplay)
    plt.show()

