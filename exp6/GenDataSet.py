import numpy as np
import scipy.io as scio


def GenDataSet(faces, ids, classnum, trainsetnum, testsetnum):
    # the function seperate the dataset into trainset and testset, returning their correponding labels
    # parameter:
    # - classnum: the number of classes used
    # - trainsetnum: the number of samples of a single class in trainset
    # - testsetnum: the number of samples of a single class in testset
    searchindex = 0
    traindataset = []
    testdataset = []
    traindatatag = []
    testdatatag = []
    for i in range(classnum):
        for j in range(trainsetnum):
            traindataset.append( faces[searchindex] )
            traindatatag.append(i+1)
            searchindex += 1
        for j in range(testsetnum):
            testdataset.append( faces[searchindex] )
            testdatatag.append(i+1)
            searchindex += 1
        while ids[searchindex] == i+1:
            if i == classnum - 1 :
                break
            searchindex += 1

    traindataset = np.array(traindataset)
    testdataset = np.array(testdataset)
    traindatatag = np.array(traindatatag)
    testdatatag = np.array(testdatatag)
    return [traindataset,traindatatag,testdataset,testdatatag]

if __name__ == '__main__' :
    # test model
    path = './faces-ids-n6680-m417-20x20.mat'
    data = scio.loadmat(path)
    ids = data['ids']
    faces = data['faces']
    [traindataset,traindatatag,testdataset,testdatatag] = GenDataSet(faces, ids, 20, 4, 2)
    print('finish')
