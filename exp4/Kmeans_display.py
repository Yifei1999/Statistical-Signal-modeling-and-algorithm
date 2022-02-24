# written by Liu Yifei, Nov 16, 2021
# exp4, course: statistic signal analysis and modeling
# this program is used to demostrate the effect of Kmeans algorithm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class KmeansCluster:
    def __init__(self):
        self.M_AttrNum = 3
        self.Epoch = 0
        self.Dataset  = None
        self.ClusterCentral  = []
        self.ClusterNum = None
        self.DataNum = 0

    def Initalise(self,dataset,clusternum,epoch):
        self.Dataset = [[]]
        for j in range(clusternum-1):
            self.Dataset.append([])
        for i,v in enumerate(dataset):
            index = int(np.floor(np.random.rand() *  (clusternum) ))
            self.Dataset[index].append(v)

        self.DataNum = len(dataset)
        self.ClusterNum = clusternum
        self.Epoch = epoch
        for k in range(self.ClusterNum):
            self.ClusterCentral.append(None)

    def DisMetic(self,pointa,pointb):    # 3 array   return:array
        return ((pointa-pointb)*(pointa-pointb)).sum()

    def CentalCal(self,dataset):        # list * array   return:array
        CentralPoint = np.array([0,0,0])
        for i, v in enumerate(dataset):
            CentralPoint = CentralPoint + dataset[i]
        return CentralPoint / len(dataset)

    def GetMinDisCentral(self,point):    # 3 array    return:  int
        MinPointIndex = 0
        MinDis = 9999999
        for i in range(self.ClusterNum):
            if self.DisMetic(point,self.ClusterCentral[i]) < MinDis:
                MinDis = self.DisMetic(point,self.ClusterCentral[i])
                MinPointIndex = i
        return MinPointIndex

    def SingleIter(self):
        for i in range(self.ClusterNum):
                self.ClusterCentral[i] = self.CentalCal(self.Dataset[i])

        TempDataset = []
        for i in range(self.ClusterNum):
            TempDataset.append([])
        for i, v in enumerate(self.Dataset):    # 类
            for j, w in enumerate(v):        # 点
                MinPointIndex = self.GetMinDisCentral(w)
                TempDataset[MinPointIndex].append(w)      # 分类
        self.Dataset = TempDataset

    def TrainModel(self):
        for i in range(self.Epoch):
            self.SingleIter()
            fig = plt.figure()
            ax = Axes3D(fig)
            Colorset = ['r', 'b', 'g', 'y', 'k', 'c', 'm', 'w']
            for i in range(self.ClusterNum):
                x = np.array(self.Dataset[i])[:, 0]
                y = np.array(self.Dataset[i])[:, 1]
                z = np.array(self.Dataset[i])[:, 2]
                ax.scatter(x, y, z, c=Colorset[i], s=15)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()

    def EstimateMeans(self):   #  list * array
        return self.ClusterCentral

    def EstimateCov(self):  # list * (n*n array)
        Covar = []
        for i in range(self.ClusterNum):
            Covar = Covar + [np.cov(self.Dataset[i], rowvar=False)]
        return Covar

    def EstimatePro(self):
        Pro = []
        for i in range(self.ClusterNum):
            Pro = Pro + [len(self.Dataset[i])/self.DataNum]
        return Pro

if __name__ == '__main__':
    # load the Train Set
    fp_trainset = open('./Train.txt', 'r')
    attrSet = []
    classSet = []
    for line in fp_trainset:
        ls = line.strip('\n').replace('(', ' ').replace(')', ' ').replace(':', ' ').split()
        ls_data = [ls[0]] + list(map(float, ls[1:4:1]))
        classSet.append(ls_data[0])
        attrSet.append(np.array(ls_data[1:4:1]))
    fp_trainset.close()
    print('loading trainset samples:', len(attrSet))

    DataSetA = []
    DataSetB = []
    for i, v in enumerate(classSet):
        if v == 'A':
            DataSetA.append(attrSet[i])
        else:
            DataSetB.append(attrSet[i])
    print('target set A:', len(DataSetA))
    print('target set B:', len(DataSetB))

    K = KmeansCluster()
    K.Initalise(DataSetA,4,3)     # A
    K.TrainModel()
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(K.ClusterNum):
        Colorset = ['r', 'b', 'g', 'y', 'k', 'c', 'm', 'w']
        x = np.array(K.Dataset[i])[:,0]
        y = np.array(K.Dataset[i])[:,1]
        z = np.array(K.Dataset[i])[:,2]
        ax.scatter(x, y, z, c=Colorset[i],s=15)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

