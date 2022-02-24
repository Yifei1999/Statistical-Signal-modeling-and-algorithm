# written by Liu Yifei, Nov 16, 2021
# exp4, course: statistic signal analysis and modeling
# main program for Gaussian Mixture Model Classifier
# for the details estimating P Means and Cov, refer to the reference report
import numpy as np
import Kmeans_init as Kint


class GaussianMixClassifier:
    def __init__(self):
        # denote: N:Num of samples  M:Num of attributes  K: Num of mix models
        # Means: the mean of distribute
        # Covar: the Covariance of distribute
        # Pro: the ratio of a single gaussian component
        # Rpoch: iteration times
        self.Means = None # C list - K list - M array
        self.Covar = None # C list - K list - M*M array
        self.Pro = None # C list - k list
        self.K_MulNum = 0 # K
        self.M_AttrNum = 3 # M=3
        self.C_ClassNum = 2 # C=2
        self.Epoch = 0

    def Initalise(self,means,covar,pro,k,epoch=3):
        # initalise function
        self.Epoch = epoch
        self.Means = means
        self.Covar = covar
        self.Pro = pro
        self.K_MulNum = k
        self.Epoch = epoch

    def GaussianCal(self, attr, means, covar):
        # calculate the probability of a M dim gaussian distribution
        # parameter:
        # - attr: 3 array, the target compute point
        # - means: M array
        # - covar: M*M array
        return np.exp(-0.5 * (((attr - means) @ np.linalg.inv(covar)) * (attr - means)).sum()) / np.sqrt( ((2*np.pi)**3) * np.linalg.det(covar))

    def MixGaussianCal(self, attr, tag):
        # calculate the probability of a mixture gaussian distribution
        # parameter:
        # - attr: 3 array, the target compute point
        # - tag: 1 for class A model, 2 for class B model
        Probility = 0
        for i in range(self.K_MulNum):
            Probility = Probility + self.GaussianCal(attr, self.Means[tag-1][i], self.Covar[tag-1][i] ) * self.Pro[tag-1][i]
        return Probility

    def SingleIter(self,dataset,tagset):
        # doing the iteration process once
        # parameter:
        # - dataset: list - 3 list, example: [ [1.2,1.7,4.3],[3.4,5.6,1.1] ]
        # - tagset: list, the corresponding tag set, example: ['A','B','A']

        # classify the dataset into 2 set
        DataSetA = []
        DataSetB = []
        for i, v in enumerate(tagset):
            if v == 'A':
                DataSetA.append(dataset[i])
            else:
                DataSetB.append(dataset[i])
        # train A and B using corresponding dataset
        self.SingleIter_TrainSingleType(DataSetA, 1)
        self.SingleIter_TrainSingleType(DataSetB, 2)

    def SingleIter_TrainSingleType(self,dataset,tag):
        # training mixture model for A (tag = 1) or B(tag = 2)
        # dataset: [ [],[] ]
        Pro_nk = []
        for n in range(len(dataset)):
            Pro_nk_TempRow = []    # K list
            for k in range(self.K_MulNum):
                Pro_nk_TempRow.append(self.Pro[tag-1][k] * self.GaussianCal(dataset[n], self.Means[tag-1][k], self.Covar[tag-1][k] ) / self.MixGaussianCal(dataset[n], tag) )
            Pro_nk = Pro_nk + [Pro_nk_TempRow]
        Pro_Sum = list(sum( np.array(Pro_nk)))   #  K list

        # estimate Pro
        self.Pro[tag-1] = list( np.array(Pro_Sum) / len( dataset ) )  # type  K array

        # estimate Mean
        MeansTemp = []
        for k in range(self.K_MulNum):
            FracSum = np.zeros([self.M_AttrNum])
            for i in range(len(dataset)):
                FracSum = FracSum + np.array( dataset[i] ) * Pro_nk[i][k]
            MeansTemp = MeansTemp + [FracSum / Pro_Sum[k]]
        self.Means[tag-1] = MeansTemp

        # estimate Cov
        CovarTemp = []
        for k in range(self.K_MulNum):
            FracSum = np.zeros([self.M_AttrNum,self.M_AttrNum])
            for i in range(len(dataset)):
                FracSum = FracSum + np.multiply( (np.array(dataset[i]) - self.Means[tag-1][k]).reshape(-1,1), (np.array(dataset[i]) - self.Means[tag-1][k]) ) * Pro_nk[i][k]
            CovarTemp = CovarTemp + [FracSum / Pro_Sum[k]]
        self.Covar[tag-1] = CovarTemp

        print("train",chr(64+tag),"finish")

    def TrainModel(self,dataset,tagset):
        print("start training model...")
        print("---------- start ----------")
        for i in range(self.Epoch):
            print("round:",i+1)
            self.SingleIter(dataset, tagset)
            print("---------- round: ",i+1," train finish ----------")

    def TestSamples(self,dataset,labelset):
        # test model on the test set
        print("test on given set...")
        print("load samples:",len(dataset))
        CountTrue = 0
        for Counter, SingleSample in enumerate(dataset):
            if self.MixGaussianCal(SingleSample, 1) > self.MixGaussianCal(SingleSample, 2):   # A
                if (labelset[Counter] == 'A'):
                    CountTrue = CountTrue + 1
            else: # B
                if (labelset[Counter] == 'B'):
                    CountTrue = CountTrue + 1

        print('test Set: ', Counter + 1, 'correct:', CountTrue,"Cor Rate:",CountTrue/(Counter+1))
        return CountTrue / (Counter + 1)

if __name__ == '__main__':
    # number of clusters and iteration epoch
    MixtureNum = 4
    epoch = 2
    print("Mixture NUM:",MixtureNum,"Epoch:",epoch)

    # loading the training set
    fp_trainset = open('./Train.txt', 'r')
    attrSet = []
    classSet = []
    for line in fp_trainset:
        ls = line.strip('\n').replace('(', ' ').replace(')', ' ').replace(':', ' ').split()
        ls_data = [ls[0]] + list(map(float, ls[1:4:1]))
        classSet.append(ls_data[0])
        attrSet.append(ls_data[1:4:1])
    fp_trainset.close()
    print("loading trainset samples:", len(attrSet))
    # classifying the data according to their class (A or B)
    DataSetA = []
    DataSetB = []
    for i, v in enumerate(classSet):
        if v == 'A':
            DataSetA.append(attrSet[i])
        else:
            DataSetB.append(attrSet[i])
    print("set ClassA num:", len(DataSetA))
    print("set ClassB num:", len(DataSetB))

    # load the Test Set
    fp_testset = open('./Test.txt', 'r')
    testAttrSet = []
    testClassSet = []
    for line in fp_testset:
        ls = line.strip('\n').replace('(', ' ').replace(')', ' ').replace(':', ' ').split()
        ls_data = [ls[0]] + list(map(float, ls[1:4:1]))
        testClassSet.append(ls_data[0])
        testAttrSet.append(ls_data[1:4:1])
    fp_testset.close()

    # setup a model for class A, a model for class B
    # the K-mean model is used to generate the initial value used in EM Algorithm
    KA = Kint.KmeansCluster()
    KA.Initalise(DataSetA, MixtureNum, 5)
    KA.TrainModel()
    KB = Kint.KmeansCluster()
    KB.Initalise(DataSetB, MixtureNum, 5)
    KB.TrainModel()

    # setup Gaussian Mixture Model and operate EM Algorithm
    C = GaussianMixClassifier()
    C.Initalise([KA.EstimateMeans(),KB.EstimateMeans()],[KA.EstimateCov(),KB.EstimateCov()],[KA.EstimatePro(),KB.EstimatePro()],MixtureNum ,epoch)
    C.TrainModel(attrSet,classSet)
    C.TestSamples(testAttrSet,testClassSet)


