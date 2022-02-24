from main import GaussianMixClassifier
import matplotlib.pyplot as plt
import Kmeans_init as Kint

if __name__ == '__main__':
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

    MixtureNum = 4
    epoch = 100
    # setup a model for class A, a model for class B
    # the K-mean model is used to generate the initial value used in EM Algorithm
    KA = Kint.KmeansCluster()
    KA.Initalise(DataSetA, MixtureNum, 5)
    KA.TrainModel()
    KB = Kint.KmeansCluster()
    KB.Initalise(DataSetB, MixtureNum, 5)
    KB.TrainModel()

    print("----- Set up Mixture Gaussian Model ------")
    # setup Gaussian Mixture Model and operate EM Algorithm
    C = GaussianMixClassifier()
    C.Initalise([KA.EstimateMeans(),KB.EstimateMeans()],[KA.EstimateCov(),KB.EstimateCov()],[KA.EstimatePro(),KB.EstimatePro()],MixtureNum ,epoch)

    CorrectRatio = []
    for i in range(epoch):
        C.SingleIter(attrSet,classSet)
        Ratio = C.TestSamples(testAttrSet,testClassSet)
        #Ratio = C.TestSamples(attrSet, classSet)
        CorrectRatio.append(Ratio)

    print(CorrectRatio)
    plt.plot(CorrectRatio)
    plt.show()
