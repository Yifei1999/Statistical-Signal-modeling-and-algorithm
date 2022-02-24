import numpy as np

# setup  a GMM model
class GMMmodel:
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

    def Initalise(self,means,covar,pro,k=5):
        # initalise function
        self.Means = means
        self.Covar = covar
        self.Pro = pro
        self.K_MulNum = k

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


