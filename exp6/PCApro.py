import numpy as np


class PCA:
    eigreserve = 1
    w_res = None
    transmatrix = None
    def __init__(self, eigreserve):
        self.eigreserve = eigreserve

    def FitModel(self, data):
        samplenum = data.shape[0]
        C_oigin = 1.0 / (samplenum - 1) * data.T @ data
        w, v = np.linalg.eig(C_oigin)
        w_res = []
        v_res = []
        for i in range(self.eigreserve):
            w_res.append(w[i])
            v_res.append((v[:, i]).T)
        self.w_res = np.array(w_res)
        self.transmatrix = np.array(v_res)

    def GegradationVec(self,matrix):
        # receive a matrix and return the correponding recovered array
        # each col is a attribute vector from a sample
        matrix_remain = self.transmatrix @ matrix
        matrix_recovered = self.transmatrix.T @ matrix_remain
        return matrix_recovered


