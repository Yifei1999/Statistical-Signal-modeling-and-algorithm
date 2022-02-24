import numpy as np

import cv2
import numpy as np
import  matplotlib.pyplot as plt
from numba import jit
import time


img = cv2.imread("./1.jpg")


@jit
def imgMin(img,val):
    H,W,C = img.shape
    img_min = np.zeros((H,W),np.uint8)
    for i in range(H):
        for j in range(W):
            img_min[i,j] = np.max(img[i,j]) +1
    return img_min

t2 = time.time()
imgMin(img,1)
t3 = time.time()
print("加速时间：",(t3-t2))


plt.show()