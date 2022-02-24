import cv2 as cv

def GreyStat(img):
    # estimate the prameter beta in the Grabcut algorithm
    sum = 0
    for i in range(img.shape[0]-1):
        for j in range(img.shape[1]-1):
           sum = sum + ((img[i][j].astype('int32') - img[i+1][j].astype('int32'))**2).sum() + ( (img[i][j].astype('int32') - img[i][j+1].astype('int32'))**2).sum()
    sum = float(sum)*2/(img.shape[0]-1)/(img.shape[1]-1)/2
    return 1/sum

# img = cv.imread('./baboon.jpg')
# print(GreyStat(img))