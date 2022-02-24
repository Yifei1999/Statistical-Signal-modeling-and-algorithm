import cv2

class Selector():
    def __init__(self,image,size):
        self.img = image
        self.img_copy = []
        self.ix = 0
        self.iy = 0
        self.rectflag = False
        self.rectsize = size

    def MouseEvent(self,event, x, y, flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ix = x
            self.iy = y
            self.img = self.img_copy.copy()
            cv2.rectangle(self.img, (self.ix, self.iy), (self.ix+self.rectsize, self.iy+self.rectsize), [255, 255, 255], 2)

    # 复制一份送入
    def Interface(self):
        self.img_copy = self.img.copy()
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.MouseEvent)
        while (1):
            cv2.imshow('image', self.img)
            if cv2.waitKey(20) & 0xFF == 27:  # 27是ESC ，退出
                break

if __name__ == "__main__":
    img = cv2.imread('./groupimage.png')
    Sel = Selector(img,40)
    Sel.Interface()
    print(Sel.ix,Sel.iy)
