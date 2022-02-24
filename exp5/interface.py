
import cv2



class Selector():
    def __init__(self,image):
        self.img = image
        self.img_copy = []
        self.ix = 0
        self.iy = 0
        self.rectflag = False
        self.rect = [0, 0, 0, 0]

    def MouseEvent(self,event, x, y, flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.rectflag = True
            self.ix = x
            self.iy = y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.rectflag == True:
                self.img = self.img_copy.copy()
                cv2.rectangle(self.img, (self.ix, self.iy), (x, y), [255, 255, 255], 2)
                self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
        elif event == cv2.EVENT_LBUTTONUP:
            self.rectflag = False
            cv2.rectangle(self.img, (self.ix, self.iy), (x, y), [255, 255, 255], 2)
            self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))

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
    img = cv2.imread('./1.jpg')
    Sel = Selector(img)
    Sel.Interface()
    print(Sel.rect)
