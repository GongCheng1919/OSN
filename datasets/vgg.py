import cv2
import numpy as np
import os
def get_vgg(vgg_path):
#     vgg_path="/data/gongcheng/CS_datasets/VGG"
#     dirs=os.listdir(vgg_path)
    dirs=['bark', 'bikes', 'boat', 'graf', 'leuven', 'trees', 'ubc', 'wall']
    print(dirs)
    imgpairs=[]
    for d in dirs:
        suffix="ppm"
        if d=="boat":
            suffix="pgm"
        img1=cv2.imread(vgg_path+"/"+d+"/img1.%s"%suffix,cv2.IMREAD_UNCHANGED)
        print(vgg_path+"/"+d+"/img1.%s"%suffix)
        for i in range(5):# pair 1 to 6
            img2=cv2.imread(vgg_path+"/"+d+"/img%d.%s"%(i+2,suffix),cv2.IMREAD_UNCHANGED)
            Hpath=vgg_path+"/"+d+"/H1to%dp"%(i+2)
            with open(Hpath) as f:
                _Hvalue=np.array([float(i) for i in f.read().split()]).reshape(3,3)
    #         print(d+"/img1.%s"%suffix,d+"/img%d.%s"%(i+2,suffix),d+"/H1to%dp"%(i+2))
            imgpair=[img1,img2,_Hvalue]
            imgpairs.append(imgpair)
    return imgpairs