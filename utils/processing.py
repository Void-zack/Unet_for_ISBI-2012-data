import numpy as np
from PIL import Image
from PIL import ImageOps
import cv2 as cv
import configparser

config = configparser.RawConfigParser()
config.read('config.txt')
edge = int(config.get('model settings','edge'))

# prep_Raw
# 3D in 4D out
# 0~255 to 0~1
# 256*256 out
def prep_raw(imgs):
    Img = []
    for img in imgs:
        Img.append(cv.resize(img,(edge,edge),cv.INTER_LINEAR))
    Img = np.array(Img)
## Histogram Equalization：
#     for i in range(len(Img)):
#         Img[i] = np.array(ImageOps.equalize(Image.fromarray((Img[i]-(255-Img[i].max()+Img[i].min())/2).astype('uint8'))))
    Img = Img.astype(np.float)/255.0
    return Img[:,:,:,np.newaxis]

# train_mask
# 3D in 4D out
# 0~255 to 0,1
# 256*256 out
def train_mask(img):
    img = test_mask(img)
    return img[:,:,:,np.newaxis]

# test_mask
# 3D in 3D out
# 0~255 to 0,1
# 256*256 out
def test_mask(imgs):
    Img = []
    for img in imgs:
        Img.append(cv.resize(img,(edge,edge),cv.INTER_LINEAR))
    Img = np.array(Img)
    for i in range(len(Img)):
        Img[i] = Img[i]/255
        Img[i][Img[i] >= 0.5] = 1
        Img[i][Img[i] < 0.5] = 0
    return Img.astype('uint8')

# plot raw image
def prep_raw_plot(imgs):
    Img = []
    for img in imgs:
        Img.append(cv.resize(img,(edge,edge),cv.INTER_LINEAR))
    Img = np.array(Img)
    Img = Img.astype(np.float)/255.0
    return Img[:,:,:,np.newaxis]