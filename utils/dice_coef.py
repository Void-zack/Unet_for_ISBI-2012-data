import numpy as np
import cv2 as cv

def dice_coef(true,pred):
    smooth=true.shape[-1]**2*0.01
    pred = np.array(pred).flatten()
    kernel = np.ones((5,5),np.uint8)
    true_E =  cv.erode(true, kernel, iterations=2).flatten()
    true_D = cv.dilate(true, kernel, iterations=2).flatten()
    true_coef = (true_E-true_D+1)
    intersection = np.sum(true_E*pred)
    return (2*intersection+smooth)/(np.sum(true_E)+np.sum(pred*true_coef)+smooth)