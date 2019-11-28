from skimage.io import imsave
import pandas as pd
import numpy as np
import configparser

def save_info(info,path):
    config = configparser.RawConfigParser()
    config.read('config.txt')
    name = config.get('task settings','task name')
    infomation = pd.DataFrame(columns=['task','acc','loss','val_acc','val_loss','duration (s)','epoch'],
                        data=[[name,info['acc'],info['loss'],info['val_acc'],info['val_loss'],info['duration'],info['epoch']]])
    infomation.to_csv(path+'Training_info.csv',index=False)
    
def save_dice(dice_list,path):
    infomation = pd.DataFrame(columns=['dice_coef'],
                        data=dice_list)
    infomation.to_csv(path+'dice_info.csv')

def save_results(Predicted_mask_list,dir_path):
    pred = np.squeeze(Predicted_mask_list)
    imsave(dir_path+'Predicted_mask.tif',pred)