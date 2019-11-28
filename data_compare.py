import os
import numpy as np
import configparser
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from skimage.io import imread
from keras.models import load_model
from utils.processing import prep_raw_plot
from utils.processing import test_mask
from utils.visualization import bina
from utils.visualization import Compare_results

def main():
    ID=int(sys.argv[1])
    ########################################
    # Set up compare configuration
    ########################################
    config = configparser.RawConfigParser()
    config.read('config.txt')
    test_path = config.get('data path','test_path')
    dir_path = config.get('data path','result_dir')
    rfile = config.get('task settings','raw_file')
    mfile = config.get('task settings','mask_file')
    ########################################
    # Predicting
    ########################################
    raw = prep_raw_plot(imread(test_path+rfile)[ID][np.newaxis,:,:])
    net = load_model(dir_path+'model.hdf5')
    pred_ = bina(np.squeeze(net.predict(raw)))
    label_ = test_mask(imread(test_path+mfile))[ID]
    raw_ = np.squeeze(raw)
    ########################################
    # Comparing
    ########################################
    Compare_results(raw_,label_,pred_,dir_path,ID)
if __name__ == "__main__":
    main()