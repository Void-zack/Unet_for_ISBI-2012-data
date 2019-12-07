import os
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from skimage.io import imread
from keras.models import load_model
from utils.processing import prep_raw
from utils.processing import prep_raw_plot
from utils.processing import test_mask
from utils.visualization import bina
from utils.visualization import Compare_results
from utils.read_config import read

def main():
    ID=int(sys.argv[1])
    ########################################
    # Set up compare configuration
    ########################################
    config = read()
    test_path = config['test_path']
    dir_path = config['result_dir']
    rfile = config['raw_file']
    mfile = config['mask_file']
    ########################################
    # Predicting
    ########################################
    img = imread(test_path+rfile)[ID][np.newaxis,:,:]
    raw = prep_raw(img)
    raw_plot = prep_raw_plot(img)
    net = load_model(dir_path+'model.hdf5')
    pred_ = bina(np.squeeze(net.predict(raw)))
    label_ = test_mask(imread(test_path+mfile))[ID]    
    raw_ = np.squeeze(raw_plot)
    ########################################
    # Comparing
    ########################################
    Compare_results(raw_,label_,pred_,dir_path,ID)
if __name__ == "__main__":
    main()