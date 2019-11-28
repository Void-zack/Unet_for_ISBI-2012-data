import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import configparser
from skimage.io import imread
from keras.models import load_model
from utils.processing import prep_raw
from utils.processing import test_mask
from utils.visualization import Dice_visualization
from utils.save_info import save_results
 
########################################
# Set up Predicting configuration
########################################
config = configparser.RawConfigParser()
config.read('config.txt')
test_path = config.get('data path','test_path')
dir_path = config.get('data path','result_dir')
rfile = config.get('task settings','raw_file')
mfile = config.get('task settings','mask_file')
save = config.get('predict settings','save result(YES/NO)')

########################################
# Read data
########################################
raw = prep_raw(imread(test_path+rfile))
mask = test_mask(imread(test_path+mfile))

########################################
# Read model and predict images
########################################
net = load_model(dir_path+'model.hdf5')
print('\n...Predicting...')
Predicted_mask = net.predict(raw,batch_size=1,verbose=1)

########################################
# Save results and draw dice curve
########################################
if save=='YES':
    save_results(Predicted_mask,dir_path)
Dice_visualization(Predicted_mask,mask,dir_path)