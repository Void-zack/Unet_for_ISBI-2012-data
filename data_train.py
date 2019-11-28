from skimage.io import imread
import time 
import os
import configparser
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from utils.processing import *
from utils.net import Unet
from utils.generator import m_gen
from utils.save_info import save_info
from utils.visualization import Training_visualization
import json

########################################
# Set up training configuration
########################################
config = configparser.RawConfigParser()
config.read('config.txt')
train_path = config.get('data path','train_path')
dir_path = config.get('data path','result_dir')
gpu_usage = config.get('model settings','gpu_usage')
seed = int(config.get('model settings','seed'))
epoch = int(config.get('train settings','epoch'))
steps_per_epoch = int(config.get('train settings','step_per_epoch'))
split_rate = float(config.get('train settings','split_rate'))
rfile = config.get('task settings','raw_file')
mfile = config.get('task settings','mask_file')

########################################
# Set up GPU usage mode
########################################
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usage

########################################
# Read & Prepare training data
########################################
raw = prep_raw(imread(train_path+rfile))
mask = train_mask(imread(train_path+mfile))
rtrain,rval = train_test_split(raw,test_size=split_rate, random_state=seed)
mtrain,mval = train_test_split(mask,test_size=split_rate, random_state=seed)

########################################
# Training
########################################
net = Unet()
checkpoint= ModelCheckpoint(dir_path+'model.hdf5',monitor='val_loss',verbose=1,mode='min',save_best_only=True)
print('\n...Training...')
start_time = time.time()
history = net.fit_generator(m_gen(rtrain,mtrain),
                            validation_data=(rval,mval),
                            shuffle=True,
                            callbacks=[checkpoint],
                            epochs=epoch,
                            steps_per_epoch=steps_per_epoch,
                            verbose=1)
end_time = time.time()
sum_time = end_time-start_time
print('last %.2f seconds'%sum_time)
print('\nTraining Finished!')

########################################
# Draw the training curve
########################################
Training_visualization(history,epoch,dir_path)

########################################
# Save the training info
########################################
info = {}
#train
info['acc'] = history.history['binary_accuracy'][-1]
info['loss'] = history.history['loss'][-1]
info['val_acc'] = history.history['val_binary_accuracy'][-1]
info['val_loss'] = history.history['val_loss'][-1]
#model
info['duration'] = sum_time
info['epoch'] = epoch
save_info(info,dir_path)