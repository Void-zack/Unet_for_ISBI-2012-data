from keras.layers import Input,Conv2D,MaxPooling2D,Dropout,UpSampling2D,concatenate
from keras.models import *
from keras.optimizers import *
from keras import initializers
from keras.utils import multi_gpu_model
import configparser

#============== READING CONFIG FILE ==============#
config = configparser.RawConfigParser()
config.read('config.txt')
edge = int(config.get('model settings','edge'))
gpus = int(config.get('model settings','gpus'))
seed = int(config.get('model settings','seed'))
lr = float(config.get('model settings','lr'))
#================= BUILDING MODEL =================#
kernel_initializer=initializers.he_normal(seed=seed)
loss = 'binary_crossentropy'
metrics = 'binary_accuracy'

def Unet():
    inputs = Input((edge,edge, 1),name='input')
    # -------------------------------------------------------------------------------------------------------------
    conv1_0 = Conv2D(64 , 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(inputs)
    conv1_1 = Conv2D(64 , 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv1_0)
    pool1 = MaxPooling2D(pool_size=(2))(conv1_1)
    # -------------------------------------------------------------------------------------------------------------
    conv2_0 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool1)
    conv2_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv2_0)
    pool2 = MaxPooling2D(pool_size=(2))(conv2_1)
    # -------------------------------------------------------------------------------------------------------------
    conv3_0 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool2)
    conv3_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv3_0)
    pool3 = MaxPooling2D(pool_size=(2))(conv3_1)
    # -------------------------------------------------------------------------------------------------------------
    conv4_0 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool3)
    conv4_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv4_0)
    drop4 = Dropout(0.5)(conv4_1)
    # 下采样结束
    # -------------------------------------------------------------------------------------------------------------
    # 开始上采样
    up5 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(drop4))
    merge5 = concatenate([conv3_1, up5], axis=3)
    conv5_0 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge5)
    conv5_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv5_0)
    # -------------------------------------------------------------------------------------------------------------
    up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(conv5_1))
    merge6 = concatenate([conv2_1, up6], axis=3)
    conv6_0 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge6)
    conv6_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv6_0)
    # -------------------------------------------------------------------------------------------------------------
    up7 = Conv2D(64 , 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(conv6_1))
    merge7 = concatenate([conv1_1, up7], axis=3)
    conv7_0 = Conv2D(64 , 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge7)
    conv7_1 = Conv2D(64 , 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv7_0)
    conv7 = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv7_1)
    # ----------------------------------------------------------------------------------------------------------
    conv8 = Conv2D(1, 1, activation='sigmoid',name='output')(conv7)
    # -------------------------------------------------------------------------------------------------------------
    
    model = Model(inputs=inputs, outputs=conv8) 
    if gpus>=2:
        model = multi_gpu_model(model,gpus=gpus)
    model.compile(optimizer=Adam(lr=lr), loss=loss, metrics=[metrics])

    return model