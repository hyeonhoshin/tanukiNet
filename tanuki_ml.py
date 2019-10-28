import numpy as np
import os
from PIL import Image

# Import necessary items from Keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D, ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def give_time(X, y, memory_size = 3):
    # Make time-dependent data
    # X : (data_idx, x, y) -> (data_idx, looking, x, y)
    data_size, width, height = X.shape[0], X.shape[1], X.shape[2]
    X_t = np.zeros((data_size, memory_size, width, height, 3), dtype='uint8')

    # y : (data_idx, x, y) -> (data_idx - memory_size, x, y)
    # memory_size가 3일때, idx 0~2 의 데이터를 바탕으로 idx 2의 답이 답임.
    y_t = np.expand_dims(y, axis=3)
    y_t = np.roll(y_t, - (memory_size - 1), axis=0)

    end_idx = 0

    for i, e in enumerate(X):
        try:
            X_t[i] = X[i:i + memory_size]
        except:
            end_idx = i
            print('* Give time : timed array has length = {}'.format(end_idx))
            break

    return X_t[:end_idx], y_t[:end_idx]

def generate_model(input_shape, pool_size):

    inputs = Input(input_shape)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool1)
    drop1 = Dropout(0.2)(conv2)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(drop1)
    drop2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(drop2)
    drop3 = Dropout(0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=pool_size)(drop3)
    #
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool2)
    drop4 = Dropout(0.2)(conv3)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(drop4)
    drop5 = Dropout(0.2)(conv3)
    pool3 = MaxPooling2D(pool_size=pool_size)(drop5)

    up = UpSampling2D(size = pool_size)(pool3)
    deconv1 = Conv2DTranspose(128, (3, 3), padding='valid', strides=(1, 1), activation='relu')(up)
    drop5 = Dropout(0.2)(deconv1)
    deconv1 = Conv2DTranspose(128, (3, 3), padding='valid', strides=(1, 1), activation='relu')(drop5)
    drop5 = Dropout(0.2)(deconv1)

    # Unet 따라하기
    merge = concatenate([pool2,drop5], axis = 3)

    up = UpSampling2D(size = pool_size)(merge)
    deconv1 = Conv2DTranspose(64, (3, 3), padding='valid', strides=(1, 1), activation='relu')(up)
    drop5 = Dropout(0.2)(deconv1)
    deconv1 = Conv2DTranspose(64, (3, 3), padding='valid', strides=(1, 1), activation='relu')(drop5)
    drop5 = Dropout(0.2)(deconv1)
    deconv1 = Conv2DTranspose(64, (3, 3), padding='valid', strides=(1, 1), activation='relu')(drop5)
    drop5 = Dropout(0.2)(deconv1)

    up = UpSampling2D(size = pool_size)(drop5)
    deconv1 = Conv2DTranspose(32, (3, 3), padding='valid', strides=(1, 1), activation='relu')(up)
    deconv_final = Conv2DTranspose(1, (3, 3), padding='valid', strides=(1, 1), activation='relu')(deconv1)
    
    model = Model(input = inputs, output = deconv_final)
    
    return model

def read_set(target, resized_shape):
    '''
    # Input
    target : 뒤져볼 폴더명
    
    # Output
    imgs, labels = 데이터와 정답. numpy 형식
    '''
    print('Start to read {} and {}'.format(target+'/test/data', target+'/label'))
    
    # Train data Read
    imgs = []

    ## data Read
    for root, dirs, files in os.walk(target+'/test/data'):
        # 일정 순서대로 읽기
        dirs.sort()
        files.sort()
        for fname in files:
            full_fname = os.path.join(root, fname)
            tmp_img = Image.open(full_fname)
            tmp_arr = np.array(tmp_img.resize(resized_shape))
            imgs.append(tmp_arr)
            del(tmp_img)
            del(tmp_arr)

    ##label Read
    labels = []

    for root, dirs, files in os.walk(target+'/reinforced_imgs/test/label'):
        # 일정 순서대로 읽기
        dirs.sort()
        files.sort()
        for fname in files:
            full_fname = os.path.join(root, fname)
            tmp_img = Image.open(full_fname)
            tmp_arr = np.array(tmp_img.resize(resized_shape))
            labels.append(tmp_arr[1:-1,:])
            del(tmp_img)
            del(tmp_arr)

    X = np.array(imgs)
    y = np.array(labels)
    
    del(labels)
    del(imgs)
    
    return X, y