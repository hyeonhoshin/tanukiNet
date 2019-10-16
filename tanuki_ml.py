import numpy as np
import os
from PIL import Image

# Import necessary items from Keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D, BilinearUpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

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
    # Keras FCN : AtrousFCN 수정
    img_input = Input(shape=input_shape)
    image_size = input_shape[0:2]

    # Block 1
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2)(img_input)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2)(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2)(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(1024, (7, 7), activation='relu', padding='same', dilation_rate=(2, 2),
                      name='fc1', kernel_regularizer=l2)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(1024, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2)(x)
    x = Dropout(0.5)(x)
    #classifying layer
    x = Conv2D(2, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2)(x)

    x = BilinearUpSampling2D(target_size=image_size)(x)

    model = Model(img_input, x)
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