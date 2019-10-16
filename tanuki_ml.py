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
    ### Here is the actual neural network ###
    model = Sequential()
    # Normalizes incoming inputs. First layer needs the input shape to work
    model.add(BatchNormalization(input_shape=input_shape))

    # Below layers were re-named for easier reading of model summary; this not necessary
    # Conv Layer 1
    model.add(TimeDistributed(Conv2D(60, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv1')))
    # Conv Layer 2
    model.add(TimeDistributed(Conv2D(50, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv2')))

    # Pooling 1
    model.add(TimeDistributed(MaxPooling2D(pool_size=pool_size)))

    # Conv Layer 3
    model.add(TimeDistributed(Conv2D(40, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv3')))
    # Conv Layer 4
    model.add(TimeDistributed(Conv2D(30, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv4')))
    # Conv Layer 5
    model.add(TimeDistributed(Conv2D(20, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv5')))
    # Pooling 2
    model.add(TimeDistributed(MaxPooling2D(pool_size=pool_size)))

    # Conv Layer 6
    model.add(ConvLSTM2D(filters=20, kernel_size=(3, 3), strides=(1, 1), activation='relu', data_format='channels_last',
                         padding='same', return_sequences=True))

    model.add(ConvLSTM2D(filters=10, kernel_size=(3, 3), strides=(1, 1), activation='relu', data_format='channels_last',
                         padding='valid', return_sequences=True))

    # Conv Layer 7
    model.add(ConvLSTM2D(filters=5, kernel_size=(3, 3), strides=(1, 1), activation='relu', data_format='channels_last',
                         padding='valid', return_sequences=False))

    # Pooling 3
    model.add(MaxPooling2D(pool_size=pool_size))

    # Upsample 1
    model.add(UpSampling2D(size=pool_size))

    # Deconv 1
    model.add(Conv2DTranspose(10, (3, 3), padding='valid', strides=(1, 1), activation='relu',output_shape = model.layers[8].output_shape[1:], name='Deconv1'))

    # Deconv 2
    model.add(Conv2DTranspose(20, (3, 3), padding='valid', strides=(1, 1), activation='relu',output_shape = model.layers[7].output_shape[1:], name='Deconv2'))

    # Upsample 2
    model.add(UpSampling2D(size=pool_size))

    # Deconv 3
    model.add(Conv2DTranspose(30, (3, 3), padding='valid', strides=(1, 1), activation='relu',output_shape = model.layers[5].output_shape, name='Deconv3'))

    # Deconv 4
    model.add(Conv2DTranspose(40, (3, 3), padding='valid', strides=(1, 1), activation='relu',output_shape = model.layers[4].output_shape, name='Deconv4'))

    # Deconv 5
    model.add(Conv2DTranspose(50, (3, 3), padding='valid', strides=(1, 1), activation='relu',output_shape = model.layers[3].output_shape, name='Deconv5'))

    # Upsample 3
    model.add(UpSampling2D(size=pool_size))

    # Deconv 6
    model.add(Conv2DTranspose(60, (3, 3), padding='valid', strides=(1, 1), activation='relu',output_shape = model.layers[1].output_shape, name='Deconv6'))

    # Final layer - only including one channel so 1 filter
    model.add(Conv2DTranspose(1, (3, 3), padding='valid', strides=(1, 1), activation='relu',output_shape = model.layers[0].output_shape, name='Final'))

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

def generate_prior_model(input_shape, pool_size):
    ### Here is the actual neural network ###
    model = Sequential()
    # Normalizes incoming inputs. First layer needs the input shape to work
    model.add(BatchNormalization(input_shape=input_shape))

    # Below layers were re-named for easier reading of model summary; this not necessary
    # Conv Layer 1
    model.add(Conv2D(8, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv1'))

    # Conv Layer 2
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv2'))

    # Pooling 1
    model.add(MaxPooling2D(pool_size=pool_size))

    # Conv Layer 3
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv3'))
    model.add(Dropout(0.2))

    # Conv Layer 4
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv4'))
    model.add(Dropout(0.2))

    # Conv Layer 5
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv5'))
    model.add(Dropout(0.2))

    # Pooling 2
    model.add(MaxPooling2D(pool_size=pool_size))

    # Conv Layer 6
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv6'))
    model.add(Dropout(0.2))

    # Conv Layer 7
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv7'))
    model.add(Dropout(0.2))

    # Pooling 3
    model.add(MaxPooling2D(pool_size=pool_size))

    # Upsample 1
    model.add(UpSampling2D(size=pool_size))

    # Deconv 1
    model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv1'))
    model.add(Dropout(0.2))

    # Deconv 2
    model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv2'))
    model.add(Dropout(0.2))

    # Upsample 2
    model.add(UpSampling2D(size=pool_size))

    # Deconv 3
    model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv3'))
    model.add(Dropout(0.2))

    # Deconv 4
    model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv4'))
    model.add(Dropout(0.2))

    # Deconv 5
    model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv5'))
    model.add(Dropout(0.2))

    # Upsample 3
    model.add(UpSampling2D(size=pool_size))

    # Deconv 6
    model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv6'))

    # Final layer - only including one channel so 1 filter
    model.add(Conv2DTranspose(1, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Final'))

    return model

def generate_model_without_Dropout(input_shape, pool_size):
    # 만약 이게 결과가 더 잘 나오면 학습 속도가 매우 느린거임.
    ### Here is the actual neural network ###
    model = Sequential()
    # Normalizes incoming inputs. First layer needs the input shape to work
    model.add(BatchNormalization(input_shape=input_shape))

    # Below layers were re-named for easier reading of model summary; this not necessary
    # LSTM Conv Layer 1
    model.add(ConvLSTM2D(filters=60, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last',
                         padding='valid', return_sequences=True))

    # LSTM Conv Layer 2
    model.add(ConvLSTM2D(filters=50, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last',
                         padding='valid'))

    # Pooling 1
    model.add(MaxPooling2D(pool_size=pool_size))

    # Conv Layer 3
    model.add(Conv2D(40, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv3'))

    # Conv Layer 4
    model.add(Conv2D(30, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv4'))

    # Conv Layer 5
    model.add(Conv2D(20, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv5'))

    # Pooling 2
    model.add(MaxPooling2D(pool_size=pool_size))

    # Conv Layer 6
    model.add(Conv2D(10, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv6'))

    # Conv Layer 7
    model.add(Conv2D(5, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv7'))

    # Pooling 3
    model.add(MaxPooling2D(pool_size=pool_size))

    # Upsample 1
    model.add(UpSampling2D(size=pool_size))

    # Deconv 1
    model.add(Conv2DTranspose(10, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv1'))

    # Deconv 2
    model.add(Conv2DTranspose(20, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv2'))

    # Upsample 2
    model.add(UpSampling2D(size=pool_size))

    # Deconv 3
    model.add(Conv2DTranspose(30, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv3'))

    # Deconv 4
    model.add(Conv2DTranspose(40, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv4'))

    # Deconv 5
    model.add(Conv2DTranspose(50, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv5'))

    # Upsample 3
    model.add(UpSampling2D(size=pool_size))

    # Deconv 6
    model.add(Conv2DTranspose(60, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv6'))

    # Final layer - only including one channel so 1 filter
    model.add(Conv2DTranspose(1, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Final'))

    return model

def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model