import numpy as np
import os
from PIL import Image

# Import necessary items from Keras
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.callbacks import Callback

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
    model.add(Conv2D(60, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1'))

    # Conv Layer 2
    model.add(Conv2D(50, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv2'))

    # Pooling 1
    model.add(MaxPooling2D(pool_size=pool_size))

    # Conv Layer 3
    model.add(Conv2D(40, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3'))
    model.add(Dropout(rate=0.2))

    # Conv Layer 4
    model.add(Conv2D(30, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv4'))
    model.add(Dropout(rate=0.2))

    # Conv Layer 5
    model.add(Conv2D(20, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv5'))
    model.add(Dropout(rate=0.2))

    # Pooling 2
    model.add(MaxPooling2D(pool_size=pool_size))

    # Conv Layer 6
    model.add(Conv2D(10, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv6'))
    model.add(Dropout(rate=0.2))

    # Conv Layer 7
    model.add(Conv2D(5, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv7'))
    model.add(Dropout(rate=0.2))

    # Pooling 3
    model.add(MaxPooling2D(pool_size=pool_size))

    # Upsample 1
    model.add(UpSampling2D(size=pool_size))

    # Deconv 1
    model.add(Conv2DTranspose(10, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv1'))
    model.add(Dropout(rate=0.2))

    # Deconv 2
    model.add(Conv2DTranspose(20, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv2'))
    model.add(Dropout(rate=0.2))

    # Upsample 2
    model.add(UpSampling2D(size=pool_size))

    # Deconv 3
    model.add(Conv2DTranspose(30, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv3'))
    model.add(Dropout(rate=0.2))

    # Deconv 4
    model.add(Conv2DTranspose(40, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv4'))
    model.add(Dropout(rate=0.2))

    # Deconv 5
    model.add(Conv2DTranspose(50, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv5'))
    model.add(Dropout(rate=0.2))

    # Upsample 3
    model.add(UpSampling2D(size=pool_size))

    # Deconv 6
    model.add(Conv2DTranspose(60, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv6'))

    # Final layer - only including one channel so 1 filter
    model.add(Conv2DTranspose(1, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Final'))
    
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

class AdaptiveLearningrate(Callback):
    def __init__(self, threshold=0.03, decay=0.5, relax=3, verbose=0):
        super(AdaptiveLearningrate, self).__init__()
        self.threshold = threshold
        self.verbose = verbose
        self.losses = []
        self.decay = decay
        self.relax = relax
        self.relaxMax = relax
        print("\n\n===== Adaptive Learning rate Manager =====\n")
        print("Programming by Hyeonho Shin, Hanyang University")
        print("Threshold rate is set to {}, Decay rate is set to {}\n".format(self.threshold, self.decay))

    def on_epoch_end(self, epoch, logs=None):
        # Error 처리 - optimizer의 lr이 없을 경우.
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        lr_prev = float(K.get_value(self.model.optimizer.lr))

        logs = logs or {}
        loss = logs.get('val_loss')
        self.losses.append(loss)

        if len(self.losses) > 1 and (self.relax == self.relaxMax):
            # 만약 이전 epoch의 loss와의 차이가 threshold
            progress = self.losses[epoch-1] - loss # 0.0186 - 0.0183 = 0.0003 -> 0.0183의 3%이하 -> 업데이트
            self.relax = 0 # relax하도록 Relax Time 초기화
            if progress < loss * self.threshold:
                # lr Update.
                lr = lr_prev * self.decay
                K.set_value(self.model.optimizer.lr, lr)
                if self.verbose > 0:
                    print("[Adaptive LR] @ epoch {} : Change of loss = {} - {}".format(epoch, self.losses[epoch-1], loss))
                    print("[Adaptive LR] @ epoch {} : Update! {} -> {}\n".format(epoch, lr_prev, lr))
            elif self.verbose > 0:
                print("[Adaptive LR] @ epoch {} : Large progress - No change\n".format(epoch))
        elif self.relax != self.relaxMax:
            print("[Adaptive LR] @ epoch {} : Relax time - no change\n".format(epoch))
            self.relax += 1
        else:
            if self.verbose > 0:
                print("[Adaptive LR] @ epoch 0 : epoch 0 직후에는 lr을 업데이트하지 않습니다.\n")