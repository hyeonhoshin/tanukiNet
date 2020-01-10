import numpy as np
import os
from PIL import Image

# Import necessary items from Keras
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

from keras.models import Model
from keras.layers import Conv2D, Input, MaxPooling2D, Dropout, UpSampling2D,Conv2DTranspose
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.callbacks import Callback

from metrics import iou_loss_core, competitionMetric2, K
from attention_module import attach_attention_module

from skimage import feature, transform

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

    for i, _ in enumerate(X):
        try:
            X_t[i] = X[i:i + memory_size]
        except:
            end_idx = i
            print('* Give time : timed array has length = {}'.format(end_idx))
            break

    return X_t[:end_idx], y_t[:end_idx]

def generate_model(input_shape, pool_size):

    inputs = Input(input_shape)
    batch = BatchNormalization()(inputs)
    h = Conv2D(128, (3, 3), padding = 'valid', activation = 'relu')(batch)
    h = Conv2D(64, (3, 3), padding = 'valid', activation = 'relu')(h)
    h = attach_attention_module(h, attention_module = 'cbam_block')
    pool = MaxPooling2D(pool_size=pool_size)(h)

    h = Dropout(0.1)(Conv2D(56, (3, 3), padding = 'valid', activation = 'relu')(pool))
    h = Dropout(0.1)(Conv2D(48, (3, 3), padding = 'valid', activation = 'relu')(h))
    h = Dropout(0.1)(Conv2D(32, (3, 3), padding = 'valid', activation = 'relu')(h))
    h = attach_attention_module(h, attention_module = 'cbam_block')
    pool = MaxPooling2D(pool_size=pool_size)(h)

    h = Dropout(0.1)(Conv2D(16, (3, 3), padding = 'valid', activation = 'relu')(pool))
    h = Dropout(0.1)(Conv2D(8, (3, 3), padding = 'valid', activation = 'relu')(h))
    h = attach_attention_module(h, attention_module = 'cbam_block')
    pool = MaxPooling2D(pool_size=pool_size)(h)

    up = UpSampling2D(size = pool_size)(pool)
    h = Dropout(0.1)(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1, 1), activation='relu')(up))
    h = Dropout(0.1)(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1, 1), activation='relu')(h))

    up = UpSampling2D(size = pool_size)(h)
    h = Dropout(0.1)(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1, 1), activation='relu')(up))
    h = Dropout(0.1)(Conv2DTranspose(48, (3, 3), padding='valid', strides=(1, 1), activation='relu')(h))
    h = Dropout(0.1)(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1, 1), activation='relu')(h))

    up = UpSampling2D(size = pool_size)(h)
    h = Conv2DTranspose(128, (3, 3), padding='valid', strides=(1, 1), activation='relu')(up)
    deconv_final = Conv2DTranspose(1, (3, 3), padding='valid', strides=(1, 1), activation='relu')(h)

    model = Model(inputs = inputs, outputs = deconv_final)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[iou_loss_core, competitionMetric2, 'accuracy'])
 
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
        print("Threshold rate is set to {}, Decay rate is set to {}".format(self.threshold, self.decay))
        print("After updating, lr updater have a break during {} epochs\n".format(self.relaxMax))

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
            if progress < loss * self.threshold:
                self.relax = 0 # relax하도록 Relax Time 초기화
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

class path_determiner:
    def tan(self,p_start,p_end):
        return (p_end[1]-p_start[1])/(p_end[0]-p_start[0])
    def approx_path(self,img):

        # Check img's dimension
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img[..., 0]
        elif len(img.shape) != 2:
            print("[Error] Input dimension is not proper!")
            print(" Img shape = {}".format(img.shape))
            exit()
        
        sample = feature.canny(img,sigma=1,high_threshold=150,low_threshold=50)

        lines = transform.probabilistic_hough_line(sample,line_length=50)
        
        # 각이 가장 크게 변하는 왼/오른쪽 직선을 찾기
        theta = []
        for line in lines:
            p1, p2 = line
            theta.append(np.arctan((p2[1]-p1[1])/(p2[0]-p1[0])))

        priority = np.argsort(theta)
        idx1,idx2 = priority[0],priority[-1]

        selected_lines = [lines[idx1],lines[idx2]]

        # 얻어진 직선으로부터 평균값을 지나는 직선 찾기
        l1 = selected_lines[0]
        l2 = selected_lines[1]

        s1,e1 = l1
        s2,e2 = l2

        if s1[0]-s2[0] <=8 and s1[0]-s2[0] <=8:
            return -1 # Case in which we got only a line.

        s_mid = ((s1[0]+s2[0])//2, (s1[1]+s2[1])//2)
        e_mid = ((e1[0]+e2[0])//2, (e1[1]+e2[1])//2)
        l_mid = s_mid, e_mid

        return l_mid

    def draw_line(self,p1,p2):
        '''
        p1 must be upper position relative to the p2
        '''
        # Check p1 is more upper than p2
        if not(p1[1] < p2[1]):
            p1, p2 = p2, p1 # Swap
        
        try:
            tan = self.tan(p1,p2)

            idx = []

            for x in range(p1[0],p2[0]):
                y = p1[1]+tan*(x-p1[0])
                idx.append([x,int(y)])
        except ZeroDivisionError:
            idx = []

            for y in range(p1[1],p2[1]):
                idx.append([p1[1],y])

        idx.append(p2)

        return idx

def figure_to_array(fig):
    """
    plt.figure를 RGBA로 변환(layer가 4개)
    shape: height, width, layer
    """
    fig.canvas.draw()
    rgba = fig.canvas.renderer.buffer_rgba()[:,:,:]
    y = 0.2125*rgba[..., 0] + 0.7154*rgba[..., 1] + 0.0721*rgba[..., 2]
    return y[..., np.newaxis]