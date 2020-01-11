import numpy as np
import os
from PIL import Image

# Import necessary items from Keras
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

from keras.models import Model, model_from_json
from keras.layers import Conv2D, Input, MaxPooling2D, Dropout, UpSampling2D,Conv2DTranspose
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.callbacks import Callback

from metrics import iou_loss_core, competitionMetric2, K
from attention_module import attach_attention_module

from skimage import feature, transform
from skimage.draw import line_aa

from PIL.Image import fromarray, BILINEAR

from moviepy.editor import VideoFileClip

import cv2

diff_th = 8
rad_to_deg = 180/np.pi

# Class to average lanes with
class Lanes():
    def __init__(self, save, json_fname, weights_fname, scaler):
        self.recent_fit = []
        self.avg_fit = []
        self.weights = np.log(np.arange(2,save+2))
        self.recent_path = []
        self.avg_path = []
        self.resized_shape=(1640//scaler, 590//scaler)
        self.save = save

        self.model = self._read_model(json_fname,weights_fname)

    def _road_lines(self, image):
        """ Takes in a road image, re-sizes for the model,
        predicts the lane to be drawn from the model in G color,
        recreates an RGB image of a lane and merges with the
        original road image.
        """
        # Get image ready for feeding into model
        s, e = self._get_line_positions(image)

        rr,cc,val=line_aa(s[0],s[1],e[0],e[1])
        line_img = np.zeros_like(self.avg_fit[..., 0])
        line_img[cc, rr] = val * 255

        blanks = np.zeros_like(self.avg_fit)
        lane_drawn = np.dstack((blanks, line_img, blanks))
        lane_drawn = lane_drawn.astype("uint8")

        # Re-size to match the original image
        #lane_image = cv2.filter2D(lane_drawn,-1,HPF)
        lane_image = fromarray(lane_drawn)
        lane_image = lane_image.resize(self.original_size,BILINEAR)
        lane_image = np.asarray(lane_image,dtype="uint8")

        # Merge the lane drawing onto the original image
        result = cv2.addWeighted(image, 1, lane_image, 1, 0)

        return result

    def _get_line_positions(self, image):
        # Get image ready for feeding into model
        small_img = fromarray(image).resize(self.resized_shape)
        small_img = np.array(small_img,dtype="uint8")
        small_img = small_img[None,:,:,:]

        # Make prediction with neural network (un-normalize value by multiplying by 255)
        prediction = self.model.predict(small_img)[0] * 255

        # Add lane prediction to list for averaging
        self.recent_fit.append(prediction)
        # Only using last five for average
        if len(self.recent_fit) > self.save:
            self.recent_fit = self.recent_fit[1:]

        # Calculate average detection
        if len(self.recent_fit) == self.save:
            self.avg_fit = np.average(np.array([i for i in self.recent_fit]), axis = 0, weights=self.weights)
        else:
            self.avg_fit = np.average(np.array([i for i in self.recent_fit]), axis = 0)

        # Calculate theta
        path = self._approx_path(self.avg_fit)

        if path != -1: # 선이 한 개가 아닐때만 Update
            self.recent_path.append(path)
            if len(self.recent_path) > self.save:
                self.recent_path = self.recent_path[1:]

            # Calculate average theta
            if len(self.recent_path) == self.save:
                self.avg_path = np.average(np.array([i for i in self.recent_path]), axis = 0, weights=self.weights)
            else:
                self.avg_path = np.average(np.array([i for i in self.recent_path]), axis = 0)

            self.avg_path = np.asarray(self.avg_path, dtype=np.int32)

        s, e = self.avg_path
        return s, e

    def _read_model(self, json_fname, weights_fname):
        # Load Keras model
        json_file = open(json_fname, 'r')
        json_model = json_file.read()
        json_file.close()

        model = model_from_json(json_model)
        model.load_weights(weights_fname)
        model.summary()

        return model
    def _approx_path(self,img):

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

        if abs(s1[0]-s2[0]) <=diff_th and abs(s1[0]-s2[0]) <=diff_th:
            return -1 # Case in which we got only a line.

        s_mid = ((s1[0]+s2[0])//2, (s1[1]+s2[1])//2)
        e_mid = ((e1[0]+e2[0])//2, (e1[1]+e2[1])//2)
        l_mid = s_mid, e_mid

        return l_mid

    def write_output_video(self, input_path, output_path):
        vid_output = output_path
        clip1 = VideoFileClip(input_path)
        self.original_size = clip1.size

        vid_clip = clip1.fl_image(self._road_lines)
        vid_clip.write_videofile(vid_output, audio=False)

    def return_theta(self,input_path,output_path):
        vid_output = output_path
        clip1 = VideoFileClip(input_path)

        self.original_size = clip1.size

        total_frames = clip1.duration*clip1.fps
        total_frames = int(total_frames)

        self.spf = 1/clip1.fps

        for current_frame in range(total_frames):
            current_time = current_frame*self.spf
            frame = clip1.get_frame(current_time)
            theta = self._road_thetas(frame)

            print("[Frame {}] : {}".format(current_frame, theta))

    def _road_thetas(self, image):
        s, e = self._get_line_positions(image)

        tan = (e[1]-s[1])/(e[0]-s[0])
        theta = np.arctan(tan-np.pi/2)*rad_to_deg

        return theta



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