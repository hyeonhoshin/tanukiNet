'''
기존 모델을 열어 추가로 학습을 진행한다.

add_epoch.py (memorysize) (추가할 epoch수)

'''

import numpy as np
import cv2
from PIL.Image import fromarray, BILINEAR
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from keras.models import model_from_json
import sys
import warnings
import pickle
warnings.filterwarnings(action='ignore') # 귀찮은 경고 감추기

scaler = 6
resized_shape = (1640//scaler, 590//scaler)

memory_size = int(sys.argv[1])
epochs = int(sys.argv[2])
json_fname = "model_structure_when_mem_is_{}.json".format(memory_size)
weights_fname ="mem_is_{}.h5".format(memory_size)
batch_size = 10

# Load training images
X_train_t, y_train_t = pickle.load(open("train_mem_{}.p".format(memory_size), "rb" ))

# Load Keras model
json_file = open(json_fname, 'r')
json_model = json_file.read()
json_file.close()
model = model_from_json(json_model)
model.load_weights(weights_fname)

model.summary()

model.fit(X_train_t, y_train_t, batch_size, epochs, shuffle = True)

# Weights 저장
model.save_weights("mem_is_{}_add_{}.h5".format(memory_size,epochs))
print("Saved model to disk")

# Model 구조 저장
model_json = model.to_json()
with open("model_structure_when_mem_is_{}_add_{}.json".format(memory_size,epochs), "w") as json_file :
    json_file.write(model_json)