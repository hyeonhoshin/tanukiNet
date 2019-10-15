# File open
## Memory 절약을 위해 중간 평가는 Test set이 아니라, Validation split으로 진행
import os
from PIL import Image
import numpy as np
import tanuki_ml
import sys
import time
import pickle

from sklearn.utils import shuffle
start_total = time.time()

# 초기화
memory_size = int(sys.argv[1])
color_num = 3
scaler = 6
input_shape = (memory_size, 590//scaler, 1640//scaler, color_num)
resized_shape = (1640//scaler, 590//scaler)
batch_size = 5
epochs = 20
pool_size = (2, 2)

print("Training model with memory size =", memory_size)

# Load training images
X_train, y_train, fnames = pickle.load(open("tanuki_train.p", "rb" ))

# Normalize labels - training images get normalized to start in the network
y_train = y_train/255
y_train = y_train[:,1:-1,:-1] # 차원 조정

# Give time data
# First step, Find boundary index
boundary = [0]
for i in range(0,len(fnames)-1):
    if fnames[i] != fnames[i+1]:
        boundary.append(i)

# Second step, calculate separate timed matrix and combine
# do - Make first array, i = 0

first = boundary[0]
second = boundary[1]
X_train_t, y_train_t = tanuki_ml.give_time(X_train[first:second],y_train[first:second], memory_size = memory_size)

for i in range(1, len(boundary)-1):
    first = boundary[i]
    second = boundary[i+1]
    X_t, y_t = tanuki_ml.give_time(X_train[first:second],y_train[first:second], memory_size = memory_size)
    X_train_t = np.append(X_train_t, X_t, axis=0)
    y_train_t = np.append(y_train_t, y_t, axis=0)

print("X_train_t is {}, y_train_t is {}".format(X_train_t.shape,y_train_t.shape))
print("Element of X_train_ is {}".format(X_train_t[0].shape))


print("Make pickle in train_mem_{}.p".format(memory_size))

with open('train_mem_{}.p'.format(memory_size),'wb') as f :
        pickle.dump((X_train_t, y_train_t), f, protocol=4)
    
print("Pickle is enougly cooked!")