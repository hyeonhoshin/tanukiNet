'''
모델을 생성하고 학습시킨다.
폴더 하나씩에 대해서 학습시킨다.

python train_model_with_mem 5
'''

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
batch_size = 50
epochs = 20
pool_size = (2, 2)

print("Training model with memory size =", memory_size)
print("Final data will be written in", "mem_is_{}.h5".format(memory_size))

# Load training images
X_train, y_train, fnames = pickle.load(open("tanuki_train.p", "rb" ))

# Normalize labels - training images get normalized to start in the network
y_train = y_train/255
y_train = y_train[:,1:-1,:-1, np.newaxis] # 차원 조정

# Give time data
# First step, Find boundary index
boundary = [0]
for i, e in enumerate(fnames):
    try:
        if fnames[i] != fnames[i+1]:
            boundary.append(i)
    except:
        break

# Second step, calculate separate timed matrix and combine
X_train_temp = []
y_train_temp = []

for i, e in enumerate(boundary):
    first = boundary[i]
    try:
        second = boundary[i+1]
    except:
        break
    X_t, y_t = tanuki_ml.give_time(X_train[first:second],y_train[first:second], memory_size = memory_size)
    X_train_temp.append(X_t)
    y_train_temp.append(y_t)

X_train_t = np.array(X_train_temp)
y_train_t = np.array(y_train_temp)

del(X_train_temp)
del(y_train_temp)

print("X_train_t is {}, y_train_t is {}".format(X_train_t.shape,y_train_t.shape))
print("Element of X_train_ is {}".format(X_train_t[0].shape))

# Model generation
model = tanuki_ml.generate_model(input_shape, pool_size)
model.compile(optimizer='Adam', loss='mean_squared_error')
model.summary()

# Data 부풀리기 - 일단 안함. Flip 정도는 후에 구현

# 학습
start_train = time.time()
model.fit(X_train_t, y_train_t, batch_size, epochs, shuffle = True)
end_train = time.time()

end_total = time.time()

# Weights 저장
model.save_weights("mem_is_{}.h5".format(memory_size))
print("Saved model to disk")

# Model 구조 저장
model_json = model.to_json()
with open("model_structure_when_mem_is_{}.json".format(memory_size), "w") as json_file :
    json_file.write(model_json)

## 실험 데이터 저장
f=open("train_result_mem_is_{}.txt".format(memory_size),'w')

# 총 걸린 시간
min, sec = divmod(end_total-start_total, 60)
f.write("Total run time : {}min {}sec\n".format(int(min),int(sec)))

# 총 걸린 학습 시간
min, sec = divmod(end_train-start_train, 60)
f.write("Pure training time : {}min {}sec\n".format(int(min),int(sec)))

f.close()
