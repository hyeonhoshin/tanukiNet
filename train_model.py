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
color_num = 3
scaler = 6
input_shape = (590//scaler, 1640//scaler, color_num)
resized_shape = (1640//scaler, 590//scaler)
batch_size = 32
epochs = 20
pool_size = (2, 2)

print("Training start")

# Load training images
X_train, y_train, _ = pickle.load(open("tanuki_train.p", "rb" ))
y_train = y_train[:, 1:-1,-1, np.newaxis]

# Model generation
model = tanuki_ml.generate_model(input_shape, pool_size)
model.compile(optimizer='Adam', loss='mean_squared_error')
model.summary()

# Data 부풀리기 - 일단 안함. Flip 정도는 후에 구현

# 학습
start_train = time.time()
model.fit(X_train, y_train, batch_size, epochs, shuffle = True)
end_train = time.time()

end_total = time.time()

# Weights 저장
model.save_weights("model.h5")
print("Saved model to disk")

# Model 구조 저장
model_json = model.to_json()
with open("model.json", "w") as json_file :
    json_file.write(model_json)

## 실험 데이터 저장
f=open("lstm_train_result_mem_is_{}.txt",'w')

# 총 걸린 시간
min, sec = divmod(end_total-start_total, 60)
f.write("Total run time : {}min {}sec\n".format(int(min),int(sec)))

# 총 걸린 학습 시간
min, sec = divmod(end_train-start_train, 60)
f.write("Pure training time : {}min {}sec\n".format(int(min),int(sec)))

f.close()
