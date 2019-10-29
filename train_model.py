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
import sys
import time
import pickle
import tanuki_ml

from sklearn.utils import shuffle

start_total = time.time()

# 초기화
color_num = 3
scaler = 6
input_shape = (590//scaler, 1640//scaler, color_num)
resized_shape = (1640//scaler, 590//scaler)
batch_size = 32
epochs = 30
pool_size = (2, 2)

print("Training start")

# Load training images
X_train, y_train, _ = pickle.load(open("tanuki_train.p", "rb" ))
y_train = y_train[:, 1:-1,:-1, np.newaxis]/255.0

# Model generation
model = tanuki_ml.generate_model(input_shape, pool_size)
model.summary()

# Load test data
X_test, y_test, _ = pickle.load(open("tanuki_test.p", "rb" ))
y_test = y_test[:, 1:-1,:-1, np.newaxis]/255.0

# Adaptive Learning rate 기능
callback_list = [
  tanuki_ml.AdaptiveLearningrate(threshold=0.01, decay=0.8, relax=3, verbose=1)
]

# 학습
start_train = time.time()
hist = model.fit(X_train, y_train, batch_size, epochs, validation_data = (X_test, y_test), shuffle = True, callbacks=callback_list, verbose=2)
end_train = time.time()

end_total = time.time()

# Weights 저장
model.save_weights("tanukiNetv1.h5")
print("Saved model to disk")

# Model 구조 저장
model_json = model.to_json()
with open("tanukiNetv1.json", "w") as json_file :
    json_file.write(model_json)

## 실험 데이터 저장
f=open("train_result.txt",'w')

# 총 걸린 시간
min, sec = divmod(end_total-start_total, 60)
f.write("Total run time : {}min {}sec\n".format(int(min),int(sec)))

# 총 걸린 학습 시간
min, sec = divmod(end_train-start_train, 60)
f.write("Pure training time : {}min {}sec\n".format(int(min),int(sec)))

f.close()

# History 저장
with open('history.p','wb') as f :
    pickle.dump(hist.history, f, protocol=4)