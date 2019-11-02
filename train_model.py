# 인자 처리
import argparse
parser = argparse.ArgumentParser(description="Training tanukiNetv2", epilog='Improved by Hyeonho Shin,\nmotivated from https://github.com/mvirgo/MLND-Capstone')

parser.add_argument('--train',type=str, required=False, default="tanuki_train.p" ,help = 'Train pickle file name')
parser.add_argument('--test',type=str, required=False, default="tanuki_test.p", help = 'Test picle file name')
parser.add_argument('-w','--weights',type=str, required=False, default="tanukiNetv2.h5", help = 'Weights file name')
parser.add_argument('-str','--structure',type=str, required=False, default="tanukiNetv2.json", help = 'Model structure will be saved as this name')
parser.add_argument('-hist','--history',type=str, required=False, default='history.p', help = 'Train history file name')

args = parser.parse_args()

# 학습 관련 시작
import os
from PIL import Image
import numpy as np
import sys
import time
import pickle
import tanuki_ml

from sklearn.utils import shuffle

print("\n=========Training start=========\n")
print("  Model structure will be saved as [{}]".format(args.structure))
print("  Trained weights will be saved as [{}]".format(args.weights))
print("  Training history will be saved in [{}]".format(args.history))
print("  Get train data from [{}]".format(args.train))
print("  Get test data from [{}]".format(args.test))


start_total = time.time()

# 초기화
color_num = 3
scaler = 6
input_shape = (590//scaler, 1640//scaler, color_num)
resized_shape = (1640//scaler, 590//scaler)
batch_size = 32
epochs = 30
pool_size = (2, 2)

# Load training images
X_train, y_train, _ = pickle.load(open(args.train, "rb" ))
y_train = y_train[:, 1:-1,:-1, np.newaxis]/255.0

# Model generation
model = tanuki_ml.generate_model(input_shape, pool_size)
model.summary()

# Load test data
X_test, y_test, _ = pickle.load(open(args.test, "rb" ))
y_test = y_test[:, 1:-1,:-1, np.newaxis]/255.0

# Adaptive Learning rate 기능
callback_list = [
  tanuki_ml.AdaptiveLearningrate(threshold=0.01, decay=0.8, relax=3, verbose=1)
]

# 학습
start_train = time.time()
hist = model.fit(X_train, y_train, batch_size, epochs, validation_data = (X_test, y_test), shuffle = True, callbacks=callback_list)
end_train = time.time()

end_total = time.time()

# Weights 저장
model.save_weights(args.weights)
print("Saved model to disk")

# Model 구조 저장
model_json = model.to_json()
with open(args.structure, "w") as json_file :
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
with open(args.history,'wb') as f :
    pickle.dump(hist.history, f, protocol=4)