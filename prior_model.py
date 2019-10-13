""" This file contains code for a fully convolutional
(i.e. contains zero fully connected layers) neural network
for detecting lanes. This version assumes the inputs
to be road images in the shape of 80 x 160 x 3 (RGB) with
the labels as 80 x 160 x 1 (just the G channel with a
re-drawn lane). Note that in order to view a returned image,
the predictions is later stacked with zero'ed R and B layers
and added back to the initial road image.
"""

import os
from PIL import Image
import tanuki_ml

import numpy as np
import time

start_total = time.time()
train_total = 0

# 초기화
color_num = 3
scaler = 3
input_shape = (590//scaler, 1640//scaler, color_num)
resized_shape = (1640//scaler, 590//scaler)
batch_size =30
epochs = 2
pool_size = (2, 2)

model = tanuki_ml.generate_prior_model(input_shape, pool_size)
model.compile(optimizer='Adam', loss='mean_squared_error')
model.summary()

# train 폴더 진입
# directory 목록 저장.
root = '/home/mary/ml/train'
dirs = os.listdir(root + '/data')
dirs.sort()

# 맥북용 더미 파일은 세지 않기.
if dirs[0] == '.DS_Store':
    del dirs[0]

# 그 중 한 디렉토리에서 데이터 불러옴.
for i, target_folder in enumerate(dirs):
    # 초기화
    questions = []
    answers = []
    print("Start to train model with files in {}".format(target_folder))

    fnames = os.listdir(root+'/data/'+target_folder)
    fnames.sort()
    # Train question imgs load
    for fname in fnames:
        full_fname = root+'/data/'+target_folder+'/'+fname
        tmp_img = Image.open(full_fname)
        tmp_arr = np.array(tmp_img.resize(resized_shape))
        questions.append(tmp_arr)
        del(tmp_img)
        del(tmp_arr)
    X_train = np.array(questions)/255.0

    # y_train load
    fnames = os.listdir(root + '/label/' + target_folder)
    fnames.sort()
    for fname in fnames:
        full_fname = '/home/mary/ml/reinforced_imgs/train/'+target_folder+'/'+fname
        tmp_img = Image.open(full_fname)
        tmp_arr = np.array(tmp_img.resize(resized_shape), dtype='uint8')
        answers.append(tmp_arr[2:-2,1:-1])
        del(tmp_img)
        del(tmp_arr)
    y_train = np.array(answers)[... ,np.newaxis]/255.0
    print('X_train is {}, and y_train is {}'.format(X_train.shape, y_train.shape))

    start_train = time.time()
    # 학습
    hist = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)
    end_train = time.time()
    train_total += end_train - start_train
    
    del(X_train)
    del(y_train)
    # 끝나면 free하고 다른 디렉토리에서 데이터 불러옴

end_total = time.time()
    
# h5 파일 출력
model.save_weights('prior_model.h5')
print("Saved model to disk")

# Model 구조 저장
model_json = model.to_json()
with open("prior_model_structure.json", "w") as json_file :
    json_file.write(model_json)

## 실험 데이터 저장
f=open("prior_model_train_result.txt",'w')

# 총 걸린 시간
min, sec = divmod(end_total-start_total, 60)
f.write("Total run time : {}min {}sec\n".format(min,sec))

# 총 걸린 학습 시간
min, sec = divmod(train_total, 60)
f.write("Pure training time : {}min {}sec\n".format(min,sec))

f.close()
