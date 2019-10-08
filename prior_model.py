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

# 초기화
memory_size = 3
color_num = 3
scaler = 3
input_shape = (590//scaler, 1640//scaler, color_num)
resized_shape = (1640//scaler, 590//scaler)
batch_size = 10
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
        tmp_arr = np.array(tmp_img.resize(resized_shape), dtype='uint8')
        questions.append(tmp_arr)
        del(tmp_img)
        del(tmp_arr)
    X_train = np.array(questions)

    # y_train load
    fnames = os.listdir(root + '/label/' + target_folder)
    fnames.sort()
    for fname in fnames:
        full_fname = root+'/label/'+target_folder+'/'+fname
        tmp_img = Image.open(full_fname)
        tmp_arr = np.array(tmp_img.resize(resized_shape), dtype='uint8')
        answers.append(tmp_arr[1:-1,:])
        del(tmp_img)
        del(tmp_arr)
    y_train = np.array(answers)[... ,np.newaxis]
    print('X_train is {}, and y_train is {}'.format(X_train.shape, y_train.shape))

    # 학습
    hist = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)

    del(X_train)
    del(y_train)
    # 끝나면 free하고 다른 디렉토리에서 데이터 불러옴

# h5 파일 출력
model.save_weights('full_CNN_model.h5')
print("Saved model to disk")
