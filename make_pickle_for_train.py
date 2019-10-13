import pickle
from PIL import Image
import numpy as np
import os

# 불러오기
path_data = '/home/mary/ml/train/data/'
path_label = '/home/mary/ml/reinforced_imgs/train/label/'
dirs_data = os.listdir(path_data)
dirs_data.sort()

# 처리
scaler = 6
resized_shape = (1640//scaler, 590//scaler)

X_train = []

for a_dir in dirs_data:
    fnames_data = os.listdir(path_data+a_dir)
    fnames_data.sort()
    for fname in fnames_data:
        data = Image.open(path_data+a_dir+'/'+fname,'r')
        data = np.array(data.resize(resized_shape),dtype='uint8')
        X_train.append(data)
        del(data)

# 저장
with open('tanuki_train.p','wb') as f :
    pickle.dump((X_train), f, protocol=4)