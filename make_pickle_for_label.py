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
scaler = 3
resized_shape = (1640//scaler, 590//scaler)
y_train = []

for a_dir in dirs_data:
    fnames_data = os.listdir(path_data+a_dir)
    fnames_data.sort()
    for fname in fnames_data:
        label = Image.open(path_label+a_dir+'/'+fname.replace('.jpg','.png'),'r')
        label = np.array(label.resize(resized_shape),dtype='uint8')
        y_train.append(label)

        del(label)

# 저장
with open('tanuki_label.p','wb') as f :
    pickle.dump(y_train), f, protocol=4)