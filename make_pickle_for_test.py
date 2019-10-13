import pickle
from PIL import Image
import numpy as np
import os

# 불러오기
path_data = '/home/mary/ml/test/data/'
path_label = '/home/mary/ml/reinforced_imgs/test/label/'
dirs_data = os.listdir(path_data)
dirs_data.sort()

# 처리
scaler = 6
resized_shape = (1640//scaler, 590//scaler)

X_test = []
y_test = []
folder_name = []

for a_dir in dirs_data:
    fnames_data = os.listdir(path_data+a_dir)
    for fname in fnames_data:
        data = Image.open(path_data+a_dir+'/'+fname,'r')
        label = Image.open(path_label+a_dir+'/'+fname.replace('.jpg','.png'),'r')

        data = np.array(data.resize(resized_shape),dtype='uint8')
        label = np.array(label.resize(resized_shape),dtype='uint8')

        X_test.append(data)
        y_test.append(label)
        folder_name.append(a_dir)

        del(data)
        del(label)

X_test = np.array(X_test, dtype='uint8')
y_test = np.array(y_test, dtype='uint8')

# 저장
with open('tanuki_test.p','wb') as f :
    pickle.dump((X_test, y_test, folder_name), f, protocol=4)