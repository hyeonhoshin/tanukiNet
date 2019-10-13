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
folder_name = []

for a_dir in dirs_data:
    fnames_data = os.listdir(path_data+a_dir)
    fnames_data.sort()
    for fname in fnames_data:
        folder_name.append(a_dir)

# 저장
with open('tanuki_foldername.p','wb') as f :
    pickle.dump(folder_name), f, protocol=4)