# 각 폴더에 접속

import os
from PIL import Image
import tanuki_ml
import numpy as np

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

output_path = "/home/mary/ml/reinforced_imgs"

# train 폴더 진입
# directory 목록 저장.
root = '/home/mary/ml/train'
print('Root dir is {}'.format(root))
dirs = os.listdir(root + '/label')
dirs.sort()

for i, target_folder in enumerate(dirs):
    ###### .MP4 폴더 진입
    # 초기화
    print("Start to reinforce imgs in {}".format(target_folder))

    fnames = os.listdir(root+'/label/'+target_folder)
    fnames.sort()
    # label내의 
    for fname in fnames:
        # 파일 하나 로드
        full_fname = root+'/label/'+target_folder+'/'+fname
        tmp_img = Image.open(full_fname)

        # 가공
        tmp_arr = np.array(tmp_img)
        np.place(tmp_arr, tmp_arr>=2, [255])

        # 기록
        img_to_write = Image.fromarray(tmp_arr)
        img_to_write.save(output_path+'/train/label/'+target_folder+'/'+fname)

        del(tmp_img)
        del(tmp_arr)
        del(img_to_write)

# test 폴더 진입
root = '/home/mary/ml/test'

print('Root dir is {}'.format(root))
dirs = os.listdir(root + '/label')
dirs.sort()

for i, target_folder in enumerate(dirs):
    #.MP4 폴더 진입

    # 초기화
    print("Start to reinforce imgs in {}".format(target_folder))

    fnames = os.listdir(root+'/label/'+target_folder)
    fnames.sort()
    # label내의 
    for fname in fnames:
        # 파일 하나 로드
        full_fname = root+'/label/'+target_folder+'/'+fname
        tmp_img = Image.open(full_fname)

        # 가공
        tmp_arr = np.array(tmp_img)
        np.place(tmp_arr, tmp_arr>=2, [255])

        # 기록
        img_to_write = Image.fromarray(tmp_arr)
        img_to_write.save(output_path+'/test/label/'+target_folder+'/'+fname)

        del(tmp_img)
        del(tmp_arr)
        del(img_to_write)