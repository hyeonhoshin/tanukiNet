# 인자 처리
import argparse
parser = argparse.ArgumentParser(description="주어진 비디오로부터 진행방향 모음", epilog='Improved by Hyeonho Shin,\nmotivated from https://github.com/mvirgo/MLND-Capstone')

parser.add_argument('-i','--input',type=str, required=False, default="challenge_video.mp4" ,help = 'Input file name')
parser.add_argument('-o','--output',type=str, required=False, default="output.mp4", help = 'Output file name')
parser.add_argument('-f','--frames',type=int, required=False, default=15, help = 'Number of memorized frames')

args = parser.parse_args()

print("\nDetect lanes in [{}], and then generate output video file in [{}]\n".format(args.input, args.output))

import tanuki_ml

from IPython.display import HTML
import sys
import warnings
import time
import matplotlib.pyplot as plt


'''
HPF = np.array([[-1,0,1,0,-1],
                [0,0,1,0,0],
                [1,1,1,1,1],
                [0,0,1,0,0],
                [-1,0,1,0,-1]])
                '''

# 귀찮은 경고 감추기
warnings.filterwarnings(action='ignore') 

json_fname = "tanukiNetv2.json"
weights_fname ="tanukiNetv2.h5"

start_eval = time.time() # Time check

# 얼마나 학습된 이미지의 배율(기본적으로는 6으로 학습되어 있음.)
scaler = 6

# Lane drawing using Lanes class
lanes = tanuki_ml.Lanes(save=args.frames,json_fname=json_fname, 
                        weights_fname=weights_fname, scaler=6)
lanes.run(args.input_path, args.output_path)

stop_eval = time.time() # Time check

# 총 걸린 시간
f=open("Estimate_theta_time.txt",'w')
min, sec = divmod(stop_eval-start_eval, 60)
f.write("Total run time : {}min {}sec\n".format(int(min),int(sec)))
f.close()