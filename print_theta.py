# 인자 처리
import argparse
parser = argparse.ArgumentParser(description="주어진 비디오로부터 진행방향 모음", epilog='Improved by Hyeonho Shin,\nmotivated from https://github.com/mvirgo/MLND-Capstone')

parser.add_argument('-i','--input',type=str, required=False, default="challenge_video.mp4" ,help = 'Input file name')
parser.add_argument('-o','--output',type=str, required=False, default="output.mp4", help = 'Output file name')
parser.add_argument('-f','--frames',type=int, required=False, default=15, help = 'Number of memorized frames')

args = parser.parse_args()

print("\nDetect lanes in [{}], and then generate output video file in [{}]\n".format(args.input, args.output))

import tanuki_ml
import warnings
warnings.filterwarnings(action='ignore') 

json_fname = "tanukiNetv2.json"
weights_fname ="tanukiNetv2.h5"

lanes = tanuki_ml.Lanes(args.frames, json_fname, weights_fname, scaler=6)
lanes.return_theta(args.input, args.output)