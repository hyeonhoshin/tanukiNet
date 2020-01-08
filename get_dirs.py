# 인자 처리
import argparse
parser = argparse.ArgumentParser(description="주어진 비디오로부터 진행방향 모음", epilog='Improved by Hyeonho Shin,\nmotivated from https://github.com/mvirgo/MLND-Capstone')

parser.add_argument('-i','--input',type=str, required=False, default="challenge_video.mp4" ,help = 'Input file name')
parser.add_argument('-o','--output',type=str, required=False, default="output.mp4", help = 'Output file name')
parser.add_argument('-f','--frames',type=int, required=False, default=15, help = 'Number of memorized frames')

args = parser.parse_args()

print("\nDetect lanes in [{}], and then generate output video file in [{}]\n".format(args.input, args.output))

import tanuki_ml
import numpy as np
import cv2
from PIL.Image import fromarray, BILINEAR
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from keras.models import model_from_json
import sys
import warnings
import time

warnings.filterwarnings(action='ignore') # 귀찮은 경고 감추기

scaler = 6
resized_shape = (1640//scaler, 590//scaler)

json_fname = "tanukiNetv2.json"
weights_fname ="tanukiNetv2.h5"

# Load Keras model
json_file = open(json_fname, 'r')
json_model = json_file.read()
json_file.close()
model = model_from_json(json_model)
model.load_weights(weights_fname)
model.summary()

scaler = 6
resized_shape = (1640//scaler, 590//scaler)

save = args.frames

vid_output = args.output
clip1 = VideoFileClip(args.input)
original_size = clip1.size

path_toolbox = tanuki_ml.path_determiner()

# Class to average lanes with
class Lanes():
    def __init__(self, weights = np.log(np.arange(2,save+2))):
        self.recent_fit = []
        self.avg_fit = []
        self.weights = weights
        self.theta = []

lanes = Lanes()

def road_lines(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """
    # Get image ready for feeding into model
    small_img = fromarray(image).resize(resized_shape)
    small_img = np.array(small_img,dtype="uint8")
    small_img = small_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > save:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    if len(lanes.recent_fit) == save:
        lanes.avg_fit = np.average(np.array([i for i in lanes.recent_fit]), axis = 0, weights=lanes.weights)
    else:
        lanes.avg_fit = np.average(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Calculate theta
    path = path_toolbox.approx_path(lanes.avg_fit)
    theta_line = path_toolbox.draw_line(path[0],path[-1])

    blanks = np.zeros_like(theta_line)
    lane_drawn = np.dstack((blanks, theta_line, blanks))
    lane_drawn = lane_drawn.astype("uint8")

    # Re-size to match the original image
    lane_image = fromarray(lane_drawn)
    lane_image = lane_image.resize(original_size,BILINEAR)
    lane_image = np.asarray(lane_image,dtype="uint8")

    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    return result

