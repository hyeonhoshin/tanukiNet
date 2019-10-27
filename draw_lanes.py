'''
draw_lanes.py memory_size input_video output_video 
'''

import numpy as np
import cv2
from PIL.Image import fromarray, BILINEAR
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from keras.models import model_from_json
import sys
import warnings
warnings.filterwarnings(action='ignore') # 귀찮은 경고 감추기

scaler = 6
resized_shape = (1640//scaler, 590//scaler)

memory_size = int(sys.argv[1])
json_fname = "tanukiNetv1.json".format(memory_size)
weights_fname ="tanukiNetv1.h5".format(memory_size)

# Load Keras model
json_file = open(json_fname, 'r')
json_model = json_file.read()
json_file.close()
model = model_from_json(json_model)
model.load_weights(weights_fname)

model.summary()

# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_question = np.empty((1, 96, 272, 1))
        self.initialized = False
        self.recent_ans = []
        self.avg_ans = []

def road_lines(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """

    # Image를 memory size 만큼 받아서 한번에 predict
    small_img = fromarray(image).resize(resized_shape)
    small_img = np.asarray(small_img,dtype="uint8")
    small_img = small_img[None,:,:,:]/255.0 # (1, 96, 272, 1)

    if lanes.recent_question.shape[0] >= memory_size:
        # 이 경우에만 예측과 갈아치우기를 한다.
        # 이전 프레임 지우기
        lanes.recent_question = np.append(lanes.recent_question, small_img, axis=0)
        lanes.recent_question = lanes.recent_question[1:]
        prediction = model.predict(lanes.recent_question[np.newaxis])[0]*255

        lanes.recent_ans.append(prediction)

        if len(lanes.recent_ans) > 5:
            lanes.recent_ans = lanes.recent_ans[1:]

        # Calculate average detection
        lanes.avg_ans = np.mean(np.array([i for i in lanes.recent_ans]), axis = 0)

        # Generate fake R & B color dimensions, stack with G
        blanks = np.zeros_like(lanes.avg_ans)
        lane_drawn = np.dstack((blanks, lanes.avg_ans, blanks))
        lane_drawn = lane_drawn.astype("uint8")

        # Re-size to match the original image
        lane_image = fromarray(lane_drawn)
        lane_image = lane_image.resize((1280, 720),BILINEAR)
        lane_image = np.asarray(lane_image,dtype="uint8")

        # Merge the lane drawing onto the original image
        result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    elif lanes.initialized == True:
        print("=== Case 1 : image stacking only ===")
        lanes.recent_question = np.append(lanes.recent_question, small_img, axis=0)
        result = fromarray(image).resize((1280, 720))
        result = np.array(result)

    elif lanes.initialized == False:
        print("=== Case 2 : initializing ===")
        lanes.recent_question = small_img# (1, 96, 272, 1)
        result = fromarray(image).resize((1280, 720))
        result = np.array(result) # (720, 1280, 3) 
        lanes.initialized = True

    return result

# Global variable lanes.recent_question
lanes = Lanes()

# Where to save the output video
vid_output = sys.argv[3]

# Location of the input video
clip1 = VideoFileClip(sys.argv[2])

vid_clip = clip1.fl_image(road_lines)
vid_clip.write_videofile(vid_output, audio=False)
