import os
from PIL import Image
import numpy as np

scaler = 3
resized_shape = (1640//scaler, 590//scaler)

target = '/Users/tanukimong/ML/capstone/train/label/05312327_0001.MP4/00000.png'

img = Image.open(target)
img = img.resize(resized_shape)

import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()

print(np.asarray(img))