""" This file contains code for a fully convolutional
(i.e. contains zero fully connected layers) neural network
for detecting lanes. This version assumes the inputs
to be road images in the shape of 80 x 160 x 3 (RGB) with
the labels as 80 x 160 x 1 (just the G channel with a
re-drawn lane). Note that in order to view a returned image,
the predictions is later stacked with zero'ed R and B layers
and added back to the initial road image.
"""

import os
from PIL import Image
import tanuki_ml
import numpy as np

# 초기화
memory_size = 3
color_num = 3
scaler = 3
input_shape = (590//scaler, 1640//scaler, color_num)
resized_shape = (1640//scaler, 590//scaler)
batch_size = 10
epochs = 2
pool_size = (2, 2)

model = tanuki_ml.generate_prior_model(input_shape, pool_size)
model.compile(optimizer='Adam', loss='mean_squared_error', metrics = ['accuracy'])
model.load_weights("full_CNN_model.h5","r")

X_test, y_test = tanuki_ml.read_set('/home/mary/ml/test', resized_shape)

loss_and_metrics = model.evaluate(X_test, y_test[..., np.newaxis], batch_size, verbose = 1)

print('Loss is {:3f}, Accuracy is {:3f}'.format(loss_and_metrics[0],loss_and_metrics[1]*100))
