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
import time

start_total = time.time()

# 초기화
memory_size = 3
color_num = 3
scaler = 3
input_shape = (590//scaler, 1640//scaler, color_num)
resized_shape = (1640//scaler, 590//scaler)
batch_size = 35
epochs = 2
pool_size = (2, 2)

model = tanuki_ml.generate_prior_model(input_shape, pool_size)
model.compile(optimizer='Adam', loss='mean_squared_error', metrics = ['accuracy'])
model.load_weights("prior_model.h5","r")

X_test, y_test = tanuki_ml.read_set('/home/mary/ml/test', resized_shape)
y_test = y_test[..., np.newaxis]

print("Dimension of y_test is {}".format(y_test.shape))

start_test = time.time()
loss_and_metrics = model.evaluate(X_test, y_test[:,1:-1, 1:-1], batch_size, verbose = 1)
end_test = time.time()

print('Loss is {:.5f}, Accuracy is {:.5f}%'.format(loss_and_metrics[0],loss_and_metrics[1]*100))

end_total = time.time()

## 실험 데이터 저장
f=open("Prior_model_test_result.txt".format(memory_size),'w')

# 총 걸린 시간
min, sec = divmod(end_total-start_total, 60)
f.write("Total run time : {}min {}sec\n".format(min,sec))

# 총 걸린 평가 시간
min, sec = divmod(end_test-start_test, 60)
f.write("Pure test time : {}min {}sec\n".format(min,sec))

# Loss and Accuracy
f.write('Loss is {}, Accuracy is {}%\n'.format(loss_and_metrics[0],loss_and_metrics[1]*100))

f.close()
