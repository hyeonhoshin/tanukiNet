# File open
## Memory 절약을 위해 중간 평가는 Test set이 아니라, Validation split으로 진행
import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Import necessary items from Keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

from loss_functions import softmax_sparse_crossentropy_ignoring_last_label

import tanuki_ml

print("Training start")

# Load training images
train_images,labels, _ = pickle.load(open("tanuki_train.p", "rb" ))

input_shape = train_images.shape[1:]

# Make into arrays as the neural network wants these
labels = labels[..., np.newaxis]

print("Train_imgs is {}, labels is {}".format(train_images.shape, labels.shape))

# Normalize labels - training images get normalized to start in the network
labels = labels / 255

# Shuffle images along with their labels, then split into training/validation sets
train_images, labels = shuffle(train_images, labels)

# Test size may be 10% or 20%
X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1)

# Batch size, epochs and pool size below are all paramaters to fiddle with for optimization
batch_size = 128
epochs = 10
pool_size = (2, 2)
input_shape = X_train.shape[1:]

# Model generation
model = tanuki_ml.generate_model(input_shape, pool_size)
model.compile(optimizer='Adam', loss='mean_squared_error', metrics = ['accuracy'])
model.summary()

# 학습
model.fit(X_train, y_train, batch_size, epochs, shuffle = True, validation_data=(X_val, y_val))

# Using a generator to help the model use less data
# Channel shifts help with shadows slightly
datagen = ImageDataGenerator(channel_shift_range=0.2)
datagen.fit(X_train)

# Save model architecture and weights
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')