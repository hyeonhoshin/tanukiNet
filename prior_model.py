import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os

# Import necessary items from Keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Deconvolution2D, Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import tanuki_ml

# Load training images
train_images,labels, _ = pickle.load(open("tanuki_train.p", "rb" ))

# Make into arrays as the neural network wants these
labels = labels[:,1:-1,:-1, np.newaxis]

print("Train_imgs is {}, labels is {}".format(train_images.shape, labels.shape))

# Normalize labels - training images get normalized to start in the network
labels = labels / 255

# Shuffle images along with their labels, then split into training/validation sets
X_train, y_train = shuffle(train_images, labels)

# Batch size, epochs and pool size below are all paramaters to fiddle with for optimization
batch_size = 100
epochs = 20
pool_size = (2, 2)
input_shape = X_train.shape[1:]

model = tanuki_ml.unet(input_size = input_shape)
model.summary()
    
# Save model architecture and weights
model_json = model.to_json()
with open("tanuki_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights('tanuki_model.h5')