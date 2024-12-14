import pandas as pd
import numpy as np
import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Flatten
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array,load_img

conv_base = VGG16(
weights = 'imagenet',
include_top = False,  # remove FC layers
input_shape = (150,150,3)
)

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))

#Freeze the convolution layers, we just want to train the FC layers not to retrain the convolution layers
conv_base.trainable = False

#DataGenerator
batch_size = 32
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizental_flip = True
)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
 directory = 'path of train image directory',
 target_size = (150,15),
 batch_size = batch_size,
 class_mode = 'binary'
)

validation_generator = test_datagen.flow_from_directory(
 directory = 'path of train image directory',
 target_size = (150,15),
 batch_size = batch_size,
 class_mode = 'binary'
)
 
model.compile (optimizer = 'adam', loss = 'binary_crossentropy', metrics = ["accuracy"])
history = model.fit_generator(train_generator,epochs = 100, validation_data = validation_generator)
