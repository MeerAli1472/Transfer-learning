import pandas as pd
import numpy as np
import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Flatten
from keras.applications.vgg16 import VGG16

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

#Generator to train the model on baches
train_ds = keras.utils.image_dataset_from_directory(
 directory = 'path of train image directory',
 labels = 'infered',
 label_mode = 'int',
 batch_size = 32,
 image_size = (150,15)
)

validation_ds = keras.utils.image_dataset_from_directory(
 directory = 'path of validation image directory',
 labels = 'infered',
 label_mode = 'int',
 batch_size = 32,
 image_size = (150,15)
 
 def process(image,label):
	image = tensorflow.cast(image/255., tensorflow.float32)
	return image,label
train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

model.compile (optimizer = 'adam', loss = 'binary_crossentropy', metrics = ["accuracy"])
history = model.fit(train_ds,epochs = 100, validation_data = validation_ds)
