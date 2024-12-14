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

#un Freeze the convolution layers, we just want to train the FC layers and to retrain the last convolution layers
conv_base.trainable = True
set_trainable = False

for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
for layer in conv_base.layers:
    print(layer.name,layer.trainable)
        
model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))



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

#it often tell that keep lr low while doing transfer learning
model.compile (optimizer = keras.aptimizers.RMSprop(lr=1e-5), loss = 'binary_crossentropy', metrics = ["accuracy"])
history = model.fit(train_ds,epochs = 100, validation_data = validation_ds)