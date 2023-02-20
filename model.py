import numpy as np
import pandas as pd
from tensorflow.python import keras

from tensorflow.python.keras.applications.resnet import preprocess_input
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.text_dataset import text_dataset_from_directory
size=224
datagen=ImageDataGenerator(preprocessing_function=preprocess_input,)

train=datagen.flow_from_directory('/home/rogue/Music/digits/digi/',target_size=(size,size),batch_size=32,class_mode='categorical')
validate=datagen.flow_from_directory('/home/rogue/Music/digits/val/',target_size=(size,size),batch_size=32,class_mode='categorical')


model=Sequential()
model.add(Conv2D(20,kernel_size=(3,3),activation='relu',input_shape=(size,size,3)))
model.add(Conv2D(20,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(20,kernel_size=(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit_generator(train,steps_per_epoch=20,epochs=50,validation_data=validate,validation_steps=800)
model.save("/home/rogue/Music/digits.h5")
