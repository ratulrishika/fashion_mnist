# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 00:14:20 2020

@author: Ratul
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Flatten,Conv2D,MaxPooling2D,Dropout,Dense
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
y_train.max()
x_test.shape
plt.imshow(x_train[5])
y_train[5]
Y_train=to_categorical(y_train)
Y_test=to_categorical(y_test)
X_train=x_train.reshape(60000,28,28,1)
X_test=x_test.reshape(10000,28,28,1)
X_train.shape
y_train.shape
X_test.shape
y_test.shape

X_train.max()
y_train
X_train=X_train/255
X_test=X_test/255

model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=[28,28,1]))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=64,epochs=10,validation_data=[X_test,Y_test])

