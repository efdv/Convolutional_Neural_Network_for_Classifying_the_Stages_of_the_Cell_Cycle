# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 13:56:37 2023

@author: ef.duquevazquez
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import Adam

def base_model():
    model = Sequential()
    model.add(Conv2D(64, (10,10), input_shape= (64, 64, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (5,5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(7, activation='softmax'))

    optimizer = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
