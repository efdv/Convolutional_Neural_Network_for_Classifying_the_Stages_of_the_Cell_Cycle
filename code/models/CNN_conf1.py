# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 13:56:37 2023|

@author: ef.duquevazquez
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

# def base_model(): cnn1
#     model = Sequential()
#     model.add(Conv2D(128, (5,5), input_shape= (64, 64, 3), activation='relu'))
#     model.add(MaxPooling2D())
#     model.add(Conv2D(64, (3,3), activation='relu'))
#     model.add(MaxPooling2D())
#     model.add(Dropout(0.2))
#     model.add(Flatten())
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(7, activation='softmax'))

#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#     return model

def base_model(): #cnn2
    model = Sequential()
    model.add(Conv2D(64, (5,5), input_shape= (64, 64, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model