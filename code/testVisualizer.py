# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:37:19 2023

@author: ef.duquevazquez
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import defaultdict
import visualkeras
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

def base_model(): 
    model = Sequential(name="My_CNN_Model")  # Agregamos un nombre al modelo
    model.add(Conv2D(64, (10,10), input_shape=(64, 64, 3), activation='relu', name="Conv2D_1"))
    model.add(MaxPooling2D(name="MaxPooling2D_1"))
    model.add(Conv2D(32, (5,5), activation='relu', name="Conv2D_2"))
    model.add(MaxPooling2D(name="MaxPooling2D_2"))
    model.add(Flatten(name="Flatten"))
    model.add(Dense(32, activation='relu', name="Dense_1"))
    model.add(Dropout(0.2, name="Dropout"))
    model.add(Dense(16, activation='relu', name="Dense_2"))
    model.add(Dense(7, activation='softmax', name="Output"))

    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

model = base_model()


#visulkeras
color_map = defaultdict(dict)
color_map[Conv2D]['fill'] = '#094F89'
color_map[MaxPooling2D]['fill'] = '#00a0b0'
color_map[Flatten]['fill'] = '#a0e3ea'
color_map[Dense]['fill'] = '#9999ff'
color_map[Dropout]['fill'] = '#3333ee'
# visualkeras.layered_view(model, to_file='modelo.png', 
#                          color_map=color_map, legend=True)
visualkeras.layered_view(model, to_file='modelo.png', color_map=color_map)

## plot model
# plot_model(model, to_file='model1.png', show_shapes=True, show_layer_names=True)

# # Visualizar la imagen
# img = plt.imread('model.png')
# plt.imshow(img)
# plt.axis('off')
# plt.show()




