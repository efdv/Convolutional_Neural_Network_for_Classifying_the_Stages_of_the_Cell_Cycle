# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:08:28 2023

@author: edgar
"""

from keras.models import load_model
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import math as mt

def normResults(y_pred):
    yp = []
    for row in y_pred:
        valmax = max(row)
        for i in range(len(row)):
            if row[i] == valmax:
                row[i] = 1
            else:
                row[i] = 0
        
        yp.append(row)
    
    return yp


#load model
name_model = 'model_CNN_3_2.h5'
model = load_model('../modelos/'+name_model)

# load data
data = np.load("../../../datasets/CellCycle-dataset-efdv/cellCycle.npy")
labels = np.load("../../../datasets/CellCycle-dataset-efdv/labels.npy")
root_graphs = "../graphs/training_test/original/"

data = data/255.0

skfolds = StratifiedKFold(n_splits=5)

num_fold = 0
for train_index, test_index in skfolds.split(data, labels):    
    if num_fold == 3:
        X_train_folds = data[train_index]
        y_train_folds = labels[train_index]
        X_test_folds  = data[test_index]
        y_test_folds  = labels[test_index]
    
        y_train_folds = to_categorical(y_train_folds)
        y_test_folds = to_categorical(y_test_folds)
        
        y_pred = model.predict(X_test_folds)
        
        ypred_cm = []
        y_test_folds_cm = []
        for i in range(len(y_pred)):
           ypred_cm.append(np.argmax(y_pred[i]))
           y_test_folds_cm.append(np.argmax(y_test_folds[i]))
          
        yp = [] 
        yp = normResults(y_pred)
    num_fold += 1 
    

# for i in ypred_cm
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x = [i for i in range(len(ypred_cm))]
ax.scatter(x, ypred_cm, x, marker='o')
plt.show()



d = mt.sqrt(x_2 - x_1)**2 + (y_2 2- y_1)**2
