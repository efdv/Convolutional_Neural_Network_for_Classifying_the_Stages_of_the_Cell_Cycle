from __future__ import print_function
import time

import numpy as np
import pandas as pd


from sklearn.manifold import TSNE

#matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

from keras.utils import to_categorical
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold

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


data = np.load("../../../datasets/cellcycle_dataset_ch4/datasetRana1/cellCycle.npy")
labels = np.load("../../../datasets/cellcycle_dataset_ch4/datasetRana1/labels.npy")
data = data/255.0

skfolds = StratifiedKFold(n_splits=5)

num_fold = 0
for train_index, test_index in skfolds.split(data, labels):
    
    X_train_folds = data[train_index]
    y_train_folds = labels[train_index]
    X_test_folds  = data[test_index]
    y_test_folds  = labels[test_index]
    
    y_train_folds = to_categorical(y_train_folds)
    y_test_folds = to_categorical(y_test_folds)

    model = load_model('E:/OneDrive - Universidad de Guanajuato/EF-Duque-Vazquez-Doctorado/projects/cell_cycle/modelos/model_CNN_with_wgandiv_3_2.h5')

    y_pred = model.predict(X_test_folds)
    
    ypred_cm = []
    y_test_folds_cm = []
    for i in range(len(y_pred)):
        ypred_cm.append(np.argmax(y_pred[i]))
        y_test_folds_cm.append(np.argmax(y_test_folds[i]))
        
    yp = [] 
    yp = normResults(y_pred)
    
    results = []
    n_correct = sum(yp == y_test_folds)
    results.append(n_correct/len(y_pred))
X_index = [i for i in range(13790)]
X_index = np.array(X_index)
yp_mod = [np.where(i == 1) for i in yp ]
yp_mod = np.array(yp_mod)


tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(yp_mod[:,0])

plt.figure(figsize=(10, 6))
plt.scatter(X_index[:], yp_mod[:,0])
plt.title('t-SNE Plot')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()