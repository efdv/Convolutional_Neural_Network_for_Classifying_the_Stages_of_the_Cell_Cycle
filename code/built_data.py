# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:18:56 2023

@author: ef.duquevazquez
"""

import glob
import cv2
import numpy as np 


# def load_images(dirfile, phase, dlabels):
#     # stage = []
#     # label = []
#     cont = 0
#     for ipath in dirfile:
#         print(cont)
#         I = cv2.imread(ipath)
#         resized_image = cv2.resize(I, (64,64))
#         # stage.append(resized_image)
#         # label.append(dlabels[phase])
#         cont += 1

#     return stage, label


path = "../../../datasets/cellcycle_dataset_ch4/datasetRana1/"
nameFolders = ['G1', 'S', 'G2', 'Prophase', 'Metaphase', 'Anaphase', 'Telophase']

cellCycle = []
labels = []
dlabels = {'G1':0, 'S':1, 'G2':2, 'Prophase':3, 'Metaphase':4, 'Anaphase':5, 'Telophase':6 }

for phase in nameFolders:
    
    dirfile =  glob.glob(path+phase+'/*')
    # stage,label = load_images(dirfile, phase, dlabels)
    cont = 0
    for ipath in dirfile:
        print("%s ---- %i----%i"%(phase,cont, len(dirfile)))
        I = cv2.imread(ipath)
        resized_image = cv2.resize(I, (64,64))
        cont += 1
        cellCycle.append(resized_image)
        labels.append(dlabels[phase])

cellCycle = np.array(cellCycle)
cellCycle = cellCycle
labels = np.array(labels)

np.save('../../../datasets/cellcycle_dataset_ch4/datasetRana1/cellCycle.npy', cellCycle)
np.save('../../../datasets/cellcycle_dataset_ch4/datasetRana1/labels.npy', labels)


