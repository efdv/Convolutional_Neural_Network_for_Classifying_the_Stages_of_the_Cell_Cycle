# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:42:38 2023

@author: ef.duquevazquez
"""
import glob
import cv2

def load_images(dirfile, phase, dlabels):
    stage = []
    label = []
    cont = 0
    for ipath in dirfile:
        print(cont)
        I = cv2.imread(ipath)
        stage.append(I)
        label.append(dlabels[phase])
        cont += 1
    
    return stage, label

path = "D:/OneDrive - Universidad de Guanajuato/EF-Duque-Vazquez-Doctorado/datasets/CellCycle/"

nameFolders = ['G1', 'S', 'G2', 'M']
dirG1 =  glob.glob(path+nameFolders[0]+'/*')

cellCycle = []
dlabels = {'G1':0,'S':1,'G2':2, 'M':3 }
labels = []    
for phase in nameFolders:
    print(phase)
    dirfile =  glob.glob(path+phase+'/*')
    stage,label = load_images(dirfile, phase, dlabels)
    cellCycle.append(stage)
    labels.append(label)



# Tamaño y Normalización
cellCycle_norm = []
for stage in cellCycle:
    normList = []
    for img in stage:
        img = cv2.resize(img, [64,64])
        img = img/255
        
        normLis.append(img)
    

# Validación cruzada y particionamiento