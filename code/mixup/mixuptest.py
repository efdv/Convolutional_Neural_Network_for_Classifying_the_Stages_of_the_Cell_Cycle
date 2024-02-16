# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:22:15 2023

@author: ef.duquevazquez
"""
import cv2
import glob
import numpy as np
import tensorflow as tf
import os

liststages = ['Prophase', 'Metaphase', 'Anaphase', 'Telophase']
numsamples = [3500, 3500, 3500, 1750]

def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

for stage, nosamples in zip(liststages, numsamples):
    print(stage)
    namedir = 'mixup_'+stage
    os.makedirs(namedir, exist_ok=True)

    path = "../../../../datasets/cellcycle_dataset_ch4/cellcycle_ch4-wgandiv/"+stage+"/"
    dirfile =  glob.glob(path+'/*')
    dataset = []
    for ipath in dirfile[:nosamples]:
        I = cv2.imread(ipath)
        I = (I /127.5) - 1
        dataset.append(I)

    batch_size = 64
    

    dataset = np.array(dataset)
    dataset = dataset.astype("float32") / 255.0
    dataset = np.reshape(dataset, (-1, 64, 64, 3))


    # mixup

    batch_size = tf.shape(dataset)[0]
    l = sample_beta_distribution(batch_size, 0.2, 0.8)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    
    images = dataset * x_l * (1 - x_l)




    for i, image in enumerate(images):
        filename = namedir + "/%d.png" % (i)
        img = cv2.normalize(np.array(image), None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(filename, img)
        # print(label.numpy().tolist())