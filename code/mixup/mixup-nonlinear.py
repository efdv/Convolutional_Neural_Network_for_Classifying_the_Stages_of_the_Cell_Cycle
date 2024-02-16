# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 12:53:03 2023

@author: ef.duquevazquez
"""
import tensorflow as tf
import glob
import cv2
import numpy as np

def mixup(images):
  # mix_coeff = tf.random.uniform([])
  mix_coeff = np.random.uniform()
  image = mix_coeff * images[0] + (1 - mix_coeff) * images[1]
  return image

# def vert_concat(images):
#   (_, H, W, C) = images.shape
#   dim_H = tf.random.uniform([], 0, H, dtype=tf.int32)
#   image = tf.concat([images[0][:dim_H, :, :],
#                      images[1][dim_H:, :, :]], 0)
#   return image

def vert_concat(images):
    (_, H, W, C) = images.shape
    dim_H = np.random.randint(0, H)
    image = np.concatenate([images[0][:dim_H, :, :],
                            images[1][dim_H:, :, :]], axis=0)
    return image

def horiz_concat(images):
  # (_, H, W, C) = images.get_shape().as_list()
  (_, H, W, C) = images.shape
  dim_W = np.random.randint(0, W)
  image = np.concatenate([images[0][:, :dim_W, :],
                     images[1][:, dim_W:, :]], axis=1)
  return image

def vh_mixup(images):
  image1 = vert_concat(images)
  # image1 = tf.cast(image1, tf.float32)
  image2 = horiz_concat(images)
  # image2 = tf.cast(image2, tf.float32)
  return mixup([image1, image2])




path = "../../../../datasets/cellcycle_dataset_ch4/cellcycle_ch4-wgandiv/Anaphase/"
dirfile =  glob.glob(path+'/*')
dataset = []
for ipath in dirfile[:2]:
    I = cv2.imread(ipath)
#    I = I/255
    dataset.append(I)
    
images = np.array(dataset)
images = np.reshape(images, (-1, 64, 64,3))


m = vh_mixup(dataset)
m = np.array(m)
# namedir = 'mixup_'+stage
dire = "E:/OneDrive - Universidad de Guanajuato/EF-Duque-Vazquez-Doctorado/projects/cell_cycle/code/mixup"
for i, image in enumerate(m):
    filename = dire+ "/%d.png" % (i)
    img = cv2.normalize(np.array(image), None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(filename, img)


# m = cv2.normalize(np.array(m), None, 0, 255, cv2.NORM_MINMAX)
m = (m - np.min(m)) * (255 / (np.max(m) - np.min(m)))

cv2.imshow('Imagen', nn)
cv2.waitKey(0)