# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 20:37:03 2017

@author: wanggd
"""

import numpy as np
import matplotlib.pylab as plt


import cPickle
from os import listdir
from bottleneck import argpartition

def split_num(checkcode):
    res = []
    threshold = (np.max(checkcode) - np.min(checkcode)) / 2 
    for i in range(4):
        res.append(checkcode[3:15, 2 + i * 10:i * 10 + 10,1] > threshold)
    return np.asarray(res, dtype='uint8')

def load_datasets(dir_name = './digits'):
    data = []
    label = []
    for file_name in listdir(dir_name):
        label.append(int(file_name[0]))
        path = "%s/%s"%(dir_name, file_name)
        with open(path, 'rb') as f:
            num = cPickle.load(f)
            data.append(num)
    data = np.asarray(data, dtype='uint8')
    label = np.asarray(label, dtype='uint8')
    return data,label

def KNN(unknown, data, label, k=3):
    shape = data.shape
    data = np.reshape(data, (shape[0], shape[1] * shape[2]))
    unknown = np.reshape(unknown, shape[1] * shape[2])
    similarity_list = []
    for i in range(len(data)):
        similarity = np.sum(unknown == data[i])
        similarity_list.append(similarity)
    similarity_list = np.asarray(similarity_list)
    print(similarity_list)
    index = argpartition(-similarity_list, k)[:k]
    return np.median(label[index])
    
def img2num(img,data=None,label=None):
    if data is None and label is None:
        data, label = load_datasets()
    nums = split_num(img)
    res = []
    for i in range(len(nums)):
        num = KNN(nums[i], data, label)
        res.append(num)
    return np.asarray(res, dtype='uint8')
    
img = plt.imread('./images/1.png')
plt.imshow(img)
res = img2num(img)
print(res)












