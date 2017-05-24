# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 13:59:35 2017

@author: wanggd
"""

import os
import glob
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
from config import *
import imutils
import cv2
import matplotlib.pyplot as plt 
from skimage import color
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import numpy as np

def extract_features():
    
    if not os.path.isdir(pos_fea_dir):
        os.makedirs(pos_fea_dir)
    if not os.path.isdir(neg_fea_dir):
        os.makedirs(neg_fea_dir)
        
    for img_path in glob.glob(os.path.join(pos_dir, '*.png')):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        fd = hog(img, orientations, pixels_per_cell, 
                 cells_per_block, visualize, normalize)
        fd_name = os.path.split(img_path)[1].split(".")[0] + '.feat'
        fd_path = os.path.join(pos_fea_dir, fd_name)
        joblib.dump(fd, fd_path)
    
    for img_path in glob.glob(os.path.join(neg_dir, '*')):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        fd = hog(img, orientations, pixels_per_cell, 
                 cells_per_block, visualize, normalize)
        fd_name = os.path.split(img_path)[1].split(".")[0] + '.feat'
        fd_path = os.path.join(neg_fea_dir, fd_name)
        joblib.dump(fd, fd_path)
        
    print('feature extraction completely')
    print('Classifier saved to ../data/features')

def train():
    
    if os.path.exists(model_path):
        print('A model trained before')
        return 
        
    if not os.path.exists(pos_fea_dir):
        extract_features()
        
    data = []
    label = []
    
    print('Load features...')
    # Load the positive features    
    for fea_path in glob.glob(os.path.join(pos_fea_dir, '*.feat')):
        fd = joblib.load(fea_path)
        data.append(fd)
        label.append(1)

    # Load the negative features
    for fea_path in glob.glob(os.path.join(neg_fea_dir,"*.feat")):
        fd = joblib.load(fea_path)
        data.append(fd)
        label.append(0)

    clf = LinearSVC()
    print('Training a Linear SVM Classifier')
    
    clf.fit(data, label)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    joblib.dump(clf, os.path.join(model_path,'svm.model'))
    print('Classifier saved to '+model_path)
    
    
def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input 'image' of size 
    equal to 'window_size'. The first image returned top-left 
    co-ordinate (0, 0) and are increment in both x and y directions
    by the 'step_size' supplied.

    So, the input parameters are-
    image - Input image
    window_size - Size of Sliding Window 
    step_size - incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    '''
    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])   

def detector(filename):
    
    im = cv2.imread(filename)
    im = imutils.resize(im, width = min(400, im.shape[1]))
    downscale = 1.25

    clf = joblib.load(os.path.join(model_path, 'svm.model'))

    #List to store the detections area
    detections = []
    #The current scale of the image 
    scale = 0

    for im_scaled in pyramid_gaussian(im, downscale = downscale):
        #The list contains detections at the current scale
        #cv2.imshow('dd', im_scaled)
        #cv2.waitKey()
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            im_window = color.rgb2gray(im_window)
            
            fd = hog(im_window, orientations, 
                     pixels_per_cell, cells_per_block, 
                     visualise=visualize, transform_sqrt=True)

            fd = fd.reshape(1, -1)
            pred = clf.predict(fd)
            '''
            cv2.imshow('d', im_window)
            cv2.waitKey()
            print pred,
            print clf.decision_function(fd)
            '''
            if pred == 1:
                if clf.decision_function(fd) > 0.6:
                    detections.append((int(x * (downscale**scale)), 
                                       int(y * (downscale**scale)), 
                                       clf.decision_function(fd), 
                                       int(min_wdw_sz[0] * (downscale**scale)),
                                       int(min_wdw_sz[1] * (downscale**scale))))            
        scale += 1

    clone = im.copy()

    for (x_tl, y_tl, _, w, h) in detections:
        cv2.rectangle(im, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness = 2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    print "sc: ", sc
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)
    if isinstance(pick, list):
        return
    print "shape, ", pick.shape

    for(xA, yA, xB, yB) in pick:
        cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
    cv2.imshow('d', im)
    cv2.waitKey()
    '''
    plt.axis("off")
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.title("Raw Detection before NMS")
    plt.show()
  
    plt.axis("off")
    plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
    plt.title("Final Detections after applying NMS")
    plt.show()
    '''
    cv2.imshow('d', clone)
    cv2.waitKey()

def test_folder(foldername):
    filenames = glob.iglob(os.path.join(foldername, '*'))
    for filename in filenames:
        detector(filename)

if __name__ == '__main__':
    foldername = 'test_image'
    test_folder(foldername)       
        
        
        
        
        
        
        
        
        
        
    
    
    