#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
import numpy as np
np.random.seed(7)
import pandas as pd
import math
import os
import time
import cv2

from PIL import Image
from fashion_code.constants import paths
from keras.preprocessing.image import img_to_array, load_img
from os.path import join
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.python.lib.io import file_io


import keras.backend as K
import tensorflow as tf
from sklearn.preprocessing import normalize, scale
from keras.utils import Sequence

def read_img(fname, size, gcp=False):
    if gcp:
        with file_io.FileIO(fname, 'rb') as f:
            image_bytes = f.read()
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img = img.resize(size, Image.BILINEAR)
            return img_to_array(img)
    else:
        img = load_img(fname, target_size=size)
        
        return img_to_array(img)

class DataGenerator(Sequence):

    def __init__(self, list_IDs, imgEmbd_dir, label_dir, classs_number, batch_size=256, shuffle = True):
        'Initialization'
        # For image-only model, row of X is image ID, row of Y is 40-class labels
        self.list_IDs = list_IDs # index number generate by random split
        self.imgEmbd_dir = imgEmbd_dir # npy folder of embedding (X)
        self.label_dir = label_dir # npy folder of the label (Y)
        self.batch_size = batch_size
        self.classs_number = classs_number # number of classes
        self.shuffle = shuffle
        self.on_epoch_end()
    

    def __len__(self):
        'The number of batches per epoch'
        return int(math.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        batch_X, batch_y = self.__data_generation(list_IDs_temp)
        return batch_X, batch_y
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'

        X = np.empty((self.batch_size, 4096))
        y = np.empty((self.batch_size, self.classs_number))

        for i, ID in enumerate(list_IDs_temp):    
            X[i,:] = np.expand_dims(np.load(self.imgEmbd_dir + '/' + str(ID) +'.npy'), axis = 0)
            y[i,:] = np.expand_dims(np.load(self.label_dir + '/' + str(ID) +'.npy'), axis = 0)       
        X = normalize(X)
        X = scale(X)  
        return X, y
    
def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def focal_loss(y_true, y_pred):
    gamma = 2.0
    epsilon = K.epsilon()
    pt = y_pred * y_true + (1-y_pred) * (1-y_true)
    pt = K.clip(pt, epsilon, 1-epsilon)
    CE = -K.log(pt)
    FL = K.pow(1-pt, gamma) * CE
    loss = K.sum(FL, axis=1)
    return loss

def focal_loss2(y_true, y_pred):#with tensorflow
    eps = 1e-12
    gamma=2
    #alpha=0.75
    alpha = 0.25
    y_pred=K.clip(y_pred,eps,1.-eps)#improve the stability of the focal loss 
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    # when y_true == 1, outut y_pred, otherwise output 1
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    # when y_true == 0, outut y_pred, otherwise output 0
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))\
            -K.sum((1-alpha)* K.pow( pt_0, gamma) * K.log(1. - pt_0))

def focal_loss_f1(y_true, y_pred):
    gamma = 2.0
    epsilon = K.epsilon()
    pt = y_pred * y_true + (1-y_pred) * (1-y_true)
    pt = K.clip(pt, epsilon, 1-epsilon)
    CE = -K.log(pt)
    FL = K.pow(1-pt, gamma) * CE
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    
    loss = K.sum(FL*(1 - K.mean(f1)), axis=1)
    return loss


def get_test_data(X_test, x_npy_folder, y_npy_folder, n_class = 11):
    n_exp = len(X_test) # x_test is a list of index for testing data
    test_data_x = np.empty((n_exp, 4096))
    test_data_y = np.empty((n_exp, n_class))
    
    for i, index in enumerate(X_test):

        test_data_x[i,:] = np.expand_dims(np.load(x_npy_folder + '/' + str(index) +'.npy'),\
                                          axis = 0)
        test_data_y[i,:] = np.expand_dims(np.load(y_npy_folder + '/' + str(index) +'.npy'),\
                                          axis = 0)  
        
    test_data_x = normalize(test_data_x)
    test_data_x = scale(test_data_x) 
    
    return(test_data_x, test_data_y)

def multilabel_confusion_matrix(y_true, y_pred, sample_weight=None,
                                labels=None, samplewise=False):
    # Compute a confusion matrix for each class or sample

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
    check_consistent_length(y_true, y_pred, sample_weight)

    if y_type not in ("binary", "multiclass", "multilabel-indicator"):
        raise ValueError("%s is not supported" % y_type)

    present_labels = unique_labels(y_true, y_pred)
    if labels is None:
        labels = present_labels
        n_labels = None
    else:
        n_labels = len(labels)
        labels = np.hstack([labels, np.setdiff1d(present_labels, labels,
                                                 assume_unique=True)])

    if y_true.ndim == 1:
        if samplewise:
            raise ValueError("Samplewise metrics are not available outside of "
                             "multilabel classification.")

        le = LabelEncoder()
        le.fit(labels)
        y_true = le.transform(y_true)
        y_pred = le.transform(y_pred)
        sorted_labels = le.classes_

        # labels are now from 0 to len(labels) - 1 -> use bincount
        tp = y_true == y_pred
        tp_bins = y_true[tp]
        if sample_weight is not None:
            tp_bins_weights = np.asarray(sample_weight)[tp]
        else:
            tp_bins_weights = None

        if len(tp_bins):
            tp_sum = np.bincount(tp_bins, weights=tp_bins_weights,
                                 minlength=len(labels))
        else:
            # Pathological case
            true_sum = pred_sum = tp_sum = np.zeros(len(labels))
        if len(y_pred):
            pred_sum = np.bincount(y_pred, weights=sample_weight,
                                   minlength=len(labels))
        if len(y_true):
            true_sum = np.bincount(y_true, weights=sample_weight,
                                   minlength=len(labels))

        # Retain only selected labels
        indices = np.searchsorted(sorted_labels, labels[:n_labels])
        tp_sum = tp_sum[indices]
        true_sum = true_sum[indices]
        pred_sum = pred_sum[indices]

    else:
        sum_axis = 1 if samplewise else 0

        # All labels are index integers for multilabel.
        # Select labels:
        if not np.array_equal(labels, present_labels):
            if np.max(labels) > np.max(present_labels):
                raise ValueError('All labels must be in [0, n labels) for '
                                 'multilabel targets. '
                                 'Got %d > %d' %
                                 (np.max(labels), np.max(present_labels)))
            if np.min(labels) < 0:
                raise ValueError('All labels must be in [0, n labels) for '
                                 'multilabel targets. '
                                 'Got %d < 0' % np.min(labels))

        if n_labels is not None:
            y_true = y_true[:, labels[:n_labels]]
            y_pred = y_pred[:, labels[:n_labels]]

        # calculate weighted counts
        true_and_pred = y_true.multiply(y_pred)
        tp_sum = count_nonzero(true_and_pred, axis=sum_axis,
                               sample_weight=sample_weight)
        pred_sum = count_nonzero(y_pred, axis=sum_axis,
                                 sample_weight=sample_weight)
        true_sum = count_nonzero(y_true, axis=sum_axis,
                                 sample_weight=sample_weight)

    fp = pred_sum - tp_sum
    fn = true_sum - tp_sum
    tp = tp_sum

    if sample_weight is not None and samplewise:
        sample_weight = np.array(sample_weight)
        tp = np.array(tp)
        fp = np.array(fp)
        fn = np.array(fn)
        tn = sample_weight * y_true.shape[1] - tp - fp - fn
    elif sample_weight is not None:
        tn = sum(sample_weight) - tp - fp - fn
    elif samplewise:
        tn = y_true.shape[1] - tp - fp - fn
    else:
        tn = y_true.shape[0] - tp - fp - fn

    return np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)
