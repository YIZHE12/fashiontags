#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .constants import paths, num_classes
from .util import read_img
from keras.utils import Sequence
from tensorflow.python.lib.io import file_io
from os.path import join
import keras.backend as K
import numpy as np
import pandas as pd
import os

from keras.applications.vgg19 import preprocess_input ### add for the compress models
# prepare the image for the VGG model

class SequenceFromDisk(Sequence):

    def __init__(self, mode, batch_size, img_size, preprocessfunc=None):
        self.mode = mode
        self.batch_size = batch_size
        self.img_size = img_size
        self.preprocessfunc = preprocessfunc
        self.paths = paths[mode]
        self.csv = pd.read_csv(self.paths['csv'])
        self.n_samples = len(self.csv)
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))

        if self.mode != 'test':
            self.labels = np.load(self.paths['labels'])

        print('SequenceFromDisk <{}>: {} samples'.format(mode, self.n_samples))

    def get_all_labels(self):
        if self.mode != 'test':
            return self.labels
        else:
            raise AttributeError(
                'This is a {} data generator, no labels available!'.format(self.mode))

    def __getitem__(self, idx):
        images = []

        start = idx * self.batch_size
        end = np.min([start + self.batch_size, self.n_samples])
        idxs = np.arange(start, end)

        for i in idxs:
            try:
                row = self.csv.iloc[i+256000*2, :]
                img_id = row['imageId']
                img_path = join(self.paths['dir'], '{}.jpg'.format(img_id))
                #img = read_img(img_path, self.img_size)
                img = read_img(img_path)
                img = preprocess_input(img)
                images.append(img)
            except Exception as e:
                print('Failed to read index {}'.format(i))
                img=np.zeros((224, 224, 3))
                images.append(img)

        images = np.stack(images).astype(K.floatx())

        if self.preprocessfunc:
            images = self.preprocessfunc(images)

        if self.mode == 'test':
            return images
        else:
            return images, self.labels[idxs]

    def __len__(self):
        return self.n_batches
