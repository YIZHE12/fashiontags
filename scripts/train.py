import numpy as np
# fix random seed for reproducibility
np.random.seed(7)
import pandas as pd
import math
import os
import time
import cv2
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, scale
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn.preprocessing import normalize, scale


import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.metrics import binary_accuracy 
from keras import regularizers
from keras import optimizers
import keras_metrics as km

from tensorflow.python.ops import array_ops
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf

import seaborn as sns
import matplotlib.pyplot as plt

from fashion_code.my_utils import *
from keras.optimizers import Adam

os.chdir('/home/ubuntu/hunter_fashion_multi_gpu')
exp_name_base = 'train_final'



x_folder = './data/train_x_npy/'
y_folder = './data/train_sub_y_npy/'

n_class = 70
n_examples = 256000*3
n_epoch = 100
batch_size = 512
imgEmbd_dir = x_folder
label_dir = y_folder


##############################################################################
X = [i for i in range(n_examples)]
y = X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.11, random_state=16)


train_generator = DataGenerator(X_train, x_folder, y_folder, n_class, batch_size=batch_size)
validation_generator = DataGenerator(X_val, x_folder, y_folder, n_class, batch_size = batch_size)
test_data_x, test_data_y = get_test_data(X_test, x_folder, y_folder, n_class =  n_class)

print('='*20)
print('training examples:', np.shape(X_train))
print('validation examples:', np.shape(X_val))
print('test examples:', np.shape(test_data_x))
print('='*20)

##############################################################################

# create model
model = Sequential()
model.add(Dense(1048, activation='relu', input_dim= 4096, kernel_initializer='he_normal'))
model.add(BatchNormalization())
# Arch 3, add drop out
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
# model.add(Dense(n_class, activation='softmax', kernel_initializer='he_normal'))
model.add(Dense(n_class, activation='sigmoid', kernel_initializer='he_normal'))


precision = km.binary_precision(label=1)
recall = km.binary_recall(label=0)
adam = Adam(lr=0.0001)  #reduce learning rate later in the training process

##############################################################################
exp_name = exp_name_base + '_binary_loss' 
NAME = exp_name + str(int(time.time()))
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
filepath= exp_name +"-epoch-{epoch:02d}.hdf5"

mc = ModelCheckpoint('./models/'+filepath, monitor='val_loss', verbose=1, \
                     save_best_only=True, save_weights_only=True)

ec = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

model.compile(optimizer=adam,
              loss="binary_crossentropy",
              metrics=[km.binary_precision(), km.binary_recall()])

print(model.summary())
# model.load_weights('./models/train_final_focal_loss_retrain_load_15_-epoch-28.hdf5')
model.fit_generator(generator=train_generator,
                    # steps_per_epoch = 50,
                    validation_data=validation_generator,
                    epochs = n_epoch,
#                     validation_steps = 50,
                    shuffle = True,
                    callbacks = [tensorboard, mc, ec])
##############################################################################
exp_name = exp_name_base + '_f1_loss' 
NAME = exp_name + str(int(time.time()))
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
filepath= exp_name +"-epoch-{epoch:02d}.hdf5"

mc = ModelCheckpoint('./models/'+filepath, monitor='val_loss', verbose=1, \
                     save_best_only=True, save_weights_only=True)

ec = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

model.compile(optimizer=adam,
              loss=f1_loss,
              metrics=[km.binary_precision(), km.binary_recall()])

print(model.summary())
# model.load_weights('./models/train_final_focal_loss_retrain_load_15_-epoch-28.hdf5')
model.fit_generator(generator=train_generator,
                    # steps_per_epoch = 50,
                    validation_data=validation_generator,
                    epochs = n_epoch,
#                     validation_steps = 50,
                    shuffle = True,
                    callbacks = [tensorboard, mc, ec])

##############################################################################
exp_name = exp_name_base + '_focal_loss' 
NAME = exp_name + str(int(time.time()))
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
filepath= exp_name +"-epoch-{epoch:02d}.hdf5"

mc = ModelCheckpoint('./models/'+filepath, monitor='val_loss', verbose=1, \
                     save_best_only=True, save_weights_only=True)

ec = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

model.compile(optimizer=adam,
#               loss="binary_crossentropy",
              loss = focal_loss,
              metrics=[km.binary_precision(), km.binary_recall()])

print(model.summary())
# model.load_weights('./models/train_final_focal_loss_retrain_load_15_-epoch-28.hdf5')
model.fit_generator(generator=train_generator,
                    # steps_per_epoch = 50,
                    validation_data=validation_generator,
                    epochs = n_epoch,
#                     validation_steps = 50,
                    shuffle = True,
                    callbacks = [tensorboard, mc, ec])













