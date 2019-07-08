
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
from fashion_code.callbacks import F1Utility, Finetuning, MultiGPUCheckpoint
from fashion_code.constants import num_classes, paths, GCP_paths
from fashion_code.generator_test import SequenceFromDisk

from keras.applications.xception import Xception, preprocess_input
from keras.utils.training_utils import multi_gpu_model
#from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import Model, load_model
from keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from os.path import join
import time
import sys
from keras.models import Sequential 

from keras.applications.xception import preprocess_input
from keras.models import load_model
from sklearn.metrics import f1_score

from keras.applications import vgg19

import cv2

def write_npy(predict, prefix, folder):
    for i in range(len(predict)):
        out_path = folder+str(i+prefix)+'.npy'
        np.save(out_path, predict[i,:])


data = pd.read_csv('./data/labels_train.csv')
y_label = np.load('./data/labels_train.npy')

index = 0
prefix = index  # first batch
npy_folder = './data/embedding_x/'
os.mkdir(y_folder)
y_folder = './data/embedding_y/'
os.mkdir(y_folder)
# folder to store the embedding data

# load pre-trained VGG19 model
base_model = vgg19.VGG19(weights='imagenet') #imports the mobilenet model
for layer in base_model.layers:
    layer.trainable=False

output = base_model.get_layer('fc2').output
model=Model(inputs=base_model.input, outputs=output)
model.compile('adam', 'binary_crossentropy')


val_gen = SequenceFromDisk('train', 256, (224, 224))
labels = val_gen.get_all_labels()
preds = model.predict_generator(val_gen, verbose=1, steps = 1)
n_examples = np.shape(preds)[0]
print('total number of new training data:', n_examples)
y_label = y_label[index:index+n_examples,:]


# get the label subset starting from the choosen index

# select the labels that the company interested
select_index = [ 2,   7,   8,  11,  13,  16,  18,  19,  24,  26,  38,  40,  41,
        42,  43,  44,  49,  50,  53,  54,  56,  58,  60,  61,  66,  72,
        78,  80,  84,  85,  89,  90,  93,  94,  96,  99, 100, 101, 104,
       112, 116, 118, 124, 128, 135, 137, 147, 149, 155, 156, 158, 167,
       172, 175, 177, 183, 186, 187, 190, 191, 196, 197, 199, 200, 204,
       205, 215, 219, 225, 226]


y_selected = y_label_test[:,select_index]

np.save('./data/' + 'train_img_x.npy', preds)
np.save('./data/' + 'train_img_y.npy', y_label)
np.save('./data/' + 'train_img_y_select.npy', y_selected)

script_start_time = time.time()
print('%0.2f min: Start processing data'%((time.time() - script_start_time)/60))
write_npy(y_label, prefix, y_folder)
print('%0.2f min: Finish processing data'%((time.time() - script_start_time)/60))

# save embedding data n npy file one by one
script_start_time = time.time()
print('%0.2f min: Start processing data'%((time.time() - script_start_time)/60))
write_npy(y_selected, prefix, npy_folder)
print('%0.2f min: Finish processing data'%((time.time() - script_start_time)/60))








