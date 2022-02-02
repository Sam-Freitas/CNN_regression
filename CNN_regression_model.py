from xml.etree.ElementInclude import include
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, BatchNormalization, Activation, Dense
from tensorflow.keras import backend as K
from tensorflow.python.keras.engine import training
from torch import channels_last, dropout
from utils.DropBlock import DropBlock2D
from skimage import measure
from tqdm import tqdm
from natsort import natsorted
import matplotlib.pyplot as plt
import albumentations as A
import tensorflow as tf
import numpy as np
import imutils
import shutil
import random
import glob
import sys
import cv2
import os


def fully_connected_CNN(use_dropout = False, height = 128, width = 128, channels = 1, kernal_size = (3,3)):

    inital_filter_size = 16
    dropsize = 0.9
    blocksize = 7

    inputs = Input((height, width, channels))

    s = inputs

    # first block of convolutions
    conv_1 = Conv2D(inital_filter_size, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same')(s)
    conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    conv_1 = Conv2D(inital_filter_size, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_1)
    conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    pool_1 = MaxPooling2D((2,2))(conv_1)

    # second block of convolutions
    conv_1 = Conv2D(inital_filter_size*2, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_1)
    conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    conv_1 = Conv2D(inital_filter_size*2, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_1)
    conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    pool_1 = MaxPooling2D((2,2))(conv_1)

    # third block of convolutions
    conv_1 = Conv2D(inital_filter_size*4, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_1)
    conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    conv_1 = Conv2D(inital_filter_size*4, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_1)
    conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    pool_1 = MaxPooling2D((2,2))(conv_1)

    # first block of convolutions
    conv_2 = Conv2D(inital_filter_size, (5,5), activation = None, kernel_initializer = 'he_normal', padding = 'same')(s)
    conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)

    conv_2 = Conv2D(inital_filter_size, (5,5), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_2)
    conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)

    pool_2 = MaxPooling2D((2,2))(conv_2)

    # second block of convolutions
    conv_2 = Conv2D(inital_filter_size*2, (5,5), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_2)
    conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)

    conv_2 = Conv2D(inital_filter_size*2, (5,5), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_2)
    conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)

    pool_2 = MaxPooling2D((2,2))(conv_2)

    # third block of convolutions
    conv_2 = Conv2D(inital_filter_size*4, (5,5), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_2)
    conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)

    conv_2 = Conv2D(inital_filter_size*4, (5,5), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_2)
    conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)

    pool_2 = MaxPooling2D((2,2))(conv_2)

    to_dense = tf.keras.layers.Concatenate()([pool_1,pool_2])

    flattened = tf.keras.layers.Flatten()(pool_1)
    
    d = Dense(1024,activation='relu')(flattened)

    d = Dropout(0.3)(d, training = use_dropout)

    output = Dense(1,activation='linear')(d)

    model = Model(inputs=[inputs], outputs=[output])

    return model

def ResNet50v2_regression(use_dropout = False, height = 128, width = 128, channels = 1):

    base_model = tf.keras.applications.resnet_v2.ResNet50V2(
        include_top = False, weights = None, input_shape = (height,width,channels)
    )
    last_base_layer = base_model.get_layer('post_bn').output
    x = tf.keras.layers.Flatten()(last_base_layer)
    x = Dense(1024,activation='relu')(x)
    x = Dropout(0.5)(x, training = use_dropout)
    x = Dense(1,activation='linear')(x)

    model = Model(inputs = base_model.input, outputs = x)


    return model

def plot_model(model):

    try:
        tf.keras.utils.plot_model(
            model, to_file='model.png', show_shapes=True, show_dtype=True,
            show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)
    except:
        print("Exporting model to png failed")
        print("Necessary packages: pydot (pip) and graphviz (brew)")

def load_rotated_minst_dataset():

    import time

    (X,labels),(_,_) = tf.keras.datasets.mnist.load_data()

    only_1 = [labels==1][0]

    labels = labels[only_1]
    X = X[only_1]

    X = X[3]

    X_out = []
    y_out = []

    for i in range(50):
        angle = 0
        y_out.append(angle)
        rot = imutils.rotate(X, angle=angle)
        resized = cv2.resize(rot, (32,32), interpolation = cv2.INTER_LINEAR)
        X_out.append(resized)

    for i in range(950):
        angle = np.random.randint(low = -90, high = 90)
        y_out.append(angle)
        rot = imutils.rotate(X, angle=angle)
        resized = cv2.resize(rot, (32,32), interpolation = cv2.INTER_LINEAR)
        X_out.append(resized)

    X_out = np.asarray(X_out)
    y_out = np.asarray(y_out)

    X_out_test = []
    y_out_test = []
    for i in range(200):
        angle = np.random.randint(low = -90, high = 90)
        y_out_test.append(angle)
        rot = imutils.rotate(X, angle=angle)
        resized = cv2.resize(rot, (32,32), interpolation = cv2.INTER_LINEAR)
        X_out_test.append(resized)

    X_out_test = np.asarray(X_out_test)
    y_out_test = np.asarray(y_out_test)

    return (X_out,y_out), (X_out_test,y_out_test)