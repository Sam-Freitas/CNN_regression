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


def fully_connected_CNN_v2(use_dropout = False, height = 128, width = 128, channels = 1, kernal_size = (3,3), inital_filter_size = 16,dropsize = 0.9,blocksize = 7):

    inputs = Input((height, width, channels))

    s = inputs

    # first block of convolutions
    conv_3 = Conv2D(inital_filter_size, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same')(s)
    conv_3 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_3, training = use_dropout)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)

    conv_3 = Conv2D(inital_filter_size, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_3)
    conv_3 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_3, training = use_dropout)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)

    pool_3 = MaxPooling2D((2,2))(conv_3)

    # second block of convolutions
    conv_3 = Conv2D(inital_filter_size*2, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_3)
    conv_3 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_3, training = use_dropout)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)

    conv_3 = Conv2D(inital_filter_size*2, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_3)
    conv_3 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_3, training = use_dropout)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)

    pool_3 = MaxPooling2D((2,2))(conv_3)

    # third block of convolutions
    conv_3 = Conv2D(inital_filter_size*4, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_3)
    conv_3 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_3, training = use_dropout)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)

    conv_3 = Conv2D(inital_filter_size*4, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_3)
    conv_3 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_3, training = use_dropout)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)

    pool_3 = MaxPooling2D((2,2))(conv_3)

    # first block of convolutions
    conv_2 = Conv2D(inital_filter_size, (7,7), activation = None, kernel_initializer = 'he_normal', padding = 'same')(s)
    conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)

    conv_2 = Conv2D(inital_filter_size, (7,7), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_2)
    conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)

    pool_2 = MaxPooling2D((2,2))(conv_2)

    # second block of convolutions
    conv_2 = Conv2D(inital_filter_size*2, (7,7), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_2)
    conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)

    conv_2 = Conv2D(inital_filter_size*2, (7,7), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_2)
    conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)

    pool_2 = MaxPooling2D((2,2))(conv_2)

    # third block of convolutions
    conv_2 = Conv2D(inital_filter_size*4, (7,7), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_2)
    conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)

    conv_2 = Conv2D(inital_filter_size*4, (7,7), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_2)
    conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)

    pool_2 = MaxPooling2D((2,2))(conv_2)

    # first block of convolutions
    conv_1 = Conv2D(inital_filter_size, (5,5), activation = None, kernel_initializer = 'he_normal', padding = 'same')(s)
    conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    conv_1 = Conv2D(inital_filter_size, (5,5), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_1)
    conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    pool_1 = MaxPooling2D((2,2))(conv_1)

    # second block of convolutions
    conv_1 = Conv2D(inital_filter_size*2, (5,5), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_1)
    conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    conv_1 = Conv2D(inital_filter_size*2, (5,5), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_1)
    conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    pool_1 = MaxPooling2D((2,2))(conv_1)

    # third block of convolutions
    conv_1 = Conv2D(inital_filter_size*4, (5,5), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_1)
    conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    conv_1 = Conv2D(inital_filter_size*4, (5,5), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_1)
    conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    pool_1 = MaxPooling2D((2,2))(conv_1)

    cat_layer = tf.keras.layers.Concatenate()([pool_1,pool_2,pool_3])

    # conv_final = Conv2D(256, (1,1), activation = None, kernel_initializer = 'he_normal', padding = 'same')(cat_layer)
    # conv_final = Activation('relu')(conv_final)

    # flattened = tf.keras.layers.Flatten()(conv_final)

    flattened = tf.keras.layers.Flatten()(cat_layer)
    d = Dense(900, activation='gelu')(flattened)
    d = Activation('gelu')(d)
    d = Dropout(0.8)(d, training = use_dropout)
    d = Dense(2048,activation='gelu')(d)
    d = Activation('gelu')(d)
    d = Dropout(0.8)(d, training = use_dropout)

    output = Dense(1,activation='linear')(d)

    model = Model(inputs=[inputs], outputs=[output])

    return model

def fully_connected_CNN(use_dropout = False, height = 128, width = 128, channels = 1, kernal_size = (3,3), inital_filter_size = 16,dropsize = 0.9,blocksize = 7):

    inputs = Input((height, width, channels))

    s = inputs

    # # first block of convolutions
    # conv_1 = Conv2D(inital_filter_size, (1,1), activation = None, kernel_initializer = 'he_normal', padding = 'same')(s)
    # conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    # conv_1 = BatchNormalization()(conv_1)
    # conv_1 = Activation('relu')(conv_1)

    # conv_1 = Conv2D(inital_filter_size, (1,1), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_1)
    # conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    # conv_1 = BatchNormalization()(conv_1)
    # conv_1 = Activation('relu')(conv_1)

    # pool_1 = MaxPooling2D((2,2))(conv_1)

    # # second block of convolutions
    # conv_1 = Conv2D(inital_filter_size*2, (1,1), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_1)
    # conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    # conv_1 = BatchNormalization()(conv_1)
    # conv_1 = Activation('relu')(conv_1)

    # conv_1 = Conv2D(inital_filter_size*2, (1,1), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_1)
    # conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    # conv_1 = BatchNormalization()(conv_1)
    # conv_1 = Activation('relu')(conv_1)

    # pool_1 = MaxPooling2D((2,2))(conv_1)

    # # third block of convolutions
    # conv_1 = Conv2D(inital_filter_size*4, (1,1), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_1)
    # conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    # conv_1 = BatchNormalization()(conv_1)
    # conv_1 = Activation('relu')(conv_1)

    # conv_1 = Conv2D(inital_filter_size*4, (1,1), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_1)
    # conv_1 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_1, training = use_dropout)
    # conv_1 = BatchNormalization()(conv_1)
    # conv_1 = Activation('relu')(conv_1)

    # pool_1 = MaxPooling2D((2,2))(conv_1)

    # # first block of convolutions
    # conv_2 = Conv2D(inital_filter_size, (7,7), activation = None, kernel_initializer = 'he_normal', padding = 'same')(s)
    # conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    # conv_2 = BatchNormalization()(conv_2)
    # conv_2 = Activation('relu')(conv_2)

    # conv_2 = Conv2D(inital_filter_size, (7,7), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_2)
    # conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    # conv_2 = BatchNormalization()(conv_2)
    # conv_2 = Activation('relu')(conv_2)

    # pool_2 = MaxPooling2D((2,2))(conv_2)

    # # second block of convolutions
    # conv_2 = Conv2D(inital_filter_size*2, (7,7), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_2)
    # conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    # conv_2 = BatchNormalization()(conv_2)
    # conv_2 = Activation('relu')(conv_2)

    # conv_2 = Conv2D(inital_filter_size*2, (7,7), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_2)
    # conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    # conv_2 = BatchNormalization()(conv_2)
    # conv_2 = Activation('relu')(conv_2)

    # pool_2 = MaxPooling2D((2,2))(conv_2)

    # # third block of convolutions
    # conv_2 = Conv2D(inital_filter_size*4, (7,7), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_2)
    # conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    # conv_2 = BatchNormalization()(conv_2)
    # conv_2 = Activation('relu')(conv_2)

    # conv_2 = Conv2D(inital_filter_size*4, (7,7), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_2)
    # conv_2 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_2, training = use_dropout)
    # conv_2 = BatchNormalization()(conv_2)
    # conv_2 = Activation('relu')(conv_2)

    # pool_2 = MaxPooling2D((2,2))(conv_2)

    # first block of convolutions
    conv_3 = Conv2D(inital_filter_size, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same')(s)
    conv_3 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_3, training = use_dropout)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)

    conv_3 = Conv2D(inital_filter_size, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_3)
    conv_3 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_3, training = use_dropout)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)

    pool_3 = MaxPooling2D((2,2))(conv_3)

    # second block of convolutions
    conv_3 = Conv2D(inital_filter_size*2, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_3)
    conv_3 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_3, training = use_dropout)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)

    conv_3 = Conv2D(inital_filter_size*2, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_3)
    conv_3 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_3, training = use_dropout)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)

    pool_3 = MaxPooling2D((2,2))(conv_3)

    # third block of convolutions
    conv_3 = Conv2D(inital_filter_size*4, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_3)
    conv_3 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_3, training = use_dropout)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)

    conv_3 = Conv2D(inital_filter_size*4, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_3)
    conv_3 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_3, training = use_dropout)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)

    pool_3 = MaxPooling2D((2,2))(conv_3)

    # # first block of convolutions
    # conv_4 = Conv2D(inital_filter_size, (11,11), activation = None, kernel_initializer = 'he_normal', padding = 'same')(s)
    # conv_4 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_4, training = use_dropout)
    # conv_4 = BatchNormalization()(conv_4)
    # conv_4 = Activation('relu')(conv_4)

    # conv_4 = Conv2D(inital_filter_size, (11,11), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_4)
    # conv_4 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_4, training = use_dropout)
    # conv_4 = BatchNormalization()(conv_4)
    # conv_4 = Activation('relu')(conv_4)

    # pool_4 = MaxPooling2D((2,2))(conv_4)

    # # second block of convolutions
    # conv_4 = Conv2D(inital_filter_size*2, (11,11), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_4)
    # conv_4 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_4, training = use_dropout)
    # conv_4 = BatchNormalization()(conv_4)
    # conv_4 = Activation('relu')(conv_4)

    # conv_4 = Conv2D(inital_filter_size*2, (11,11), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_4)
    # conv_4 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_4, training = use_dropout)
    # conv_4 = BatchNormalization()(conv_4)
    # conv_4 = Activation('relu')(conv_4)

    # pool_4 = MaxPooling2D((2,2))(conv_4)

    # # third block of convolutions
    # conv_4 = Conv2D(inital_filter_size*4, (11,11), activation = None, kernel_initializer = 'he_normal', padding = 'same') (pool_4)
    # conv_4 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_4, training = use_dropout)
    # conv_4 = BatchNormalization()(conv_4)
    # conv_4 = Activation('relu')(conv_4)

    # conv_4 = Conv2D(inital_filter_size*4, (11,11), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv_4)
    # conv_4 = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_4, training = use_dropout)
    # conv_4 = BatchNormalization()(conv_4)
    # conv_4 = Activation('relu')(conv_4)

    # pool_4 = MaxPooling2D((2,2))(conv_4)

    # to_dense = tf.keras.layers.Concatenate()([pool_1,pool_2,pool_3,pool_4])

    flattened = tf.keras.layers.Flatten()(pool_3)
    
    d = Dense(1024,activation='relu')(flattened)

    d = Dropout(0.3)(d, training = use_dropout)

    d = Dense(128,activation='relu')(flattened)

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

def load_rotated_minst_dataset(seed = None):

    (X,labels),(_,_) = tf.keras.datasets.mnist.load_data()

    only_1 = [labels==1][0]

    labels = labels[only_1]
    X = X[only_1]

    X = X[3]

    X_out = []
    y_out = []

    if seed is not None:
        np.random.seed(seed)

    y_out = np.asarray(np.random.randint(low = -90, high = 90, size = 1500))

    for count,this_angle in enumerate(y_out):
        rot = imutils.rotate(X, angle=this_angle)
        X_out.append(rot)

    X_out = np.asarray(X_out)
    y_out = np.asarray(y_out)

    X_out_test = X_out[-200:]
    y_out_test = y_out[-200:]

    X_out_val = X_out[-500:-200]
    y_out_val = y_out[-500:-200]

    X_out = X_out[:1000]
    y_out = y_out[:1000]

    return (X_out,y_out),(X_out_val,y_out_val) ,(X_out_test,y_out_test)