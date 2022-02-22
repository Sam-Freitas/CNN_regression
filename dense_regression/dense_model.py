from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, BatchNormalization, Activation, Dense, LSTMCell
from tensorflow.keras import backend as K
from tensorflow.python.keras.engine import training
from skimage import measure
from tqdm import tqdm
from natsort import natsorted
import matplotlib.pyplot as plt
import albumentations as A
import tensorflow as tf
import numpy as np
import shutil
import random
import glob
import sys
import cv2
import os

def fully_connected_dense_model(num_features = 2048, use_dropout = False):

    inputs_data = Input(shape = (num_features,))

    s = Dense(num_features)(inputs_data)
    d = Activation('gelu')(s)
    d = Dropout(0.8)(d, training = use_dropout)
    d = Dense(2048)(d)
    d = Activation('gelu')(d)
    d = Dropout(0.8)(d, training = use_dropout)
    d = Dense(128)(d)
    d = Activation('gelu')(d)
    d = Dropout(0.8)(d, training = use_dropout)

    inputs_metadata = Input(shape = (2,)) # sex, tissue type

    sm = Dense(2)(inputs_metadata)
    dm = Activation('gelu')(sm)
    dm = Dropout(0.8)(dm)
    dm = Dense(128)(dm)
    dm = Activation('gelu')(dm)
    dm = Dropout(0.8)(dm)

    cat_layer = tf.keras.layers.Concatenate()([d,dm])

    output = Dense(1,activation='linear')(cat_layer)

    model = Model(inputs=[inputs_data,inputs_metadata], outputs=[output])

    return model

def fully_connected_dense_model_old(num_features = 2048, use_dropout = False):

    inputs_data = Input(shape = (num_features,))

    s = Dense(num_features)(inputs_data)
    d = Activation('gelu')(s)
    d = Dropout(0.8)(d, training = use_dropout)
    d = Dense(2048)(d)
    d = Activation('gelu')(d)
    d = Dropout(0.8)(d, training = use_dropout)

    output = Dense(1,activation='linear')(d)

    model = Model(inputs=[inputs_data], outputs=[output])

    return model