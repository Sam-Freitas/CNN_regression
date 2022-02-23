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
    d = Dense(4)(d)
    d = Activation('gelu')(d)
    d = Dropout(0.25)(d, training = use_dropout)

    inputs_metadata = Input(shape = (2,)) # sex, tissue type

    sm = Dense(2)(inputs_metadata)
    dm = Activation('gelu')(sm)
    dm = Dropout(0.5)(dm)

    cat_layer = tf.keras.layers.Concatenate()([d,dm])

    # out_cat = Dense(256)(cat_layer)
    # out_cat = Activation('gelu')(out_cat)
    # out_cat = Dropout(0.5)(out_cat)

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

def plot_model(model):
    try:
        tf.keras.utils.plot_model(
            model, to_file='dense_regression/model.png', show_shapes=True, show_dtype=True,
            show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)
    except:
        print("Exporting model to png failed")
        print("Necessary packages: pydot (pip) and graphviz (brew)")