from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, BatchNormalization, Activation, Dense
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

def fully_connected_dense_model(num_features = 10000, use_dropout = False):

    inputs = Input(num_features)

    s = Dense(num_features)(inputs)
    d = Activation('gelu')(s)
    d = Dropout(0.8)(d, training = use_dropout)
    d = Dense(2048)(d)
    d = Activation('gelu')(d)
    d = Dropout(0.8)(d, training = use_dropout)
    # d = Dense(1024)(d)
    # d = Activation('gelu')(d)
    # d = Dropout(0.5)(d, training = use_dropout)

    output = Dense(1,activation='linear')(d)

    model = Model(inputs=[inputs], outputs=[output])

    return model