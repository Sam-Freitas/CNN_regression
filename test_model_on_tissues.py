import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import json
import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from natsort import natsorted, natsort_keygen
from CNN_regression_model import fully_connected_CNN_v2, plot_model,test_on_improved_val_loss
from sklearn.preprocessing import PowerTransformer

print('reading in metadata')

this_tissue = 'All_tissues;'

dataset = ''

temp = np.load('data_arrays/train.npz')
X_train,X_meta_train,y_train = temp['X'],temp['X_meta'],temp['y']
temp = np.load('data_arrays/val.npz')
X_val,X_meta_val,y_val = temp['X'],temp['X_meta'],temp['y']
temp = np.load('data_arrays/test.npz')
X_test,X_meta_test,y_test = temp['X'],temp['X_meta'],temp['y']

X_all = np.concatenate((X_train,X_val))
X_meta_all = np.concatenate((X_meta_train,X_meta_val))
y_all = np.concatenate((y_train,y_val))

