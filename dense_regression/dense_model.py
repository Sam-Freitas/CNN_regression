from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, AlphaDropout, Dropout, Lambda, Conv1D, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, BatchNormalization, Activation, Dense, LSTMCell
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
import json
import sys
import cv2
import os

def fully_connected_dense_model(num_features = 2048, num_cat = 21, use_dropout = False, dropout_amount = 0.8):

    # rna-seq data input
    inputs_data = Input(shape = (num_features,))
    s = Dense(num_features+1,input_shape = inputs_data.shape)(inputs_data)
    d = Activation('sigmoid')(s)
    d = Dropout(dropout_amount)(d, training = use_dropout)
    d = Dense(6000)(d)
    d = Activation('elu')(d)
    d = Dropout(dropout_amount)(d, training = use_dropout)
    # metdata inputs 
    inputs_metadata = Input(shape = (2,)) # sex, tissue type
    sm = Dense(512,input_shape = inputs_metadata.shape)(inputs_metadata)
    dm = Activation('elu')(sm)

    # block outputs data
    d_output = Dense(1,activation='linear')(d)
    dm_output = Dense(1,activation='linear')(dm)

    #concatinations
    out_concat = tf.keras.layers.Add()([d_output,dm_output])

    # final output layes for data exportation
    output = Dense(1,activation='linear',name = 'coninious_output')(out_concat)

    model = Model(inputs=[inputs_data,inputs_metadata], outputs=[output])#,out_categorical])

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


class test_on_improved_val_loss(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):

        curr_path = os.path.split(__file__)[0]

        curr_val_loss = logs['val_loss']
        try:
            val_loss_hist = self.model.history.history['val_loss']
        except:
            val_loss_hist = curr_val_loss + 1

        if epoch == 0:
            try:
                os.mkdir(os.path.join(curr_path, 'output_images_testing_during'))
            except:
                shutil.rmtree(os.path.join(curr_path, 'output_images_testing_during'))
                os.mkdir(os.path.join(curr_path, 'output_images_testing_during'))

        if curr_val_loss <= np.min(val_loss_hist) or epoch == 0:
            print("val_loss improved to:",curr_val_loss)

        if (epoch % 25) == 0:

            temp = np.load(os.path.join(curr_path,'data_arrays','test.npz'))
            X_test,X_meta_test,y_test,y_test_cat,bins = temp['X'],temp['X_meta'],temp['y'],temp['y_cat'],temp['bins']

            eval_result = self.model.evaluate([X_test,X_meta_test],[y_test],batch_size=1,verbose=0,return_dict=True)
            print(eval_result)

            # plt.figure(1)
            plt.close('all')

            predicted = self.model.predict([X_test,X_meta_test],batch_size=1).squeeze()

            cor_matrix = np.corrcoef(predicted,y_test)
            cor_xy = cor_matrix[0,1]
            r_squared = round(cor_xy**2,4)
            print("Current r_squared test:",r_squared)

            res = dict()
            for key in eval_result: res[key] = round(eval_result[key],6)

            plt.scatter(y_test,predicted,color = 'r',alpha=0.2)
            plt.plot(np.linspace(np.min(y_test), np.max(y_test)),np.linspace(np.min(y_test), np.max(y_test)))
            plt.text(np.min(y_test),np.max(y_test),"r^2: " + str(r_squared),fontsize = 12)
            plt.title(json.dumps(res).replace(',', '\n'),fontsize = 10)
            plt.xlabel('Expected Age (years)')
            plt.ylabel('Predicted Age (years)')

            output_name = os.path.join(curr_path,'output_images_testing_during',str(epoch) + '_' + str(r_squared)[2:] + '.png')

            plt.savefig(fname = output_name)

            plt.close('all')