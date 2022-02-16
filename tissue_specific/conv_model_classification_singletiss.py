import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob
from natsort import natsorted, natsort_keygen
import matplotlib.pyplot as plt
import PIL
import cv2
import json

from tensorflow import keras
from tensorflow.keras import layers, datasets, models, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pathlib
import time

from sklearn.model_selection import LeaveOneOut,KFold, train_test_split,StratifiedKFold,StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from CNN_regression_model import fully_connected_CNN

from tqdm import tqdm

print('Loading in metadata')
imgs_location = '/home/u23/samfreitas/NN_trainings/IGTD/Results/Test_1_std_sorted/data'
metadata = pd.read_csv('/home/u23/samfreitas/NN_trainings/IGTD/Data/experimentDesign.csv')

sorted_metadata = metadata.copy(deep=True)
sorted_metadata = sorted_metadata.sort_values(by = 'sample_id',key = natsort_keygen())

unique_tissues = np.unique(sorted_metadata['tissue'].values)
output = pd.DataFrame()

for this_tissue in unique_tissues:

    # attempt with using single tissue data
    single_tissue_index = (sorted_metadata['tissue'].values == this_tissue)
    metadata_single_tissue = sorted_metadata.iloc[single_tissue_index]
    extracted_metadata = metadata_single_tissue['age'].values
    encoded_labels = extracted_metadata
    train_y = extracted_metadata
    num_classes = 1

    imgs_list = natsorted(glob.glob(os.path.join(imgs_location,'*.txt')))

    print('Loading in data from txt files')
    train_X = []
    for count in tqdm(range(len(imgs_list))):

        this_img = imgs_list[count]

        # this was with single tissue sample
        img_sample_id = os.path.basename(os.path.splitext(this_img)[0])[1:-5]
        if img_sample_id in metadata_single_tissue['sample_id'].values:
            temp_img = np.loadtxt(this_img, comments="#", delimiter="\t", unpack=False)
            # temp_img = np.stack((temp_img,)*3, axis=-1)
            train_X.append(temp_img)

    train_X = np.asarray(train_X)

    skf = StratifiedKFold(n_splits=3)

    print('Building models')

    kfold_counter = 1
    for train_index, test_index in skf.split(train_X, train_y):

        model = fully_connected_CNN(height=100,width=100,channels=1,use_dropout=True)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01), # learning_rate = 0.1
                    # loss=tf.keras.losses.CategoricalCrossentropy(),
                    loss=tf.keras.losses.MeanAbsoluteError(),
                    # metrics = ['accuracy',AUC_PR,AUC_ROC]
                    metrics = ['accuracy']
                    )

        checkpoint_path = "model_weights/cp.ckpt" 
        checkpoint_dir = os.path.dirname(checkpoint_path)

        epochs = 500
        checkpoint = ModelCheckpoint(filepath = checkpoint_path,monitor = "val_loss", mode = 'min',
            save_best_only = True,verbose=1,save_weights_only=True) #use checkpoint instead of sequential() module
        earlystop = EarlyStopping(monitor = 'val_loss', min_delta=0.01,
            patience = 500, verbose = 1,restore_best_weights = True) #stop at best epoch
        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor=0.5,
            patience=50, min_lr=0.000001, verbose = 1)

        # first split into training and testing datasets
        kfold_X = train_X[train_index]
        kfold_y = train_y[train_index]
        # this test is not going to be used in the training
        kfold_test = (train_X[test_index],train_y[test_index])

        print("fitting model", this_tissue, kfold_counter)

        # fit the model
        history = model.fit(
            kfold_X,kfold_y,
            validation_data = kfold_test,
            epochs=epochs,
            batch_size = test_index.size,
            callbacks= [earlystop, checkpoint],
            verbose = 0
        )

        # load in the best checkpoint
        model.load_weights(checkpoint_path)

        # evaluate results
        eval_result = model.evaluate(train_X,train_y,steps=1,return_dict = True,batch_size = 1) #get evaluation results
        eval_result['tissue'] = this_tissue
        eval_result['Kfold'] = kfold_counter
        
        # export evaluation reults to a dict and then print it
        res = dict()
        for key in eval_result: res[key] = eval_result[key]
        print("Overall performance")
        print(res)

        plt.figure(figsize = (12,12), dpi = 96)
        for this_key in list(history.history.keys()):
            b = history.history[this_key]
            plt.plot(b,label = this_key)

        plt.legend(loc="upper left")
        plt.ylim([0,25])
        plt.title(json.dumps(res))
        plt.savefig(fname= str(this_tissue) + "_training_history_" + str(kfold_counter) + ".png")
        # plt.show()

        plt.close('all')

        kfold_counter = kfold_counter + 1

        print("Examples")
        for i in range(10):
            print(train_y[i],model.predict(np.expand_dims(train_X[i],axis = 0)).squeeze()) 

        eval_result['total_error'] = np.sum(np.abs(train_y - model.predict(train_X).squeeze() ))

        output = output.append(eval_result, ignore_index=True)

        del model

        output.to_csv("output.csv",index = False)

    

print('eof')