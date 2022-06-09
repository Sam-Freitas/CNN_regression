from tqdm import tqdm
from natsort import natsorted, natsort_keygen
import matplotlib.pyplot as plt
import albumentations as A
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from dense_model import fully_connected_dense_modelv2, plot_model,test_on_improved_val_lossv3,diff_func
import json
from scipy import stats
from numpy.polynomial import Polynomial
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
import random
import glob

print('reading in data')
plt.ioff()

this_tissue = 'Blood;PBMC'

temp = np.load('dense_regression/data_arrays/test.npz')
X_test,y_test = temp['X'],temp['y']

num = (X_test.shape[1],X_test.shape[2])
age_normalizer = 1

epochs = 5
batch_size = 128

k_folds = glob.glob(os.path.join('dense_regression/data_arrays','*.npz'))
num_k_folds = 0
for npzs in k_folds:
    if 'train' in npzs:
        num_k_folds += 1

temp = np.load('dense_regression/data_arrays/All_data.npz')
X_norm,y_norm = temp['X'],temp['y']

training_histories = []

for i in range(num_k_folds):
    temp = np.load('dense_regression/data_arrays/train'+ str(i) +'.npz')
    train_idx = temp['idx']
    temp = np.load('dense_regression/data_arrays/val'+ str(i) +'.npz')
    val_idx = temp['idx']

    X_train, y_train = X_norm[train_idx],y_norm[train_idx]
    X_val, y_val = X_norm[val_idx],y_norm[val_idx]

    X_train, y_train = diff_func(X_train, y_train,age_normalizer=age_normalizer)
    X_val, y_val = diff_func(X_val, y_val,age_normalizer=age_normalizer)

    model = fully_connected_dense_modelv2(num_features = num,use_dropout=True,dropout_amount = 0.1)
    sample_weights = (np.abs(y_train)+1)**(1/2)

    save_checkpoints = tf.keras.callbacks.ModelCheckpoint(
        filepath = 'dense_regression/checkpoints/checkpoints' + str(i) + '/cp.ckpt', monitor = 'val_loss',
        mode = 'min',save_best_only = True,save_weights_only = True, verbose = 1)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 150)
    Reduce_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=100)
    on_epoch_end = test_on_improved_val_lossv3()
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001,amsgrad=False) # 0.001
    model.compile(optimizer=optimizer,loss='MeanAbsoluteError',metrics=['RootMeanSquaredError'])

    inital_epoch = (i*epochs)
    this_epochs = inital_epoch + epochs

    model.summary()

    history = model.fit([X_train],y_train,
        validation_data = ([X_val],y_val),
        batch_size=batch_size,epochs=this_epochs, initial_epoch = inital_epoch,
        callbacks=[on_epoch_end,save_checkpoints,earlystop,Reduce_LR],
        verbose=1,
        sample_weight = sample_weights) 

    training_histories.append(history)

    model.save_weights('dense_regression/model_weights/model_weights' + str(i) + '/model_weights')

    del model

# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
	stackX = None
	for model in members:
		# make prediction
		yhat = model.predict(inputX,batch_size=batch_size)
		# stack predictions into [rows, members, probabilities]
		if stackX is None:
			stackX = yhat
		else:
			stackX = np.dstack((stackX, yhat))
	# flatten predictions to [rows, members x probabilities]
	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
	return stackX

models = []
count = 0
for i in range(num_k_folds):

    this_train_hist = training_histories[i].history['val_loss']

    if (np.max(this_train_hist) - np.min(this_train_hist)) > (1/age_normalizer):

        models.append(fully_connected_dense_modelv2(num_features = num,use_dropout=True,dropout_amount = 0.1))

        checkpoint_path = 'dense_regression/checkpoints/checkpoints' + str(i) + '/cp.ckpt'
        models[count].load_weights(checkpoint_path)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True) # 0.00001
        models[count].compile(optimizer=optimizer,loss='MeanAbsoluteError',metrics=['RootMeanSquaredError'])

        count = count + 1

print("using:", count, "of the cross valid")
X_train, y_train = diff_func(X_norm, y_norm,age_normalizer=age_normalizer)

ensemble_prediction_test = stacked_dataset(models, X_test)
ensemble_prediction_train = stacked_dataset(models, X_train)

avg_prediction_test = np.mean(ensemble_prediction_test,axis = 1)
avg_prediction_train = np.mean(ensemble_prediction_train,axis = 1)

cor_matrix = np.corrcoef(avg_prediction_test.squeeze(),y_test)
cor_xy = cor_matrix[0,1]
r_squared_test = round(cor_xy**2,4)
print("Test",r_squared_test)

cor_matrix = np.corrcoef(avg_prediction_train.squeeze(),y_train)
cor_xy = cor_matrix[0,1]
r_squared_train = round(cor_xy**2,4)
print("Train",r_squared_train)

test_error = np.sum(np.abs(y_test-avg_prediction_test))/len(y_test)
train_error = np.sum(np.abs(y_train-avg_prediction_train))/len(y_train)
m, b = np.polyfit(y_test,avg_prediction_test, deg = 1)

plt.figure(1)
plt.scatter(y_train,avg_prediction_train,color = 'r',alpha=0.05, label = 'training data')
plt.scatter(y_test,avg_prediction_test,color = 'b',alpha=0.3, label = 'testing data')
plt.plot(np.linspace(np.min(y_train), np.max(y_train)),np.linspace(np.min(y_train), np.max(y_train)))
plt.plot(y_test, m*y_test + b,'m-')

plt.text(np.min(y_train),np.max(y_train),"r^2: " + str(r_squared_train),fontsize = 12, color = 'r')
plt.text(np.min(y_train),np.max(y_train)-(0.2)*np.max(y_train),"r^2: " + str(r_squared_test),fontsize = 12, color = 'b')

plt.legend(loc = 'upper center')
plt.title('Train error: ' + str(round(train_error,4)) + ' --- Test error: ' + str(round(test_error,4)),fontsize = 10)
plt.xlabel('Expected Age (years)')
plt.ylabel('Predicted Age (years)')

plt.savefig(fname = "dense_regression/test_model_predictions" + str(this_tissue).replace('/','-') + ".png")

plt.close('all')