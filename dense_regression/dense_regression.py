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
from dense_model import fully_connected_dense_model, plot_model,test_on_improved_val_loss
import json
from scipy import stats
from numpy.polynomial import Polynomial
from sklearn.model_selection import train_test_split
import random

# set up variables 

def scheduler(epoch, lr):
    if (epoch % 20 == 0) and epoch > 0:
        lr = lr*.9
        print(round(lr,10))
    return lr

num = 10000
this_tissue = 'All_tissues'

temp = np.load('dense_regression/data_arrays/train.npz')
X_train,X_meta_train,y_train = temp['X'],temp['X_meta'],temp['y']
temp = np.load('dense_regression/data_arrays/val.npz')
X_val,X_meta_val,y_val = temp['X'],temp['X_meta'],temp['y']
temp = np.load('dense_regression/data_arrays/test.npz')
X_test,X_meta_test,y_test = temp['X'],temp['X_meta'],temp['y']

X_all = np.concatenate((X_train,X_val,X_test))
X_meta_all = np.concatenate((X_meta_train,X_meta_val,X_meta_test))
y_all = np.concatenate((y_train,y_val,y_test))

# del data, metadata, metadata_healthy, single_tissue_index, sorted_std_idx_ascend, SRR_values, this_metadata, this_imgs_meta_idx, srr_id, this_data

print('Setting up model')
model = fully_connected_dense_model(num_features = num, use_dropout=True,dropout_amount = 0.25)
plot_model(model)

epochs = 10000

save_checkpoints = tf.keras.callbacks.ModelCheckpoint(
    filepath = 'dense_regression/model_weights/cp.ckpt', monitor = 'val_loss',
    mode = 'min',save_best_only = True,save_weights_only = True, verbose = 1)
redule_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_loss', factor = 0.9, patience = 50, min_lr = 0, verbose = 1)
earlystop = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',min_delta = 0.01,patience = 1000, verbose = 1)
on_epoch_end = test_on_improved_val_loss()
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

# optimizer = tf.keras.optimizers.RMSprop(momentum=0.75)#, momentum=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
loss = tf.keras.losses.MeanAbsoluteError()

model.compile(optimizer=optimizer,loss=loss,metrics=['MSE'])

model.summary()

history = model.fit([X_train,X_meta_train],y_train,
    validation_data = ([X_val,X_meta_val],y_val),
    batch_size=32,epochs=epochs,
    callbacks=[save_checkpoints,earlystop,redule_lr,on_epoch_end],
    verbose=1)

del model

model = fully_connected_dense_model(num_features = num, use_dropout=False)
model.load_weights('dense_regression/model_weights/cp.ckpt')
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer=optimizer,loss='MAE',metrics=['MSE'])

eval_result = model.evaluate([X_test,X_meta_test],y_test,batch_size=1,verbose=1,return_dict=True)
print(eval_result)

plt.figure(1)

predicted = model.predict([X_test,X_meta_test],batch_size=1).squeeze()

cor_matrix = np.corrcoef(predicted.squeeze(),y_test)
cor_xy = cor_matrix[0,1]
r_squared = round(cor_xy**2,4)
print("test",r_squared)

res = dict()
for key in eval_result: res[key] = round(eval_result[key],6)

model.save('dense_regression/compiled_models/' + str(r_squared)[2:] + this_tissue + '_' + str(num))

plt.scatter(y_test,predicted,color = 'r',alpha=0.2)
plt.plot(np.linspace(np.min(y_test), np.max(y_test)),np.linspace(np.min(y_test), np.max(y_test)))
plt.text(np.min(y_test),np.max(y_test),"r^2: " + str(r_squared),fontsize = 12)
plt.title(json.dumps(res))
plt.xlabel('Expected Age (years)')
plt.ylabel('Predicted Age (years)')

plt.savefig(fname = "dense_regression/test_model_predictions" + str(this_tissue).replace('/','-') + ".png")

plt.close('all')

eval_result = model.evaluate([X_all,X_meta_all],y_all,batch_size=1,verbose=1,return_dict=True)
print(eval_result)

plt.figure(2)

predicted = model.predict([X_all,X_meta_all],batch_size=1).squeeze()

cor_matrix = np.corrcoef(predicted.squeeze(),y_all)
cor_xy = cor_matrix[0,1]
r_squared = round(cor_xy**2,4)
print("train",r_squared)

res = dict()
for key in eval_result: res[key] = round(eval_result[key],6)

plt.scatter(y_all,predicted,color = 'r',alpha=0.2)
plt.plot(np.linspace(np.min(y_all), np.max(y_all)),np.linspace(np.min(y_all), np.max(y_all)))
plt.text(np.min(y_all),np.max(y_all),"r^2: " + str(r_squared),fontsize = 12)
plt.title(json.dumps(res))
plt.xlabel('Expected Age (years)')
plt.ylabel('Predicted Age (years)')

plt.savefig(fname = "dense_regression/train_model_predictions" + str(this_tissue).replace('/','-') + ".png")

plt.close('all')

plt.figure(3)
for this_key in list(history.history.keys()):
    b = history.history[this_key]
    plt.plot(b,label = this_key)

plt.legend(loc="upper left")
plt.ylim([0,50])
plt.title(json.dumps(res))
plt.savefig(fname=  "dense_regression/training_history" + str(this_tissue).replace('/','-') + ".png")

plt.close('all')

print('eof')