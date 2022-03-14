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
num = X_train.shape[1]

inital_filter_size = 8
dropsize = 0.85
blocksize = 5

model = fully_connected_CNN_v2(
    height=X_train.shape[1],width=X_train.shape[2],
    use_dropout=True,inital_filter_size=inital_filter_size,dropsize = dropsize,blocksize = blocksize
    )
plot_model(model)

epochs = 100
batch_size = 16

save_checkpoints = tf.keras.callbacks.ModelCheckpoint(
    filepath = 'checkpoints/cp.ckpt', monitor = 'val_loss',
    mode = 'min',save_best_only = True,save_weights_only = True, verbose = 1)
redule_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_loss', factor = 0.1, patience = 250, min_lr = 0, verbose = 1)
earlystop = tf.keras.callbacks.EarlyStopping(restore_best_weights=False,
    monitor = 'val_loss',min_delta = 0,patience = 5000, verbose = 1)

on_epoch_end = test_on_improved_val_loss()

# optimizer = tf.keras.optimizers.RMSprop(momentum=0.75)#, momentum=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer=optimizer,loss='MeanSquaredError',metrics=['RootMeanSquaredError'])

history = model.fit([X_train,X_meta_train],y_train,
    validation_data = ([X_val,X_meta_val],y_val),
    batch_size=batch_size,epochs=epochs,
    callbacks=[earlystop,on_epoch_end,save_checkpoints,redule_lr],
    verbose=1)

model.save_weights('model_weights/model_weights')

del model

print("earlystop weights")

model = fully_connected_CNN_v2(
    height=X_train.shape[1],width=X_train.shape[2],use_dropout=False,
    inital_filter_size=inital_filter_size,dropsize = dropsize,blocksize = blocksize
    )
model.compile(optimizer=optimizer,loss='MeanSquaredError',metrics=['RootMeanSquaredError'])
model.load_weights('model_weights/model_weights')

eval_result = model.evaluate([X_test,X_meta_test],[y_test],batch_size=1,verbose=0,return_dict=True)
print(eval_result)

res = dict()
for key in eval_result: res[key] = round(eval_result[key],6)

plt.figure(1)

predicted_test = model.predict([X_test,X_meta_test],batch_size=1).squeeze()
predicted_train = model.predict([X_all,X_meta_all],batch_size=1).squeeze()

cor_matrix = np.corrcoef(predicted_test.squeeze(),y_test)
cor_xy = cor_matrix[0,1]
r_squared_test = round(cor_xy**2,4)
print("test",r_squared_test)

cor_matrix = np.corrcoef(predicted_train.squeeze(),y_all)
cor_xy = cor_matrix[0,1]
r_squared_train = round(cor_xy**2,4)
print("train",r_squared_train)

model.save('compiled_models/' + str(r_squared_test)[2:] + this_tissue + '_' + str(num))

plt.scatter(y_all,predicted_train,color = 'r',alpha=0.2, label = 'training data')
plt.scatter(y_test,predicted_test,color = 'b',alpha=0.3, label = 'testing data')
plt.plot(np.linspace(np.min(y_all), np.max(y_all)),np.linspace(np.min(y_all), np.max(y_all)))

plt.text(np.min(y_all),np.max(y_all),"r^2: " + str(r_squared_train),fontsize = 12, color = 'r')
plt.text(np.min(y_all),np.max(y_all)-5,"r^2: " + str(r_squared_test),fontsize = 12, color = 'b')

plt.legend(loc = 'upper center')
plt.title(json.dumps(res).replace(',','\n'),fontsize = 10)
plt.xlabel('Expected Age (years)')
plt.ylabel('Predicted Age (years)')

plt.savefig(fname = "test_model_predictions" + str(this_tissue).replace('/','-') + ".png")

plt.close('all')

plt.figure(3)
for this_key in list(history.history.keys()):
    b = history.history[this_key]
    plt.plot(b,label = this_key)

plt.legend(loc="upper left")
plt.ylim([0,30])
plt.title(json.dumps(res))
plt.savefig(fname=  "training_history" + str(this_tissue).replace('/','-') + ".png")

plt.close('all')

print("Restored weights")
del model

model = fully_connected_CNN_v2(
    height=X_train.shape[1],width=X_train.shape[2],use_dropout=False,
    inital_filter_size=inital_filter_size,dropsize = dropsize,blocksize = blocksize
    )
model.compile(optimizer=optimizer,loss='MeanSquaredError',metrics=['RootMeanSquaredError'])
model.load_weights('checkpoints/cp.ckpt')

eval_result = model.evaluate([X_test,X_meta_test],[y_test],batch_size=1,verbose=0,return_dict=True)
print(eval_result)

res = dict()
for key in eval_result: res[key] = round(eval_result[key],6)

plt.figure(1)

predicted_test = model.predict([X_test,X_meta_test],batch_size=1).squeeze()
predicted_train = model.predict([X_all,X_meta_all],batch_size=1).squeeze()

cor_matrix = np.corrcoef(predicted_test.squeeze(),y_test)
cor_xy = cor_matrix[0,1]
r_squared_test = round(cor_xy**2,4)
print("test",r_squared_test)

cor_matrix = np.corrcoef(predicted_train.squeeze(),y_all)
cor_xy = cor_matrix[0,1]
r_squared_train = round(cor_xy**2,4)
print("train",r_squared_train)

model.save('compiled_models/' + str(r_squared_test)[2:] + this_tissue + '_' + str(num))

plt.scatter(y_all,predicted_train,color = 'r',alpha=0.2, label = 'training data')
plt.scatter(y_test,predicted_test,color = 'b',alpha=0.3, label = 'testing data')
plt.plot(np.linspace(np.min(y_all), np.max(y_all)),np.linspace(np.min(y_all), np.max(y_all)))

plt.text(np.min(y_all),np.max(y_all),"r^2: " + str(r_squared_train),fontsize = 12, color = 'r')
plt.text(np.min(y_all),np.max(y_all)-5,"r^2: " + str(r_squared_test),fontsize = 12, color = 'b')

plt.legend(loc = 'upper center')
plt.title(json.dumps(res).replace(',','\n'),fontsize = 10)
plt.xlabel('Expected Age (years)')
plt.ylabel('Predicted Age (years)')

plt.savefig(fname = "test_model_predictions_best" + str(this_tissue).replace('/','-') + ".png")

plt.close('all')

plt.close('all')

print('eof')