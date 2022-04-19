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
from CNN_regression_model import fully_connected_CNN_v2, fully_connected_CNN_v3,plot_model,test_on_improved_val_lossv3
from sklearn.preprocessing import PowerTransformer

print('reading in data')

this_tissue = 'Blood;PBMC'

temp = np.load('data_arrays/test.npz')
X_test,y_test = temp['X'],temp['y']

inital_filter_size = 8
dropsize = 0.85
blocksize = 5
layers = 3
sublayers = 0

# epochs = 15000
epochs = 5
batch_size = 1

def scheduler(epoch, lr):
    if epoch < 2000:
        lr = 0.001
    elif epoch > 1999 and epoch < 4000:
        lr = 0.00001
    else:
        lr = 0.0000001
    return lr


for i in range(3):
    temp = np.load('data_arrays/train'+ str(i) +'.npz')
    X_train,y_train = temp['X'],temp['y']
    temp = np.load('data_arrays/val'+ str(i) +'.npz')
    X_val,y_val = temp['X'],temp['y']

    model = fully_connected_CNN_v3(
    height=X_train.shape[1],width=X_train.shape[2],channels=2,
    use_dropout=True,inital_filter_size=inital_filter_size,dropsize = dropsize,blocksize = blocksize,
    layers = layers, sub_layers = sublayers
    )
    sample_weights = (np.abs(y_train)+1)**(1/2)

    save_checkpoints = tf.keras.callbacks.ModelCheckpoint(
        filepath = 'checkpoints' + str(i) + '/cp.ckpt', monitor = 'val_loss',
        mode = 'min',save_best_only = True,save_weights_only = True, verbose = 1)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    on_epoch_end = test_on_improved_val_lossv3()
    optimizer = tf.keras.optimizers.Adam() # 0.00001
    model.compile(optimizer=optimizer,loss='MeanAbsoluteError',metrics=['MeanSquaredError','RootMeanSquaredError'])

    history = model.fit([X_train],y_train,
        validation_data = ([X_val],y_val),
        batch_size=batch_size,epochs=epochs, initial_epoch = 0,
        callbacks=[on_epoch_end,save_checkpoints,lr_scheduler],
        verbose=1,
        sample_weight = sample_weights) 

    model.save_weights('model_weights' + str(i) + '/model_weights')

del model

# print("earlystop weights")

# model = fully_connected_CNN_v3(
#     height=X_train.shape[1],width=X_train.shape[2],
#     use_dropout=False,inital_filter_size=inital_filter_size,dropsize = dropsize,blocksize = blocksize,
#     layers = layers, sub_layers = sublayers
#     )
# model.compile(optimizer=optimizer,loss='MeanAbsoluteError',metrics=['MeanSquaredError','RootMeanSquaredError'])
# model.load_weights('model_weights/model_weights')

# eval_result = model.evaluate([X_test],[y_test],batch_size=1,verbose=0,return_dict=True)
# print(eval_result)

# res = dict()
# for key in eval_result: res[key] = round(eval_result[key],6)

# plt.figure(1)

# predicted_test = model.predict([X_test],batch_size=1).squeeze()
# predicted_train = model.predict([X_all],batch_size=1).squeeze()

# cor_matrix = np.corrcoef(predicted_test.squeeze(),y_test)
# cor_xy = cor_matrix[0,1]
# r_squared_test = round(cor_xy**2,4)
# print("test",r_squared_test)

# cor_matrix = np.corrcoef(predicted_train.squeeze(),y_all)
# cor_xy = cor_matrix[0,1]
# r_squared_train = round(cor_xy**2,4)
# print("train",r_squared_train)

# model.save('compiled_models/' + str(r_squared_test)[2:] + this_tissue + '_' + str(num))

# plt.scatter(y_all,predicted_train,color = 'r',alpha=0.2, label = 'training data')
# plt.scatter(y_test,predicted_test,color = 'b',alpha=0.3, label = 'testing data')
# plt.plot(np.linspace(np.min(y_all), np.max(y_all)),np.linspace(np.min(y_all), np.max(y_all)))

# plt.text(np.min(y_all),np.max(y_all),"r^2: " + str(r_squared_train),fontsize = 12, color = 'r')
# plt.text(np.min(y_all),np.max(y_all)-(0.2)*np.max(y_all),"r^2: " + str(r_squared_test),fontsize = 12, color = 'b')

# plt.legend(loc = 'upper center')
# plt.title(json.dumps(res).replace(',','\n'),fontsize = 10)
# plt.xlabel('Expected Age (years)')
# plt.ylabel('Predicted Age (years)')

# plt.savefig(fname = "test_model_predictions" + str(this_tissue).replace('/','-') + ".png")

# plt.close('all')

# plt.figure(3)
# for this_key in list(history.history.keys()):
#     b = history.history[this_key]
#     plt.plot(b,label = this_key)

# plt.legend(loc="upper left")
# plt.ylim([0,30])
# plt.title(json.dumps(res))
# plt.savefig(fname=  "training_history" + str(this_tissue).replace('/','-') + ".png")

# plt.close('all')

# print("Restored weights")
# del model

# model = fully_connected_CNN_v3(
#     height=X_train.shape[1],width=X_train.shape[2],
#     use_dropout=False,inital_filter_size=inital_filter_size,dropsize = dropsize,blocksize = blocksize,
#     layers = layers, sub_layers = sublayers
#     )
# model.compile(optimizer=optimizer,loss='MeanAbsoluteError',metrics=['MeanSquaredError','RootMeanSquaredError'])
# model.load_weights('checkpoints/cp.ckpt')

# eval_result = model.evaluate([X_test],[y_test],batch_size=1,verbose=0,return_dict=True)
# print(eval_result)

# res = dict()
# for key in eval_result: res[key] = round(eval_result[key],6)

# plt.figure(1)

# predicted_test = model.predict([X_test],batch_size=1).squeeze()
# predicted_train = model.predict([X_all],batch_size=1).squeeze()

# cor_matrix = np.corrcoef(predicted_test.squeeze(),y_test)
# cor_xy = cor_matrix[0,1]
# r_squared_test = round(cor_xy**2,4)
# print("test",r_squared_test)

# cor_matrix = np.corrcoef(predicted_train.squeeze(),y_all)
# cor_xy = cor_matrix[0,1]
# r_squared_train = round(cor_xy**2,4)
# print("train",r_squared_train)

# model.save('compiled_models/' + str(r_squared_test)[2:] + this_tissue + '_' + str(num))

# plt.scatter(y_all,predicted_train,color = 'r',alpha=0.2, label = 'training data')
# plt.scatter(y_test,predicted_test,color = 'b',alpha=0.3, label = 'testing data')
# plt.plot(np.linspace(np.min(y_all), np.max(y_all)),np.linspace(np.min(y_all), np.max(y_all)))

# plt.text(np.min(y_all),np.max(y_all),"r^2: " + str(r_squared_train),fontsize = 12, color = 'r')
# plt.text(np.min(y_all),np.max(y_all)-(0.2)*np.max(y_all),"r^2: " + str(r_squared_test),fontsize = 12, color = 'b')

# plt.legend(loc = 'upper center')
# plt.title(json.dumps(res).replace(',','\n'),fontsize = 10)
# plt.xlabel('Expected Age (years)')
# plt.ylabel('Predicted Age (years)')

# plt.savefig(fname = "test_model_predictions_best" + str(this_tissue).replace('/','-') + ".png")

# plt.close('all')

# plt.close('all')

# print('eof')