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
from CNN_regression_model import fully_connected_CNN_v2, fully_connected_CNN_v3, fully_connected_CNN_v4, plot_model,test_on_improved_val_lossv3,diff_func
from sklearn.preprocessing import PowerTransformer
import model_soup

def pipe(data, batch_size = 128, shuffle = False):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    if shuffle:
        dataset = dataset.shuffle(buffer_size = batch_size * 10)
    dataset = dataset.batch(batch_size)
    #dataset = dataset.prefetch((batch_size * 2) + 1)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

print('reading in data')
plt.ioff()

this_tissue = 'Blood;PBMC_10_removed_size50'

paths = os.path.join(os.getcwd(),'model_weights')
# paths = os.path.join(os.getcwd(),'checkpoints')

temp = np.load('data_arrays/test.npz')
X_test,y_test = temp['X'],temp['y']

inital_filter_size = 8
dropsize = 0.99
blocksize = 5
layers = 3
sublayers = 0
age_normalizer = 1
input_height = input_width = 50
epochs = 5
batch_size = 128

k_folds = glob.glob(os.path.join('data_arrays','*.npz'))
num_k_folds = 0
for npzs in k_folds:
    if 'train' in npzs:
        num_k_folds += 1

temp = np.load('data_arrays/All_data.npz')
X_norm,y_norm = temp['X'],temp['y']
temp = np.load('data_arrays/test.npz')
X_test,y_test = temp['X'],temp['y']

model = fully_connected_CNN_v4(
    height=input_height,width=input_width,channels=2,
    use_dropout=False,inital_filter_size=inital_filter_size,keep_prob = dropsize,blocksize = blocksize,
    layers = layers, sub_layers = sublayers
)
model.compile(optimizer = 'adam',loss = 'MAE')

model_paths = []
for i in range(num_k_folds):
    model_paths.append(os.path.join(paths,'model_weights' + str(i) + '.h5'))
    # model_paths.append(os.path.join(paths,'checkpoints' + str(i)))

X_train, y_train  = diff_func(X_norm, y_norm, age_normalizer=age_normalizer)
te_data = pipe((X_train, y_train ), batch_size = batch_size, shuffle = False)

comp = np.less_equal

print("\n[Greedy Soup (greedy weight update) Performance]")
greedy_model = model_soup.tf.greedy_soup(model, list(model_paths), te_data, metric = tf.keras.metrics.mean_absolute_error, compare = comp, update_greedy = False)
loss = greedy_model.evaluate(te_data)

print("\n[Uniform model soup Performance]")
uniform_model = model_soup.tf.uniform_soup(model, list(model_paths))
loss = uniform_model.evaluate(te_data)

ensemble_prediction_test = greedy_model.predict(X_test,batch_size = 128)
ensemble_prediction_train = greedy_model.predict(X_train, batch_size = 128)

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

plt.savefig(fname = "soup.png")

plt.close('all')