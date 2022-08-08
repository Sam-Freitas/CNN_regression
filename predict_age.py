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
from pathlib import Path

def diff_func_stripe(X_norm,data_to_stripe):

    print('Diff function generation')
    y_diff = []
    X_diff = []
    num_loops = X_norm.shape[0]
    count = 0

    X1 = data_to_stripe
    for j in range(num_loops):
        X2 = X_norm[j]
        X_diff.append(np.concatenate([np.atleast_3d(X1),np.atleast_3d(X2)],axis = -1).squeeze())
        count = count + 1
    X_diff = np.asarray(X_diff)

    return X_diff

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

print('reading in data')
plt.ioff()

this_tissue = 'age_predict'

inital_filter_size = 16
dropsize = 0.95
blocksize = 5
layers = 3
sublayers = 0
age_normalizer = 1
input_height = input_width = 130
dense_size = 1024

epochs = 1000
batch_size = 128

checkpoints_path = os.path.join(os.getcwd(),'checkpointsremove_15_size130_dense128')

k_folds = glob.glob(os.path.join('data_arrays','*.npz'))
num_k_folds = 0
for npzs in k_folds:
    if 'train' in npzs:
        num_k_folds += 1

temp = np.load('data_arrays/All_data.npz')
X_norm,y_norm = temp['X'],temp['y']
temp = np.load('data_arrays/test_no_diff.npz')
X_test,y_test = temp['X'],temp['y']

assert X_norm.shape[1] == input_height and input_height == X_norm.shape[2]

models = []
count = 0
print('Stacking models')
for i in range(num_k_folds):

    models.append(
        fully_connected_CNN_v4(
        height=input_height,width=input_width,channels=2,
        use_dropout=False,inital_filter_size=inital_filter_size,keep_prob = dropsize,blocksize = blocksize,
        layers = layers, sub_layers = sublayers,dense_size = dense_size)
    )
    checkpoint_path = os.path.join(checkpoints_path,'checkpoints' + str(i) + '/cp.ckpt')
    models[count].load_weights(checkpoint_path)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True) # 0.00001
    models[count].compile(optimizer=optimizer,loss='MeanAbsoluteError',metrics=['RootMeanSquaredError'])

    count = count + 1

# print("using:", count, "of the cross valid")
out = []
for count, sample in enumerate(X_test):
    sample_expanded = diff_func_stripe(X_norm,sample)
    predicted_diffs = stacked_dataset(models,sample_expanded)
    avg_predicted_diff = np.mean(predicted_diffs,axis = 1)

    predicted_ages = y_norm + avg_predicted_diff

    avg_predicted_age = np.mean(predicted_ages)
    std_predicted_age = np.std(predicted_ages)

    out.append(avg_predicted_age)

y_out = np.asarray(out)

cor_matrix = np.corrcoef(y_test,y_out)
cor_xy = cor_matrix[0,1]
r_squared_test = round(cor_xy**2,4)
print("Test",r_squared_test)


#################### this doesnt work yet

test_error = np.sum(np.abs(y_test-y_out))/len(y_test)
m, b = np.polyfit(y_test,y_out, deg = 1)

plt.figure(1)
plt.scatter(y_test,y_out,color = 'r', label = 'training data')
plt.plot(np.linspace(np.min(y_test), np.max(y_test)),np.linspace(np.min(y_test), np.max(y_test)))
plt.plot(y_test, m*y_test + b,'m-')

plt.text(np.min(y_test),np.max(y_test)-(0.2)*np.max(y_test),"r^2: " + str(r_squared_test),fontsize = 12, color = 'b')

plt.legend(loc = 'upper center')
plt.title('Test error: ' + str(round(test_error,4)),fontsize = 10)
plt.xlabel('Expected Age (years)')
plt.ylabel('Predicted Age (years)')

plt.savefig(fname = "age_prediction" + str(this_tissue).replace('/','-') + ".png")

plt.close('all')
