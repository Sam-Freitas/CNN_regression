import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from skimage import measure
from tqdm import tqdm
from natsort import natsorted, natsort_keygen
import matplotlib.pyplot as plt
import albumentations as A
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from dense_model import fully_connected_dense_model

# set up variables 
num = 784

print('loading in data')
data = pd.read_csv('dense_regression/raw_filtered_rotated.csv',header=0, index_col=0)
metadata = pd.read_csv('dense_regression/meta_filtered.csv',header=0, index_col=0)

print('parsing data')
single_tissue_index = metadata['Healthy'].values == True
data = data.iloc[single_tissue_index,:]
data_std = data.std()
sorted_std_idx_ascend = np.argsort(data_std.values)
data = data.iloc[:, sorted_std_idx_ascend[-num:]]

metadata_healthy = metadata.iloc[single_tissue_index,:]

SRR_values = metadata_healthy['SRR.ID'].values
unique_tissues = np.unique(metadata_healthy['Tissue'].values)

this_tissue = 'Blood;PBMC'
# this_tissue = 'Adipose;'
print('Current tissue',this_tissue)

X = []
y = []
for count in tqdm(range(data.shape[0])):

    this_data = data.iloc[count] # get single data
    srr_id = this_data.name # get name
    this_imgs_meta_idx = (SRR_values == srr_id) # find metadata
    this_metadata = metadata_healthy.iloc[this_imgs_meta_idx,:] # get metadata
    if (this_metadata['Tissue'].values == this_tissue).squeeze(): # if specific argument 
        y.append(metadata_healthy.iloc[this_imgs_meta_idx,:]['Age'].values.squeeze()) 
        X.append(this_data.values)

X_raw = np.asarray(X) # convert to array
y_raw = np.asarray(y)

MM = MinMaxScaler()
X_norm = MM.fit_transform(X_raw)
y_norm = y_raw

del data, data_std, metadata, metadata_healthy, single_tissue_index, sorted_std_idx_ascend, SRR_values, this_metadata, this_imgs_meta_idx, srr_id, this_data

print('Setting up model')
model = fully_connected_dense_model(num_features = num, use_dropout=False)

optimizer = tf.keras.optimizers.RMSprop()#, momentum=0.9)
model.compile(optimizer=optimizer,loss='MAE',metrics=['MSE'])

model.summary()

model.load_weights('dense_regression/model_weights_test/cp.ckpt')

eval_result = model.evaluate(X_norm,y_norm,batch_size=1,verbose=1,return_dict=True)

plt.figure(1)

predicted = model.predict(X_norm,batch_size=1).squeeze()

cor_matrix = np.corrcoef(predicted.squeeze(),y_norm)
cor_xy = cor_matrix[0,1]
r_squared = round(cor_xy**2,4)
print(r_squared)

model.save('dense_regression/compiled_models/' + str(r_squared)[2:])

plt.scatter(y_norm,predicted,color = 'r',alpha=0.25)
plt.plot(np.linspace(np.min(y_norm), np.max(y_norm)),np.linspace(np.min(y_norm), np.max(y_norm)))
plt.text(0,1,"r^2: " + str(r_squared),fontsize = 12)
plt.xlabel('Expected Age (years)')
plt.ylabel('Predicted Age (years)')

plt.savefig(fname = "dense_regression/output_" + str(this_tissue).replace('/','-') + ".png")

plt.close('all')

print('eof')