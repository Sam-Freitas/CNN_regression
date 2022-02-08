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
num = 900

print('loading in data')
data = pd.read_csv('raw_filtered_rotated.csv',header=0, index_col=0)
metadata = pd.read_csv('meta_filtered.csv',header=0, index_col=0)

print('parsing data')
single_tissue_index = metadata['Healthy'].values == True
data = data.iloc[single_tissue_index,:]
data_std = data.std()
sorted_std_idx_ascend = np.argsort(data_std.values)
data = data.iloc[:, sorted_std_idx_ascend[-num:]]

metadata_healthy = metadata.iloc[single_tissue_index,:]

SRR_values = metadata_healthy['SRR.ID'].values
unique_tissues = np.unique(metadata_healthy['Tissue'].values)

this_tissue = 'Adipose;'
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
y_norm = (y - np.min(y))/np.max(y-np.min(y))

del data, data_std, metadata, metadata_healthy, single_tissue_index, sorted_std_idx_ascend, SRR_values, this_metadata, this_imgs_meta_idx, srr_id, this_data

print('Setting up model')
model = fully_connected_dense_model(num_features = num, use_dropout=False)

optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.0001)#, momentum=0.9)
model.compile(optimizer=optimizer,loss='MAE',metrics=['MSE'])
save_checkpoints = tf.keras.callbacks.ModelCheckpoint(
    filepath = 'model_weights/cp.ckpt', monitor = 'val_loss',
    mode = 'min',save_best_only = True,save_weights_only = True, verbose = 1)
redule_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_loss', factor = 0.1, patience = 250, min_lr = 0.0000001, verbose = 1)
earlystop = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',min_delta = 0.01,patience = 20000, verbose = 1)

model.summary()

model.load_weights('model_weights/cp.ckpt')

eval_result = model.evaluate(X_norm,y_norm,batch_size=1,verbose=1,return_dict=True)

plt.figure(1)

predicted = model.predict(X_norm,batch_size=1).squeeze()

plt.scatter(y_norm,predicted,color = 'r',alpha=0.5)

plt.plot(np.linspace(0, np.max(y_norm)),np.linspace(0, np.max(y_norm)))
plt.xlabel('Expected Age (years)')
plt.ylabel('Predicted Age (years)')

plt.savefig(fname = "output_" + str(this_tissue).replace('/','-') + ".png")

plt.close('all')

print('eof')