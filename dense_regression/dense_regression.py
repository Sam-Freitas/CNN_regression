from tqdm import tqdm
from natsort import natsorted, natsort_keygen
import matplotlib.pyplot as plt
import albumentations as A
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from dense_model import fully_connected_dense_model
import json
from scipy import stats

# set up variables 
num = 1024

def idx_by_spearman_coef(data,metadata): # return the sorted calues by the smallest p values accorind to the spearman coefficient

    ages = np.asarray(metadata['Age'].values)
    output = dict()
    inital_gene_order = list(data.columns)
    for count,this_gene in enumerate(inital_gene_order):
        these_points = data[this_gene].values
        sprmn_coef = stats.spearmanr(ages,these_points)
        output[this_gene] = [sprmn_coef.correlation,sprmn_coef.pvalue]
    df = pd.DataFrame.from_dict(output,orient = 'index', columns = ['Spearman_coef','p-value'])
    df = df.sort_values(['p-value'], ascending = True)
    sorted_gene_order = list(df.index)
    idx = np.zeros(shape = (1,len(sorted_gene_order))).squeeze()
    for count,this_gene in enumerate(sorted_gene_order):
        idx[count] = inital_gene_order.index(this_gene)
    idx = idx.astype(np.int64)

    return idx,df

print('loading in data')
data = pd.read_csv('dense_regression/raw_filtered_rotated.csv',header=0, index_col=0)
metadata = pd.read_csv('dense_regression/meta_filtered.csv',header=0, index_col=0)

print('parsing data')
this_tissue = 'Blood;PBMC'
healthy_index = metadata['Healthy'].values == True
tissue_index = metadata['Tissue'].values == this_tissue

single_tissue_index = healthy_index*tissue_index
data = data.iloc[single_tissue_index,:]
metadata_healthy = metadata.iloc[single_tissue_index,:]

sprt_idx,sort_help = idx_by_spearman_coef(data,metadata_healthy)
data = data.iloc[:, sprt_idx[:num]]
# data_std = data.std()
# sorted_std_idx_ascend = np.argsort(data_std.values)
# data = data.iloc[:, sorted_std_idx_ascend[-num:]]


SRR_values = metadata_healthy['SRR.ID'].values
unique_tissues = np.unique(metadata_healthy['Tissue'].values)

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

val_idx = []
for unique_num in np.unique(y_raw): #[0::2]:
    indices = np.where(y_raw==unique_num)
    if indices[0].shape[0] > 3:
        val_idx.extend(np.where(y_raw==unique_num)[0][0:3])

train_idx = np.arange(y_norm.shape[0])
train_idx = np.delete(train_idx,val_idx)

X_train = X_norm[train_idx]
y_train = y_norm[train_idx]
X_val = X_norm[val_idx]
y_val = y_norm[val_idx]

# del data, metadata, metadata_healthy, single_tissue_index, sorted_std_idx_ascend, SRR_values, this_metadata, this_imgs_meta_idx, srr_id, this_data

print('Setting up model')
model = fully_connected_dense_model(num_features = num, use_dropout=True)

epochs = 10000

optimizer = tf.keras.optimizers.RMSprop(momentum = 0.75)#,learning_rate = 0.00001)#, momentum=0.9)
model.compile(optimizer=optimizer,loss='MAE',metrics=['MSE','accuracy'])
save_checkpoints = tf.keras.callbacks.ModelCheckpoint(
    filepath = 'dense_regression/model_weights_test/cp.ckpt', monitor = 'val_loss',
    mode = 'min',save_best_only = True,save_weights_only = True, verbose = 1)
redule_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_loss', factor = 0.1, patience = 25, min_lr = 0.0000001, verbose = 1)
earlystop = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',min_delta = 0.01,patience = 500, verbose = 1)

def scheduler(epoch, lr):
    if epoch < 100:
        lr = 0.00001
    elif epoch > 99 and epoch < 250:
        lr = 0.000005
    else:
        lr = 0.000001
    return lr

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

model.summary()

history = model.fit(X_train,y_train,
    validation_data = (X_val,y_val),
    batch_size=4,epochs=epochs,
    callbacks=[save_checkpoints,earlystop,lr_scheduler],
    verbose=1)

eval_result = model.evaluate(X_norm,y_norm,batch_size=1,verbose=1,return_dict=True)

plt.figure(1)

predicted = model.predict(X_norm,batch_size=1).squeeze()

plt.scatter(y_norm,predicted,color = 'r',alpha=0.5)

plt.plot(np.linspace(0, np.max(y_norm)),np.linspace(0, np.max(y_norm)))
plt.xlabel('Expected Age (years)')
plt.ylabel('Predicted Age (years)')

plt.savefig(fname = "dense_regression/model_predictions" + str(this_tissue).replace('/','-') + ".png")

plt.close('all')

plt.figure(2)
for this_key in list(history.history.keys()):
    b = history.history[this_key]
    plt.plot(b,label = this_key)

res = dict()
for key in eval_result: res[key] = round(eval_result[key],6)

plt.legend(loc="upper left")
plt.ylim([0,15])
plt.title(json.dumps(res))
plt.savefig(fname=  "dense_regression/training_history" + str(this_tissue).replace('/','-') + ".png")

plt.close('all')

print('eof')