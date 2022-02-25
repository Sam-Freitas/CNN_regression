from tqdm import tqdm
from natsort import natsorted, natsort_keygen
import matplotlib.pyplot as plt
import albumentations as A
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, PolynomialFeatures, PowerTransformer, QuantileTransformer
from dense_model import fully_connected_dense_model, plot_model,test_on_improved_val_loss
import json
from scipy import stats
from numpy.polynomial import Polynomial
from sklearn.model_selection import train_test_split
import random

num = 8100

def idx_by_spearman_coef(data,metadata): # return the sorted calues by the smallest p values accorind to the spearman coefficient

    ages = np.asarray(metadata['Age'].values)
    output = dict()
    inital_gene_order = list(data.columns)
    for count in tqdm(range(len(inital_gene_order))):
        this_gene = inital_gene_order[count]
        these_points = data[this_gene].values
        sprmn_coef = stats.spearmanr(ages,these_points)
        # kendalltau = stats.kendalltau(ages,these_points)

        output[this_gene] = [sprmn_coef.correlation,sprmn_coef.pvalue,count]
    df = pd.DataFrame.from_dict(output,orient = 'index', columns = ['Spearman_coef','Sp-value','row'])
    df = df.sort_values(['Sp-value'], ascending = True)
    sorted_gene_order = list(df.index)
    idx = np.zeros(shape = (1,len(sorted_gene_order))).squeeze()
    for count,this_gene in enumerate(sorted_gene_order):
        idx[count] = inital_gene_order.index(this_gene)
    idx = idx.astype(np.int64)

    return idx,df

print('loading in data')
data = pd.read_csv('dense_regression/normalized_training_data_rot.csv',header=0, index_col=0)
metadata = pd.read_csv('dense_regression/meta_filtered.csv',header=0, index_col=0)

# sort data 
metadata = metadata.sort_values(by = ['SRR.ID'], ascending = True)
data = data.sort_index(axis = 0, ascending = True)

print('parsing data')
# this_tissue = 'Blood;PBMC'
this_tissue = 'All_tissues'
healthy_index = metadata['Healthy'].values == True
tissue_index = metadata['Tissue'].values == this_tissue
age_index = metadata['Age'].values > 14

single_tissue_index = healthy_index*age_index#*tissue_index
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
X_meta = []
# fix this
for count in tqdm(range(data.shape[0])):

    this_data = data.iloc[count,:] # get single data
    srr_id = this_data.name # get name
    this_imgs_meta_idx = (SRR_values == srr_id) # find metadata
    this_metadata = metadata_healthy.iloc[this_imgs_meta_idx,:] # get metadata
    # if (this_metadata['Tissue'].values == this_tissue).squeeze(): # if specific argument 
    #     y.append(metadata_healthy.iloc[this_imgs_meta_idx,:]['Age'].values.squeeze()) 
    #     X.append(this_data.values)
    y.append(this_metadata['Age'].values[0])
    X_meta.append([str(this_metadata['Gender'].values.squeeze()),str(this_metadata['Tissue'].values.squeeze())])
    X.append(this_data.values)

X_raw = np.asarray(X) # convert to array
y_raw = np.asarray(y)
X_meta_raw = np.asarray(X_meta)

# PT = QuantileTransformer()
# X_norm = PT.fit_transform(X_raw)

MM = MinMaxScaler(feature_range = (-1,1))
X_norm = MM.fit_transform(X_raw)
y_norm = y_raw

le = LabelEncoder()
X_meta_norm = np.zeros(shape=X_meta_raw.shape)
for count,this_feature in enumerate(X_meta_raw.transpose()):
    X_meta_norm[:,count] = le.fit_transform(this_feature)

# X_norm,X_test,y_norm,y_test = train_test_split(X_norm,y_norm,test_size = 0.1,random_state = 50)

val_idx = []
not_enough_data_idx = []
for unique_num in np.unique(y_norm): #[0::2]:
    indices = np.where(y_norm==unique_num)
    if indices[0].shape[0] > 10:
        val_idx.extend(np.where(y_norm==unique_num)[0][0:5])
    elif indices[0].shape[0] > 1:
        num_to_exd = round(indices[0].shape[0]/2)
        val_idx.extend(np.where(y_norm==unique_num)[0][0:num_to_exd])
    else:
        not_enough_data_idx.append(unique_num)
val_idx = np.asarray(val_idx)
not_enough_data_values = np.asarray(not_enough_data_idx)

train_idx = np.arange(y_norm.shape[0])
train_idx = np.delete(train_idx,val_idx)

np.random.seed(50)
test_idx = np.unique(np.random.randint(low=0,high=train_idx.shape[0],size=(200,1)))
temp = train_idx[test_idx]
train_idx = np.delete(train_idx,test_idx)
test_idx = temp

# training data
X_train = X_norm[train_idx]
X_meta_train = X_meta_norm[train_idx]
y_train = y_norm[train_idx]
# validation data
X_val = X_norm[val_idx]
X_meta_val = X_meta_norm[val_idx]
y_val = y_norm[val_idx]
# test data
X_test = X_norm[test_idx]
X_meta_test= X_meta_norm[test_idx]
y_test = y_norm[test_idx]

# set up saving 
to_save = os.path.split(__file__)[0]
save_dir = 'data_arrays'
save_path = os.path.join(to_save,save_dir)

os.makedirs(save_path, exist_ok = True)

train_save_path = os.path.join(save_path,'train')
np.savez(train_save_path,X = X_train,X_meta = X_meta_train,y = y_train)

val_save_path = os.path.join(save_path,'val')
np.savez(val_save_path,X = X_val,X_meta = X_meta_val,y = y_val)

test_save_path = os.path.join(save_path,'test')
np.savez(test_save_path,X = X_test,X_meta = X_meta_test,y = y_test)

print('eof')