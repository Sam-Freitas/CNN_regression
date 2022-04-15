from tqdm import tqdm
from natsort import natsorted, natsort_keygen
import matplotlib.pyplot as plt
import albumentations as A
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, PolynomialFeatures, PowerTransformer, QuantileTransformer, StandardScaler
from dense_model import fully_connected_dense_model, plot_model,test_on_improved_val_loss
import json
from scipy import stats
from scipy.spatial.distance import correlation as dist_corr_eucl
from numpy.polynomial import Polynomial
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import random
# from minepy.mine import MINE

num = 9618

def idx_by_spearman_coef(data,metadata): # return the sorted calues by the smallest p values accorind to the spearman coefficient

    ages = np.asarray(metadata['Age'].values)
    output = dict()
    inital_gene_order = list(data.columns)
    
    for count in tqdm(range(len(inital_gene_order))):
        this_gene = inital_gene_order[count]
        these_points = data[this_gene].values
        sprmn_coef = stats.spearmanr(ages,these_points)
        dist_coef = dist_corr_eucl(ages,these_points)
        # kendalltau = stats.kendalltau(ages,these_points)
        # m = MINE()
        # m.compute_scores(ages,these_points)
        # mic_score = m.mic()

        output[this_gene] = [sprmn_coef.correlation,sprmn_coef.pvalue,dist_coef,count]
    df = pd.DataFrame.from_dict(output,orient = 'index', columns = ['Spearman_coef','Sp_value','dist_coef','row'])
    df = df.sort_values(['Sp_value'], ascending = True)
    # df = df.sort_values(['mic_score'], ascending = False)

    df2 = pd.read_csv('dense_regression/sorting_csv.csv',header=0, index_col=0)
    in_L1000 = list(df2.index)

    sorted_gene_order = list(df.index)
    idx = np.zeros(shape = (1,len(sorted_gene_order))).squeeze()
    idx_counter = 0
    for count,this_gene in enumerate(sorted_gene_order):
        if this_gene in in_L1000:
            idx[idx_counter] = inital_gene_order.index(this_gene)
            idx_counter = idx_counter +1
        else:
            pass
    idx = idx.astype(np.int64)

    df.to_csv('data_preprocessing.csv')

    return idx,df

def add_2_features(data):

    new = []

    for i in range(data.shape[0]):
        new.append([np.std(data.iloc[i,:]),np.mean(data.iloc[i,:])])

    new = np.asarray(new)

    data['std'] = new[:,0]
    data['mean'] = new[:,1]

    return data

def diff_func(X_norm,y_norm, limit_data = False):

    if not limit_data:
        print('Diff function generation')
        y_diff = []
        X_diff = []
        num_loops = y_norm.shape[0]
        count = 0
        for i in tqdm(range(num_loops)):
            X1 = X_norm[i]
            y1 = y_norm[i]
            for j in range(num_loops):
                X2 = X_norm[j]
                y2 = y_norm[j]
                X_diff.append(np.concatenate([np.atleast_3d(X1),np.atleast_3d(X2)],axis = -1).squeeze())
                y_temp = (y1-y2)/age_normalizer
                y_temp = np.round(y_temp,3)
                y_diff.append(y_temp)
                count = count + 1
        X_diff = np.asarray(X_diff)
        y_diff = np.asarray(y_diff)
    else:
        print('Diff function generation')
        y_diff = []
        X_diff = []
        num_loops = y_norm.shape[0]
        count = 0
        for i in tqdm(range(num_loops)):
            X1 = X_norm[i]
            y1 = y_norm[i]
            for j in np.unique(np.random.randint(low=0,high=y_norm.shape[0],size=(int(round(y_norm.shape[0]/2)),1))):
                X2 = X_norm[j]
                y2 = y_norm[j]
                X_diff.append(np.concatenate([np.atleast_3d(X1),np.atleast_3d(X2)],axis = -1).squeeze())
                y_temp = (y1-y2)/age_normalizer
                y_temp = np.round(y_temp,3)
                y_diff.append(y_temp)
                count = count + 1
        X_diff = np.asarray(X_diff)
        y_diff = np.asarray(y_diff)


    return X_diff, y_diff

print('loading in data')
data = pd.read_csv('dense_regression/normalized_training_data_rot.csv',header=0, index_col=0)
metadata = pd.read_csv('dense_regression/meta_filtered.csv',header=0, index_col=0)

# sort data 
metadata = metadata.sort_values(by = ['SRR.ID'], ascending = True)
data = data.sort_index(axis = 0, ascending = True)

print('parsing data')
age_normalizer = 128
# this_tissue = 'Liver;liver hepatocytes'
# this_tissue = 'All_tissues'
this_tissue = 'Blood;PBMC'
healthy_index = metadata['Healthy'].values == True
tissue_index = healthy_index.copy()
for count,temp in enumerate(metadata['Tissue'].values):
    if temp.__contains__('Retina'):
        tissue_index[count] = False
    else:
        tissue_index[count] = True
tissue_index = metadata['Tissue'].values == this_tissue
age_index = metadata['Age'].values > 14

single_tissue_index = healthy_index*tissue_index#*age_index
data = data.iloc[single_tissue_index,:]
metadata_healthy = metadata.iloc[single_tissue_index,:]

sprt_idx,sort_help = idx_by_spearman_coef(data,metadata_healthy)
data = data.iloc[:, sprt_idx[:num]]
data = add_2_features(data)

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
    y.append(this_metadata['Age'].values[0])
    X_meta.append([str(this_metadata['Gender'].values.squeeze()),str(this_metadata['Tissue'].values.squeeze())])
    X.append(this_data.values)

X_raw = np.asarray(X) # convert to array
y_raw = np.asarray(y)
X_meta_raw = np.asarray(X_meta)

X_norm = X_raw
y_norm = y_raw

# X_meta_cleaned = X_meta_raw.copy()
# for i in range(X_meta_cleaned.shape[0]):
#     temp_meta = X_meta_raw[i][1].lower()
#     X_meta_cleaned[i][1] = temp_meta.replace('/','-')

# le = LabelEncoder()
# X_meta_norm = np.zeros(shape=X_meta_cleaned.shape)
# for count,this_feature in enumerate(X_meta_cleaned.transpose()):
#     X_meta_norm[:,count] = le.fit_transform(this_feature)

# generate 10 random idx for the double blind testing 
norm_idx = np.arange(y_norm.shape[0])
np.random.seed(50)
test_idx = np.unique(np.random.randint(low=0,high=norm_idx.shape[0],size=(5,1)))
temp = norm_idx[test_idx]
norm_idx = np.delete(norm_idx,test_idx)
test_idx = temp
del temp

# generate the test train/val split
X_norm_test, y_norm_test = X_norm[test_idx], y_norm[test_idx]
X_norm, y_norm = X_norm[norm_idx], y_norm[norm_idx]

norm_idx = np.arange(y_norm.shape[0])
np.random.seed(50)
val_idx = np.unique(np.random.randint(low=0,high=norm_idx.shape[0],size=(50,1)))
temp = norm_idx[val_idx]
norm_idx = np.delete(norm_idx,val_idx)
val_idx = temp
del temp

# generate the test train/val split
X_norm_val, y_norm_val = X_norm[val_idx], y_norm[val_idx]
X_norm, y_norm = X_norm[norm_idx], y_norm[norm_idx]

# generate the diff data with seperate blinded set
X_diff,y_diff = diff_func(X_norm,y_norm,limit_data=False)
X_diff_val,y_diff_val = diff_func(X_norm_val,y_norm_val)
X_diff_test,y_diff_test = diff_func(X_norm_test, y_norm_test)


# training data
X_train = X_diff
y_train = y_diff
# validation data
X_val = X_diff_val
y_val = y_diff_val
# test data
X_test = X_diff_test
y_test = y_diff_test

# val_idx = []
# not_enough_data_idx = []
# for unique_num in np.unique(y_diff): #[0::2]:
#     indices = np.where(y_diff==unique_num)
#     if indices[0].shape[0] > 10:
#         val_idx.extend(np.where(y_diff==unique_num)[0][0:5])
#     elif indices[0].shape[0] > 1:
#         num_to_exd = round(indices[0].shape[0]/2)
#         val_idx.extend(np.where(y_diff==unique_num)[0][0:num_to_exd])
#     else:
#         not_enough_data_idx.append(unique_num)
# val_idx = np.asarray(val_idx)
# not_enough_data_values = np.asarray(not_enough_data_idx)

# train_idx = np.arange(y_diff.shape[0])
# train_idx = np.delete(train_idx,val_idx)

# # training data
# X_train = X_diff[train_idx]
# y_train = y_diff[train_idx]
# # validation data
# X_val = X_diff[val_idx]
# y_val = y_diff[val_idx]
# # test data
# X_test = X_diff_test
# y_test = y_diff_test

print('Saving data arrays')
# set up saving 
to_save = os.path.split(__file__)[0]
save_dir = 'data_arrays'
save_path = os.path.join(to_save,save_dir)

os.makedirs(save_path, exist_ok = True)

train_save_path = os.path.join(save_path,'train')
np.savez(train_save_path,X = X_train,y = y_train)

val_save_path = os.path.join(save_path,'val')
np.savez(val_save_path,X = X_val,y = y_val)

test_save_path = os.path.join(save_path,'test')
np.savez(test_save_path,X = X_test,y = y_test)

print('eof')