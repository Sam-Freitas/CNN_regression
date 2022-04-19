from tqdm import tqdm
from natsort import natsorted, natsort_keygen
import matplotlib.pyplot as plt
import albumentations as A
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, PolynomialFeatures, PowerTransformer, QuantileTransformer, StandardScaler
import json
import glob
from scipy import stats
from scipy.spatial.distance import correlation as dist_corr_eucl
from numpy.polynomial import Polynomial
from sklearn.model_selection import train_test_split
import random
# from minepy.mine import MINE

def diff_func(X_norm,y_norm, limit_data = False):
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

    return X_diff, y_diff

def get_n_samples(n,this_array,this_seed = 50):

    norm_idx = np.arange(this_array.shape[0])
    np.random.seed(this_seed)
    new_idx = np.unique(np.random.randint(low=0,high=norm_idx.shape[0],size=(n,1)))
    temp = norm_idx[new_idx]
    norm_idx = np.delete(norm_idx,new_idx)
    new_idx = temp

    return new_idx, norm_idx


age_normalizer = 1
print('loading in data')
data_path = '/groups/sutphin/NN_trainings/IGTD/Results/Liver;liver hepatocytes_1_9620/data'
data_path = r"C:\Users\Lab PC\Documents\GitHub\IGTD\Results\All_tissues_1_9620\data"
metadata_path = 'dense_regression/meta_filtered.csv'
imgs_list = natsorted(glob.glob(os.path.join(data_path,'*.txt')))
metadata = pd.read_csv(metadata_path)

metadata = metadata.sort_values(by = ['SRR.ID'], ascending = True)
imgs_list = np.sort(imgs_list)

healthy_idx = metadata['Healthy'].values
metadata_healthy = metadata.iloc[healthy_idx,:]

SRR_values = metadata_healthy['SRR.ID'].values
unique_tissues = np.unique(metadata_healthy['Tissue'].values)

# this_tissue = unique_tissues[0]
# this_tissue = 'Liver;liver hepatocytes'
this_tissue = 'Blood;PBMC'
print('Current tissue',this_tissue)

X = []
X_meta = []
y = []
print('reading in data')
for count in tqdm(range(len(imgs_list))):

    this_img = imgs_list[count]

    srr_id = os.path.basename(this_img)[1:-9]
    this_imgs_meta_idx = (SRR_values == srr_id)
    this_metadata = metadata_healthy.iloc[this_imgs_meta_idx,:]
    if (this_metadata['Tissue'].values == this_tissue).squeeze():
        y.append(metadata_healthy.iloc[this_imgs_meta_idx,:]['Age'].values.squeeze())
        temp_img = np.loadtxt(this_img, comments='#',delimiter="\t",unpack=False)
        X.append((temp_img - np.min(temp_img))/(np.max(temp_img) - np.min(temp_img)))#/np.max(temp_img))
        X_meta.append([str(this_metadata['Gender'].values.squeeze()),str(this_metadata['Tissue'].values.squeeze())])

del count,this_img,temp_img
X = np.asarray(X)
y = np.asarray(y)
X_meta = np.asarray(X_meta)

X_norm = X 
y_norm = y

# X_meta_cleaned = X_meta.copy()
# for i in range(X_meta_cleaned.shape[0]):
#     temp_meta = X_meta[i][1].lower()
#     X_meta_cleaned[i][1] = temp_meta.replace('/','-')

# le = LabelEncoder()
# X_meta_norm = np.zeros(shape=X_meta_cleaned.shape)
# for count,this_feature in enumerate(X_meta_cleaned.transpose()):
#     X_meta_norm[:,count] = le.fit_transform(this_feature)

# remove 5 random samples for testing later 
test_idx, norm_idx = get_n_samples(5,y_norm,this_seed = 50)

# split the test set off from the rest (to be k folded)
X_norm_test, y_norm_test = X_norm[test_idx], y_norm[test_idx]
X_norm, y_norm = X_norm[norm_idx], y_norm[norm_idx]

# run the diff function only for the test set
X_test,y_test = diff_func(X_norm_test, y_norm_test)

y_norm_init = y_norm.copy()
X_norm_init = X_norm.copy()

temp = np.zeros(shape=(1,))

for i in range(3):
    val_idx, norm_idx = get_n_samples(50,y_norm_init,this_seed = i)

    temp = np.concatenate([temp,val_idx],axis = 0)

    print(val_idx)

    # generate the test train/val split
    X_norm_val, y_norm_val = X_norm_init[val_idx], y_norm_init[val_idx]
    X_norm, y_norm = X_norm_init[norm_idx], y_norm_init[norm_idx]

    # generate the diff data with seperate blinded set
    X_train,y_train = diff_func(X_norm,y_norm)
    X_val,y_val = diff_func(X_norm_val,y_norm_val)

    print('Saving data arrays')
    # set up saving 
    to_save = os.path.split(__file__)[0]
    save_dir = 'data_arrays'
    save_path = os.path.join(to_save,save_dir)

    os.makedirs(save_path, exist_ok = True)

    train_save_path = os.path.join(save_path,'train' + str(i))
    np.savez(train_save_path,X = X_train,y = y_train)

    val_save_path = os.path.join(save_path,'val' + str(i))
    np.savez(val_save_path,X = X_val,y = y_val)

test_save_path = os.path.join(save_path,'test')
np.savez(test_save_path,X = X_test,y = y_test)

print('eof')

# y_diff = []
# X_diff = []
# count = 0
# for i in range(len(y)):
#     X1 = X_norm[i]
#     y1 = y_norm[i]
#     for j in range(len(y)):
#         X2 = X_norm[j]
#         y2 = y_norm[j]
#         X_diff.append(np.concatenate([np.atleast_3d(X1),np.atleast_3d(X2)],axis = -1))
#         y_temp = (y1-y2)/age_normalizer
#         y_temp = np.round(y_temp,3)
#         y_diff.append(y_temp)
#         print(count)
#         count = count + 1
# X_diff = np.asarray(X_diff)
# y_diff = np.asarray(y_diff)

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

# np.random.seed(50)
# test_idx = np.unique(np.random.randint(low=0,high=train_idx.shape[0],size=(200,1)))
# temp = train_idx[test_idx]
# train_idx = np.delete(train_idx,test_idx)
# test_idx = temp

# # training data
# X_train = X_diff[train_idx]
# y_train = y_diff[train_idx]
# # validation data
# X_val = X_diff[val_idx]
# y_val = y_diff[val_idx]
# # test data
# X_test = X_diff[test_idx]
# y_test = y_diff[test_idx]

# # # add adversarial data
# # np.random.seed(50)
# # X_rand1 = np.random.rand(X_train.shape[1],X_train.shape[2])*2
# # X_rand1 = X_train + X_rand1
# # np.random.seed(100)
# # X_rand2 = np.random.rand(X_train.shape[1],X_train.shape[2])
# # X_rand2 = X_train + X_rand2
# # np.random.seed(150)
# # X_rand3 = np.random.rand(X_train.shape[0],X_train.shape[1],X_train.shape[2])*2
# # X_rand3 = X_train + X_rand3


# # X_train = np.concatenate([X_train,X_rand1,X_rand2,X_rand3],axis = 0)
# # X_meta_train = np.concatenate([X_meta_train,X_meta_train,X_meta_train,X_meta_train],axis = 0)
# # y_train = np.concatenate([y_train,y_train,y_train,y_train],axis = 0)

# # set up saving 
# to_save = os.path.split(__file__)[0]
# save_dir = 'data_arrays'
# save_path = os.path.join(to_save,save_dir)

# os.makedirs(save_path, exist_ok = True)

# train_save_path = os.path.join(save_path,'train')
# np.savez(train_save_path,X = X_train,y = y_train)

# val_save_path = os.path.join(save_path,'val')
# np.savez(val_save_path,X = X_val,y = y_val)

# test_save_path = os.path.join(save_path,'test')
# np.savez(test_save_path,X = X_test,y = y_test)

# print('eof')