from tqdm import tqdm
from natsort import natsorted, natsort_keygen
import matplotlib.pyplot as plt
import albumentations as A
import os
import cv2
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
from sklearn.model_selection import RepeatedStratifiedKFold as rskf
import random
# from minepy.mine import MINE

def diff_func(X_norm,y_norm, limit_data = False, age_normalizer = 1):
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
img_size = 100
print('loading in data')
# data_path = '/groups/sutphin/NN_trainings/IGTD/Results/Liver;liver hepatocytes_1_9620/data'
data_path = '/groups/sutphin/NN_trainings/IGTD/Results/All_tissues_1_9620/data'
#data_path = r"C:\Users\Lab PC\Documents\GitHub\IGTD\Results\All_tissues_1_9620\data"
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

        temp_img = cv2.resize(temp_img,(img_size,img_size))

        X.append((temp_img - np.min(temp_img))/(np.max(temp_img) - np.min(temp_img)))#/np.max(temp_img))
        X_meta.append([str(this_metadata['Gender'].values.squeeze()),str(this_metadata['Tissue'].values.squeeze())])



del count,this_img,temp_img
X = np.asarray(X)
y = np.asarray(y)
X_meta = np.asarray(X_meta)

X_norm = X 
y_norm = y

# remove 5 random samples for testing later 
test_idx, norm_idx = get_n_samples(15,y_norm,this_seed = 50) # was 5

# split the test set off from the rest (to be k folded)
X_norm_test, y_norm_test = X_norm[test_idx], y_norm[test_idx]
X_norm, y_norm = X_norm[norm_idx], y_norm[norm_idx]

# run the diff function only for the test set
X_test,y_test = X_norm_test, y_norm_test # diff_func(X_norm_test, y_norm_test, age_normalizer = age_normalizer)

y_norm_init = y_norm.copy()
X_norm_init = X_norm.copy()

temp = np.zeros(shape=(1,))

to_save = os.path.split(__file__)[0]
save_dir = 'data_arrays'
save_path = os.path.join(to_save,save_dir)

os.makedirs(save_path, exist_ok = True)
train_save_path = os.path.join(save_path,'All_data')
np.savez(train_save_path,X = X_norm,y = y_norm)

test_save_path = os.path.join(save_path,'test')
np.savez(test_save_path,X = X_test,y = y_test)

n_kfolds = 10
skf = rskf(n_splits = 10, n_repeats = 1, random_state=50)
k = np.asarray(list(range(len(y_norm_init))))

count = 0
for train_idx,val_idx in skf.split(k,y_norm_init):

    print(val_idx)

    if count == 0:
        temp = val_idx
    else:
        temp = np.append(temp,val_idx)

    train_save_path = os.path.join(save_path,'val' + str(count))
    np.savez(train_save_path,idx = val_idx)

    train_save_path = os.path.join(save_path,'train' + str(count))
    np.savez(train_save_path,idx = train_idx)
    count += 1

print('validation uses',len(np.unique(np.asarray(temp))),'of',len(np.unique(np.asarray(k))), 'in dataset')
print('eof')
