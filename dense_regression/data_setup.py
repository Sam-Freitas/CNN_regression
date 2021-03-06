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
tissue_index = healthy_index.copy()
for count,temp in enumerate(metadata['Tissue'].values):
    if temp.__contains__('Retina'):
        tissue_index[count] = False
    else:
        tissue_index[count] = True
# tissue_index = metadata['Tissue'].values == this_tissue
age_index = metadata['Age'].values > 14

single_tissue_index = healthy_index*age_index*tissue_index
data = data.iloc[single_tissue_index,:]
metadata_healthy = metadata.iloc[single_tissue_index,:]

sprt_idx,sort_help = idx_by_spearman_coef(data,metadata_healthy)
data = data.iloc[:, sprt_idx[:num]]
data = add_2_features(data)
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
# MM = StandardScaler()
# X_norm = MM.fit_transform(X_raw)
X_norm = X_raw
# X_norm = X_raw
y_norm = y_raw

X_meta_cleaned = X_meta_raw.copy()
for i in range(X_meta_cleaned.shape[0]):
    temp_meta = X_meta_raw[i][1].lower()
    X_meta_cleaned[i][1] = temp_meta.replace('/','-')

le = LabelEncoder()
X_meta_norm = np.zeros(shape=X_meta_cleaned.shape)
for count,this_feature in enumerate(X_meta_cleaned.transpose()):
    X_meta_norm[:,count] = le.fit_transform(this_feature)

# X_norm,X_test,y_norm,y_test = train_test_split(X_norm,y_norm,test_size = 0.1,random_state = 50)

# PCA GOES HERE
labels = X_meta_norm[:,1]
label_names = X_meta_cleaned[:,1]
pca = PCA(n_components = 13)
X_train_pca = pca.fit_transform(X_norm)

pca = PCA(n_components=None)
X_train_pca_var = pca.fit_transform(X_norm)
variance_ratio = pca.explained_variance_ratio_
variance_ratio_cumsum = np.cumsum(variance_ratio)

k = KMeans(n_clusters = 10).fit(X_train_pca)
k_labels = k.labels_

label_names = X_meta_cleaned[:,1]
groupings = []
for i in range(len(np.unique(k_labels))):
    print('Group: ' + str(i), ',' , 'n=' + str(np.sum(k_labels==i)))
    print(np.unique(label_names[k_labels==i]))
    groupings.append(np.unique(label_names[k_labels==i]))

plt.figure()
for count,g in enumerate(np.unique(k_labels)):
    idx = (k_labels == g)
    plt.scatter(X_train_pca[idx,1],X_train_pca[idx,2], cmap = 'hsv', label = groupings[count])
plt.legend()

plt.figure()
for count,g in enumerate(np.unique(k_labels)):
    idx = (k_labels == g)
    plt.scatter(X_train_pca[idx,0],X_train_pca[idx,1], cmap = 'hsv', label = groupings[count])
plt.legend()


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

# convert to categorical 
max_bin = int(np.ceil(np.max(y_norm)/10)*10)
n_bins = int((np.ceil(np.max(y_norm)/10)*10)/5 + 1)
bins = np.linspace(0,max_bin,n_bins)
n = np.digitize(y_norm,bins)
y_cat = tf.keras.utils.to_categorical(n)

# training data
X_train = X_norm[train_idx]
X_meta_train = X_meta_norm[train_idx]
y_train = y_norm[train_idx]
y_train_cat = y_cat[train_idx]
# validation data
X_val = X_norm[val_idx]
X_meta_val = X_meta_norm[val_idx]
y_val = y_norm[val_idx]
y_val_cat = y_cat[val_idx]
# test data
X_test = X_norm[test_idx]
X_meta_test= X_meta_norm[test_idx]
y_test = y_norm[test_idx]
y_test_cat = y_cat[test_idx]


# add adversarial data
np.random.seed(50)
X_rand1 = (np.random.rand(X_train.shape[0],X_train.shape[1]) - 0.5)/2
X_rand1 = X_train + X_rand1
np.random.seed(100)
X_rand2 = np.random.rand(X_train.shape[0],X_train.shape[1])
X_rand2 = X_train + X_rand2

X_train = np.concatenate([X_train,X_rand1,X_rand2],axis = 0)
X_meta_train = np.concatenate([X_meta_train,X_meta_train,X_meta_train],axis = 0)
y_train = np.concatenate([y_train,y_train,y_train],axis = 0)

# # shuffle_train_idx = np.arange(X_train.shape[0])
# poly = PolynomialFeatures(2,interaction_only=True)
# X_train = poly.fit_transform(X_train)
# X_val = poly.fit_transform(X_val)
# X_test = poly.fit_transform(X_test)


# set up saving 
to_save = os.path.split(__file__)[0]
save_dir = 'data_arrays'
save_path = os.path.join(to_save,save_dir)

os.makedirs(save_path, exist_ok = True)

train_save_path = os.path.join(save_path,'train')
np.savez(train_save_path,X = X_train,X_meta = X_meta_train,y = y_train,y_cat = y_train_cat, bins = bins)

val_save_path = os.path.join(save_path,'val')
np.savez(val_save_path,X = X_val,X_meta = X_meta_val,y = y_val,y_cat = y_val_cat)

test_save_path = os.path.join(save_path,'test')
np.savez(test_save_path,X = X_test,X_meta = X_meta_test,y = y_test,y_cat = y_test_cat,bins = bins)

print('eof')