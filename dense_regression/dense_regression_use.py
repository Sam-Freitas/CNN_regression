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
from scipy import stats
from dense_model import fully_connected_dense_model

# set up variables 
num = 900

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

# del data, metadata, metadata_healthy, single_tissue_index, SRR_values, this_metadata, this_imgs_meta_idx, srr_id, this_data

print('Setting up model')
model = fully_connected_dense_model(num_features = num, use_dropout=False)

optimizer = tf.keras.optimizers.RMSprop()#, momentum=0.9)
model.compile(optimizer=optimizer,loss='MAE',metrics=['MSE'])

model.summary()

model.load_weights('dense_regression/model_weights/cp.ckpt')

eval_result = model.evaluate(X_norm,y_norm,batch_size=1,verbose=1,return_dict=True)

plt.figure(1)

predicted = model.predict(X_norm,batch_size=1).squeeze()

cor_matrix = np.corrcoef(predicted.squeeze(),y_norm)
cor_xy = cor_matrix[0,1]
r_squared = round(cor_xy**2,4)
print(r_squared)

model.save('dense_regression/compiled_models/' + str(r_squared)[2:] + this_tissue + '_' + str(num))

plt.scatter(y_norm,predicted,color = 'r',alpha=0.25)
plt.plot(np.linspace(np.min(y_norm), np.max(y_norm)),np.linspace(np.min(y_norm), np.max(y_norm)))
plt.text(np.min(y_norm),np.max(y_norm),"r^2: " + str(r_squared),fontsize = 12)
plt.xlabel('Expected Age (years)')
plt.ylabel('Predicted Age (years)')

plt.savefig(fname = "dense_regression/output_" + str(this_tissue).replace('/','-') + ".png")

plt.close('all')

print('eof')