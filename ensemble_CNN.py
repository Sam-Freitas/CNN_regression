import tensorflow as tf
import numpy as np
import pandas as pd
from CNN_regression_model import fully_connected_CNN,ResNet50v2_regression, plot_model,load_rotated_minst_dataset

print('Loading in metadata')
imgs_location = '/home/u23/samfreitas/NN_trainings/IGTD/Results/Test_1_low_error/data'
metadata = pd.read_csv('/home/u23/samfreitas/NN_trainings/IGTD/Data/experimentDesign.csv')

sorted_metadata = metadata.copy(deep=True)
sorted_metadata = sorted_metadata.sort_values(by = 'sample_id',key = natsort_keygen())

unique_tissues = np.unique(sorted_metadata['tissue'].values)

this_tissue = 'Kidney'

model_1 = fully_connected_CNN(height=100,width=100,use_dropout=False)
model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01),loss=tf.keras.losses.MeanSquaredError(),metrics = ['MeanAbsoluteError','accuracy'])
model_2 = fully_connected_CNN(height=100,width=100,use_dropout=False)
model_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01),loss=tf.keras.losses.MeanSquaredError(),metrics = ['MeanAbsoluteError','accuracy'])
model_3 = fully_connected_CNN(height=100,width=100,use_dropout=False)
model_3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01),loss=tf.keras.losses.MeanSquaredError(),metrics = ['MeanAbsoluteError','accuracy'])

# attempt with using single tissue data
single_tissue_index = (sorted_metadata['tissue'].values == this_tissue)
metadata_single_tissue = sorted_metadata.iloc[single_tissue_index]
extracted_metadata = metadata_single_tissue['age'].values
y = extracted_metadata

imgs_list = natsorted(glob.glob(os.path.join(imgs_location,'*.txt')))

print('Loading in data from txt files')
X = []
for count in tqdm(range(len(imgs_list))):

    this_img = imgs_list[count]

    img_sample_id = os.path.basename(os.path.splitext(this_img)[0])[1:-5]
    if img_sample_id in metadata_single_tissue['sample_id'].values:
        temp_img = np.loadtxt(this_img, comments="#", delimiter="\t", unpack=False)
        X.append(temp_img)

X = np.asarray(X)

