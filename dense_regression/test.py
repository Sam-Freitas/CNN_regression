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
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy import stats
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, BatchNormalization, Activation, Dense, LSTMCell
from tensorflow.keras import backend as K
from tensorflow.python.keras.engine import training

def cantor_pair(a,b):

    c = (1/2)*(a+b)*(a + b + 1)

    return c

def get_model(num_categories = 5,use_dropout = False):

    inputs = Input(shape = (2,))

    sm = Dense(2048)(inputs)
    dm = Activation('gelu')(sm)
    dm = Dropout(0.8)(dm,training = use_dropout)
    dm = Dense(64)(dm)
    dm = Activation('gelu')(sm)
    dm = Dropout(0.8)(dm,training = use_dropout)

    output = Dense(num_categories,activation='linear')(dm)

    model = Model(inputs=[inputs], outputs=[output])

    return model

def load_data():

    print('loading in data')
    # data = pd.read_csv('dense_regression/raw_filtered_rotated.csv',header=0, index_col=0)
    metadata = pd.read_csv('dense_regression/meta_filtered.csv',header=0, index_col=0)  

    this_tissue = 'All_tissues'
    healthy_index = metadata['Healthy'].values == True

    single_tissue_index = healthy_index#*tissue_index
    # data = data.iloc[single_tissue_index,:]
    metadata_healthy = metadata.iloc[single_tissue_index,:]

    X_raw = []
    y = []
    for count in range(metadata_healthy.shape[0]):
        this_metadata = metadata_healthy.iloc[count,:]
        X_raw.append([str(this_metadata['Gender']),str(this_metadata['Tissue'])])
    X_raw = np.asarray(X_raw)

    le = LabelEncoder()
    X_encoded = np.zeros(shape=X_raw.shape)
    for count,this_feature in enumerate(X_raw.transpose()):
        X_encoded[:,count] = le.fit_transform(this_feature)

    for count,this_data in enumerate(X_encoded):
        y.append(cantor_pair(this_data[0],this_data[1]))

    y_encoded = le.fit_transform(y)

    y_categorical = tf.keras.utils.to_categorical(y_encoded)

    return X_encoded,y_categorical,y_encoded

def train_val_split(X,y,y_encoded):

    val_idx = []
    not_enough = []
    for unique_y in np.unique(y_encoded):
        indicies = np.where(y_encoded == unique_y)
        if indicies[0].shape[0] > 10:
            val_idx.extend(indicies[0][0:5])
        else:
            not_enough.append(unique_y)

    train_idx = np.delete(np.arange(y_encoded.shape[0]),val_idx)
    val_idx = np.sort(val_idx)

    X_train = X[train_idx,:]
    y_train = y[train_idx,:]
    y_train_e = y_encoded[train_idx]
    X_val = X[val_idx,:]
    y_val = y[val_idx,:]
    y_val_e = y_encoded[val_idx]

    return X_train, y_train, y_train_e, X_val, y_val, y_val_e

X,y,y_encoded = load_data()
X_train, y_train, y_train_e, X_val, y_val, y_val_e = train_val_split(X,y,y_encoded)

model = get_model(num_categories = 1, use_dropout=True)
model.summary()

epochs = 25

earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.00001, patience=500, verbose=1, restore_best_weights=True)
save_checkpoints = tf.keras.callbacks.ModelCheckpoint(
    filepath = 'dense_regression/test_weights/cp.ckpt', monitor = 'val_loss',
    mode = 'min',save_best_only = True,save_weights_only = True, verbose = 1)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.MeanAbsoluteError()
model.compile(optimizer=optimizer,loss = loss, metrics=['accuracy'])

history = model.fit(
    x = X_train,
    y = y_train_e,
    validation_data = (X_val,y_val_e),
    epochs = epochs,
    verbose = 1,
    callbacks=[earlystop,save_checkpoints]
)

del model

model = get_model(num_categories = 1, use_dropout=False)
model.load_weights('dense_regression/test_weights/cp.ckpt')

pred = model.predict(X)

plt.scatter(y_encoded,pred,color = 'r',alpha = 0.2)
plt.plot(np.linspace(np.min(y_encoded),np.max(y_encoded)),np.linspace(np.min(y_encoded),np.max(y_encoded)))
plt.show()

print('eof')