import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob
import json
import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from natsort import natsorted, natsort_keygen
from CNN_regression_model import fully_connected_CNN,ResNet50v2_regression, plot_model,load_rotated_minst_dataset

# (X,y), (X_val,y_val), (test_X,test_y) = load_rotated_minst_dataset(seed = 50)

print('reading in metadata')
data_path = '/home/u23/samfreitas/NN_trainings/IGTD/Results/human_healthy_std_2/data'
metadata_path = '/home/u23/samfreitas/NN_trainings/IGTD/Data/human_data/meta_filtered.csv'
imgs_list = natsorted(glob.glob(os.path.join(data_path,'*.txt')))
metadata = pd.read_csv(metadata_path)

healthy_idx = metadata['Healthy'].values
metadata_healthy = metadata.iloc[healthy_idx,:]

SRR_values = metadata_healthy['SRR.ID'].values
unique_tissues = np.unique(metadata_healthy['Tissue'].values)

for this_tissue in unique_tissues:

    # this_tissue = unique_tissues[2]
    print('Current tissue',this_tissue)

    X = []
    y = []
    print('reading in data')
    for count in tqdm.tqdm(range(len(imgs_list))):

        this_img = imgs_list[count]

        srr_id = os.path.basename(imgs_list[count])[1:-9]
        this_imgs_meta_idx = (SRR_values == srr_id)
        this_metadata = metadata_healthy.iloc[this_imgs_meta_idx,:]
        if (this_metadata['Tissue'].values == this_tissue).squeeze():
            y.append(metadata_healthy.iloc[this_imgs_meta_idx,:]['Age'].values.squeeze())
            X.append(np.loadtxt(this_img, comments='#',delimiter="\t",unpack=False))

        # if '-m-' in this_img:
        #     temp_str = this_img
        #     temp_str = temp_str.replace('_',' ').replace('-',' ')
        #     temp_nums = [int(s) for s in temp_str.split() if s.isdigit()]
        #     y.append(temp_nums[1])
        #     X.append(np.loadtxt(this_img, comments='#',delimiter="\t",unpack=False))

    if not y:
        print('BAD LIST')

    del count,this_img
    X = np.asarray(X) / 255
    y = np.asarray(y)

    X_scale = X - np.median(X,axis = 0)
    y_norm = (y - np.min(y))/np.max(y-np.min(y))

    val_idx = []
    for unique_num in np.unique(y)[0::2]:
        val_idx.append(np.where(y==unique_num)[0][0])

    train_idx = np.arange(y.shape[0])
    train_idx = np.delete(train_idx,val_idx)

    X_train = X_scale[train_idx]
    y_train = y_norm[train_idx]
    X_val = X_scale[val_idx]
    y_val = y_norm[val_idx]

    model = fully_connected_CNN(height=X.shape[1],width=X.shape[2],use_dropout=True)
    # model = ResNet50v2_regression(height=X.shape[1],width=X.shape[2],use_dropout=False)
    plot_model(model)

    save_checkpoints = tf.keras.callbacks.ModelCheckpoint(
        filepath = 'model_weights/cp.ckpt', monitor = 'val_loss',
        mode = 'min',save_best_only = True,save_weights_only = True, verbose = 1)
    redule_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor = 'val_loss', factor = 0.1, patience = 3, min_lr = 0.00001, verbose = 1)
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',min_delta = 0.01,patience = 20000, verbose = 1)

    # optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.002, momentum=0.9)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.00001, momentum=0.8)#, momentum=0.9)

    model.compile(optimizer=optimizer,loss='MAE',metrics=['MSE'])

    history = model.fit(X_train,y_train,
        validation_data = (X_val,y_val),
        batch_size=16,epochs=250,
        callbacks=[save_checkpoints,earlystop],
        verbose=1)

    del model

    model = fully_connected_CNN(height=X.shape[1],width=X.shape[2],use_dropout=False)
    model.compile(optimizer=optimizer,loss='MAE',metrics=['MSE'])
    model.load_weights('model_weights/cp.ckpt')

    eval_result = model.evaluate(X_scale,y_norm,batch_size=1,verbose=1,return_dict=True)

    plt.figure(1)

    predicted = model.predict(X_scale).squeeze()

    plt.scatter(y_norm,predicted,color = 'r',alpha=0.5)

    plt.plot(np.linspace(0, np.max(y_norm)),np.linspace(0, np.max(y_norm)))
    plt.xlabel('Expected Age (years)')
    plt.ylabel('Predicted Age (years)')

    plt.savefig(fname = "model_predictions" + str(this_tissue).replace('/','-') + ".png")

    plt.close('all')

    plt.figure(2)
    for this_key in list(history.history.keys()):
        b = history.history[this_key]
        plt.plot(b,label = this_key)

    res = dict()
    for key in eval_result: res[key] = round(eval_result[key],6)

    plt.legend(loc="upper left")
    plt.ylim([0,1])
    plt.title(json.dumps(res))
    plt.savefig(fname=  "training_history" + str(this_tissue).replace('/','-') + ".png")

    plt.close('all')

print('eof')