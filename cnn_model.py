import tensorflow as tf
import numpy as np
import os
import glob
import json
import matplotlib.pyplot as plt
from natsort import natsorted, natsort_keygen
from CNN_regression_model import fully_connected_CNN,ResNet50v2_regression, plot_model,load_rotated_minst_dataset

# (X,y), (X_val,y_val), (test_X,test_y) = load_rotated_minst_dataset(seed = 50)

data_path = 'data_1'
imgs_list = glob.glob(os.path.join(data_path,'*.txt'))

X = []
y = []
for count, this_img in enumerate(imgs_list):
    if '-m-' in this_img:
        temp_str = this_img
        temp_str = temp_str.replace('_',' ').replace('-',' ')
        temp_nums = [int(s) for s in temp_str.split() if s.isdigit()]
        y.append(temp_nums[1])
        X.append(np.loadtxt(this_img, comments='#',delimiter="\t",unpack=False))
del temp_str,temp_nums,count,this_img
X = np.asarray(X)
y = np.asarray(y)

val_idx = []
for unique_num in np.unique(y):
    val_idx.append(np.where(y==unique_num)[0][0])

train_idx = np.arange(y.shape[0])
train_idx = np.delete(train_idx,val_idx)

X_train = X[train_idx]
y_train = y[train_idx]
X_val = X[val_idx]
y_val = y[val_idx]

model = fully_connected_CNN(height=X.shape[1],width=X.shape[2],use_dropout=True)
# model = ResNet50v2_regression(height=X.shape[1],width=X.shape[2],use_dropout=False)
plot_model(model)

save_checkpoints = tf.keras.callbacks.ModelCheckpoint(
    filepath = 'model_weights/cp.ckpt', monitor = 'val_loss',
    mode = 'min',save_best_only = True,save_weights_only = True, verbose = 1)
redule_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_loss', factor = 0.1, patience = 3, min_lr = 0.00001, verbose = 1)
earlystop = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',min_delta = 0.01,patience = 10000, verbose = 1)

# optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.002, momentum=0.9)
optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.0025, momentum=0.9)

model.compile(optimizer=optimizer,loss='MAE',metrics=['MSE'])

history = model.fit(X_train,y_train,
    validation_data = (X_val,y_val),
    batch_size=32,epochs=12000,
    callbacks=[save_checkpoints,earlystop],
    verbose=0)

del model

model = fully_connected_CNN(height=X.shape[1],width=X.shape[2],use_dropout=False)
model.compile(optimizer=optimizer,loss='MAE',metrics=['MSE'])
model.load_weights('model_weights/cp.ckpt')

eval_result = model.evaluate(X,y,batch_size=1,verbose=1,return_dict=True)

plt.figure(1)
for i in range(X.shape[0]):
    X_ex = np.expand_dims(X[i],axis=0)
    predicted = model.predict(X_ex).squeeze()

    print("Expected:", y[i], " Predicted:", predicted)
    plt.scatter(y[i],predicted,color = 'r',alpha=0.5)
plt.plot(np.arange(np.max(y)))
plt.xlabel('Expected Age (months)')
plt.ylabel('Predicted Age (months)')

plt.savefig(fname = 'model_predictions.png')

plt.close('all')

plt.figure(2)
for this_key in list(history.history.keys()):
    b = history.history[this_key]
    plt.plot(b,label = this_key)

res = dict()
for key in eval_result: res[key] = eval_result[key]

plt.legend(loc="upper left")
plt.ylim([0,100])
plt.title(json.dumps(res))
plt.savefig(fname= "training_history.png")

plt.close('all')

print('eof')