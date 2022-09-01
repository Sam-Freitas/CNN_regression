import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import glob
import json
import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from natsort import natsorted, natsort_keygen
from CNN_regression_model import fully_connected_CNN_v4, plot_model,test_on_improved_val_lossv3,diff_func, add_noise_to_dataset
from sklearn.preprocessing import PowerTransformer
from pathlib import Path

print('reading in data')
plt.ioff()

this_tissue = 'GTEx_diff'

temp = np.load('data_arrays/test.npz')
X_test,y_test = temp['X'],temp['y']

# inital_filter_size = 8
# dropsize = 0.85
# blocksize = 5
# layers = 3
# sublayers = 0
# age_normalizer = 1
# # input_height = 74
# # input_width = 130
# input_height = input_width = 130
# epochs = 15000
# batch_size = 128

# inital_filter_size = 12
# dropsize = 0.95
# blocksize = 5
# layers = 3
# sublayers = 0
# age_normalizer = 1
# input_height = input_width = 130
# dense_size = 1024
# batch_size = 128
# lr = 0.0001

# epochs = 10000

inital_filter_size = 12
dropsize = 0.95
blocksize = 5
layers = 3
sublayers = 0
age_normalizer = 1
input_height = input_width = 130
dense_size = 1024
batch_size = 128
lr = 0.0001

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    lr,
    decay_steps=50,
    decay_rate=0.96,
    staircase=True)

epochs = 1000

k_folds = glob.glob(os.path.join('data_arrays','*.npz'))
num_k_folds = 0
for npzs in k_folds:
    if 'train' in npzs:
        num_k_folds += 1

temp = np.load('data_arrays/All_data.npz')
X_norm,y_norm = temp['X'],temp['y']

assert X_norm.shape[1] == input_height and input_height == X_norm.shape[2]

training_histories = []
val_loss_hist = []

for i in range(8,num_k_folds):
    temp = np.load('data_arrays/train'+ str(i) +'.npz')
    train_idx = temp['idx']
    temp = np.load('data_arrays/val'+ str(i) +'.npz')
    val_idx = temp['idx']

    X_train, y_train = X_norm[train_idx],y_norm[train_idx]
    X_val, y_val = X_norm[val_idx],y_norm[val_idx]

    # X_train,y_train = add_noise_to_dataset(X_train, y_train)

    X_train, y_train = diff_func(X_train, y_train,age_normalizer=age_normalizer)
    X_val, y_val = diff_func(X_val, y_val,age_normalizer=age_normalizer)

    model = fully_connected_CNN_v4(
        height=input_height,width=input_width,channels=2,
        use_dropout=True,inital_filter_size=inital_filter_size,keep_prob = dropsize,blocksize = blocksize,
        layers = layers, sub_layers = sublayers,dense_size = dense_size
    )
    sample_weights = (np.abs(y_train)+1)**(1/2)

    save_checkpoints = tf.keras.callbacks.ModelCheckpoint(
        filepath = 'checkpoints/checkpoints' + str(i) + '/cp.ckpt', monitor = 'val_loss',
        mode = 'min',save_best_only = True,save_weights_only = True, verbose = 1)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 200, 
        restore_best_weights=True) # patience 250
    Reduce_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=450)
    on_epoch_end = test_on_improved_val_lossv3()
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    model.compile(optimizer=optimizer,loss='MeanAbsoluteError',
        metrics = ['RootMeanSquaredError'])

    inital_epoch = (i*epochs)
    this_epochs = inital_epoch + epochs

    model.summary()

    history = model.fit([X_train],y_train,
        validation_data = ([X_val],y_val),
        batch_size=batch_size,epochs=this_epochs, initial_epoch = inital_epoch,
        callbacks=[on_epoch_end,save_checkpoints,earlystop,Reduce_LR],
        verbose=1,
        sample_weight = sample_weights) 

    training_histories.append(history)
    val_loss_hist.append(history.history['val_loss'])

    df = pd.DataFrame(val_loss_hist)
    df.to_csv('checkpoints/val_loss.csv')

    # Path('model_weights').mkdir( parents=True, exist_ok=True )
    # model.save_weights('model_weights/model_weights' + str(i) + '.h5')

    del model

# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
	stackX = None
	for model in members:
		# make prediction
		yhat = model.predict(inputX,batch_size=batch_size)
		# stack predictions into [rows, members, probabilities]
		if stackX is None:
			stackX = yhat
		else:
			stackX = np.dstack((stackX, yhat))
	# flatten predictions to [rows, members x probabilities]
	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
	return stackX

def define_stacked_model(members):
	# update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
			# make not trainable
            layer.trainable = False
			# rename to avoid 'unique layer name' issue
            layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
	# define multi-headed input
    ensemble_visible = [model.input for model in members]
	# concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    merge = tf.keras.layers.concatenate(ensemble_outputs)
    hidden = tf.keras.layers.Dense(512, activation='relu')(merge)
    output = tf.keras.layers.Dense(1, activation='linear')(hidden)
    model = tf.keras.Model(inputs=ensemble_visible, outputs=output)
	# compile
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(loss='MeanAbsoluteError', optimizer=optimizer)
    return model

def fit_stacked_model(model, inputX, inputy):
	# prepare input data
    X = [inputX for _ in range(len(model.input))]
    sub_epochs = 100
    es = tf.keras.callbacks.EarlyStopping(patience=sub_epochs,restore_best_weights=True,verbose=1)
	# fit model
    model.fit(X, inputy, epochs=sub_epochs, verbose=1, callbacks=es, validation_split=0.2)
    return model

def predict_stacked_model(model, inputX):
	# prepare input data
	X = [inputX for _ in range(len(model.input))]
	# make prediction
	return model.predict(X, verbose=0)

models = []
count = 0
print('Stacking models')
for i in range(num_k_folds):

    this_train_hist = training_histories[i].history['val_loss']

    if (np.max(this_train_hist) - np.min(this_train_hist)) > (1/age_normalizer):

        models.append(
            fully_connected_CNN_v4(
            height=input_height,width=input_width,channels=2,
            use_dropout=False,inital_filter_size=inital_filter_size,keep_prob = dropsize,blocksize = blocksize,
            layers = layers, sub_layers = sublayers, dense_size = dense_size)
        )
        checkpoint_path = 'checkpoints/checkpoints' + str(i) + '/cp.ckpt'
        models[count].load_weights(checkpoint_path)
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
        models[count].compile(optimizer=optimizer,loss='MeanAbsoluteError',metrics=['RootMeanSquaredError'])

        count = count + 1


print("using:", count, "of the cross valid")
X_train, y_train = diff_func(X_norm, y_norm,age_normalizer=age_normalizer)

print('stacking models')
stacked_model = define_stacked_model(models)
print('fitting ensemble model')
stacked_model = fit_stacked_model(stacked_model, X_train, y_train)
print('saving ensemble model')
stacked_model.save('stacked_model2')

# ensemble_prediction_test = stacked_dataset(models, X_test)
# ensemble_prediction_train = stacked_dataset(models, X_train)

print('predicting using ensemble model')
ensemble_prediction_test = predict_stacked_model(stacked_model, X_test)
ensemble_prediction_train = predict_stacked_model(stacked_model, X_train)

avg_prediction_test = np.mean(ensemble_prediction_test,axis = 1)
avg_prediction_train = np.mean(ensemble_prediction_train,axis = 1)

cor_matrix = np.corrcoef(avg_prediction_test.squeeze(),y_test)
cor_xy = cor_matrix[0,1]
r_squared_test = round(cor_xy**2,4)
print("Test",r_squared_test)

cor_matrix = np.corrcoef(avg_prediction_train.squeeze(),y_train)
cor_xy = cor_matrix[0,1]
r_squared_train = round(cor_xy**2,4)
print("Train",r_squared_train)

test_error = np.sum(np.abs(y_test-avg_prediction_test))/len(y_test)
train_error = np.sum(np.abs(y_train-avg_prediction_train))/len(y_train)
m, b = np.polyfit(y_test,avg_prediction_test, deg = 1)

plt.figure(1)
plt.scatter(y_train,avg_prediction_train,color = 'r',alpha=0.05, label = 'training data')
plt.scatter(y_test,avg_prediction_test,color = 'b',alpha=0.3, label = 'testing data')
plt.plot(np.linspace(np.min(y_train), np.max(y_train)),np.linspace(np.min(y_train), np.max(y_train)))
plt.plot(y_test, m*y_test + b,'m-')

plt.text(np.min(y_train),np.max(y_train),"r^2: " + str(r_squared_train),fontsize = 12, color = 'r')
plt.text(np.min(y_train),np.max(y_train)-(0.2)*np.max(y_train),"r^2: " + str(r_squared_test),fontsize = 12, color = 'b')

plt.legend(loc = 'upper center')
plt.title('Train error: ' + str(round(train_error,4)) + ' --- Test error: ' + str(round(test_error,4)),fontsize = 10)
plt.xlabel('Expected Age (years)')
plt.ylabel('Predicted Age (years)')

plt.savefig(fname = "test_model_predictions" + str(this_tissue).replace('/','-') + ".png")

plt.close('all')

# print("earlystop weights")

# model = fully_connected_CNN_v3(
#     height=X_train.shape[1],width=X_train.shape[2],
#     use_dropout=False,inital_filter_size=inital_filter_size,dropsize = dropsize,blocksize = blocksize,
#     layers = layers, sub_layers = sublayers
#     )
# model.compile(optimizer=optimizer,loss='MeanAbsoluteError',metrics=['MeanSquaredError','RootMeanSquaredError'])
# model.load_weights('model_weights/model_weights')

# eval_result = model.evaluate([X_test],[y_test],batch_size=1,verbose=0,return_dict=True)
# print(eval_result)

# res = dict()
# for key in eval_result: res[key] = round(eval_result[key],6)

# plt.figure(1)

# predicted_test = model.predict([X_test],batch_size=1).squeeze()
# predicted_train = model.predict([X_all],batch_size=1).squeeze()

# cor_matrix = np.corrcoef(predicted_test.squeeze(),y_test)
# cor_xy = cor_matrix[0,1]
# r_squared_test = round(cor_xy**2,4)
# print("test",r_squared_test)

# cor_matrix = np.corrcoef(predicted_train.squeeze(),y_all)
# cor_xy = cor_matrix[0,1]
# r_squared_train = round(cor_xy**2,4)
# print("train",r_squared_train)

# model.save('compiled_models/' + str(r_squared_test)[2:] + this_tissue + '_' + str(num))

# plt.scatter(y_all,predicted_train,color = 'r',alpha=0.2, label = 'training data')
# plt.scatter(y_test,predicted_test,color = 'b',alpha=0.3, label = 'testing data')
# plt.plot(np.linspace(np.min(y_all), np.max(y_all)),np.linspace(np.min(y_all), np.max(y_all)))

# plt.text(np.min(y_all),np.max(y_all),"r^2: " + str(r_squared_train),fontsize = 12, color = 'r')
# plt.text(np.min(y_all),np.max(y_all)-(0.2)*np.max(y_all),"r^2: " + str(r_squared_test),fontsize = 12, color = 'b')

# plt.legend(loc = 'upper center')
# plt.title(json.dumps(res).replace(',','\n'),fontsize = 10)
# plt.xlabel('Expected Age (years)')
# plt.ylabel('Predicted Age (years)')

# plt.savefig(fname = "test_model_predictions" + str(this_tissue).replace('/','-') + ".png")

# plt.close('all')

# plt.figure(3)
# for this_key in list(history.history.keys()):
#     b = history.history[this_key]
#     plt.plot(b,label = this_key)

# plt.legend(loc="upper left")
# plt.ylim([0,30])
# plt.title(json.dumps(res))
# plt.savefig(fname=  "training_history" + str(this_tissue).replace('/','-') + ".png")

# plt.close('all')

# print("Restored weights")
# del model

# model = fully_connected_CNN_v3(
#     height=X_train.shape[1],width=X_train.shape[2],
#     use_dropout=False,inital_filter_size=inital_filter_size,dropsize = dropsize,blocksize = blocksize,
#     layers = layers, sub_layers = sublayers
#     )
# model.compile(optimizer=optimizer,loss='MeanAbsoluteError',metrics=['MeanSquaredError','RootMeanSquaredError'])
# model.load_weights('checkpoints/cp.ckpt')

# eval_result = model.evaluate([X_test],[y_test],batch_size=1,verbose=0,return_dict=True)
# print(eval_result)

# res = dict()
# for key in eval_result: res[key] = round(eval_result[key],6)

# plt.figure(1)

# predicted_test = model.predict([X_test],batch_size=1).squeeze()
# predicted_train = model.predict([X_all],batch_size=1).squeeze()

# cor_matrix = np.corrcoef(predicted_test.squeeze(),y_test)
# cor_xy = cor_matrix[0,1]
# r_squared_test = round(cor_xy**2,4)
# print("test",r_squared_test)

# cor_matrix = np.corrcoef(predicted_train.squeeze(),y_all)
# cor_xy = cor_matrix[0,1]
# r_squared_train = round(cor_xy**2,4)
# print("train",r_squared_train)

# model.save('compiled_models/' + str(r_squared_test)[2:] + this_tissue + '_' + str(num))

# plt.scatter(y_all,predicted_train,color = 'r',alpha=0.2, label = 'training data')
# plt.scatter(y_test,predicted_test,color = 'b',alpha=0.3, label = 'testing data')
# plt.plot(np.linspace(np.min(y_all), np.max(y_all)),np.linspace(np.min(y_all), np.max(y_all)))

# plt.text(np.min(y_all),np.max(y_all),"r^2: " + str(r_squared_train),fontsize = 12, color = 'r')
# plt.text(np.min(y_all),np.max(y_all)-(0.2)*np.max(y_all),"r^2: " + str(r_squared_test),fontsize = 12, color = 'b')

# plt.legend(loc = 'upper center')
# plt.title(json.dumps(res).replace(',','\n'),fontsize = 10)
# plt.xlabel('Expected Age (years)')
# plt.ylabel('Predicted Age (years)')

# plt.savefig(fname = "test_model_predictions_best" + str(this_tissue).replace('/','-') + ".png")

# plt.close('all')

# plt.close('all')

# print('eof')