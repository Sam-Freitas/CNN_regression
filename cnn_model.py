import tensorflow as tf
import numpy as np
from CNN_regression_model import fully_connected_CNN,ResNet50v2_regression, plot_model,load_rotate_minst_dataset

(X,y), (test_X,test_y) = load_rotate_minst_dataset()

model = fully_connected_CNN(height=X.shape[1],width=X.shape[2],use_dropout=True)
# model = ResNet50v2_regression(height=X.shape[1],width=X.shape[2],use_dropout=False)
plot_model(model)

save_checkpoints = tf.keras.callbacks.ModelCheckpoint(
    filepath = 'model_weights/cp.ckpt', monitor = 'val_loss',
    mode = 'min',save_best_only = True,save_weights_only = True, verbose = 1)
redule_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_loss', factor = 0.5, patience = 10, min_lr = 0.00001, verbose = 1)
earlystop = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',min_delta = 0.01,patience = 50, verbose = 1)

model.compile(optimizer='adam',loss='MAE',metrics=['MSE'])

history = model.fit(X,y,
    batch_size=64,epochs=150,validation_split=0.1,
    callbacks=[save_checkpoints,redule_lr,earlystop])

del model

model = fully_connected_CNN(height=X.shape[1],width=X.shape[2],use_dropout=False)
model.compile(optimizer='adam',loss='MAE',metrics=['MSE'])
model.load_weights('model_weights/cp.ckpt')

model_eval = model.evaluate(test_X,test_y,batch_size=1,verbose=1,return_dict=True)

for i in range(10):
    X_ex = np.expand_dims(test_X[i],axis=0)
    predicted = model.predict(X_ex).squeeze()

    print("Expected:", test_y[i], " Predicted:", predicted)


print('eof')