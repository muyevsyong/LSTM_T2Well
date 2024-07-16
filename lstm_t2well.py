import datetime
import warnings
import re
import gc
import os

import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
assert len(gpu) == 1
tf.config.experimental.set_memory_growth(gpu[0], True)

settings = {
    "units": 128,
    "hidden_layers": 5,
    "lr": 0.0001,
    "patience": 20,
    "batch_size": 1024,
    "epochs_num":1000,
}

path = "../input/tough-lstm/"
file_list = os.listdir(path)
trn_data = []
val_data = []
time_steps = 11

trn_size = int(len(file_list) * 0.8)
for file in file_list[:trn_size]:
    data_tmp = np.genfromtxt(path + file).T
    for row in range(len(data_tmp) - time_steps):
        trn_data.append(np.expand_dims(data_tmp[row:row + time_steps], axis=0))

for file in file_list[trn_size:]:
    data_tmp = np.genfromtxt(path + file).T
    for row in range(len(data_tmp) - time_steps):
        val_data.append(np.expand_dims(data_tmp[row:row + time_steps], axis=0))
        

trn_data = np.concatenate(trn_data, axis=0)
val_data = np.concatenate(val_data, axis=0)

num_trn, _, num_feat = trn_data.shape
num_val, _, _ = val_data.shape

trn_data = trn_data.reshape([-1, num_feat])
val_data = val_data.reshape([-1, num_feat])

trn_mean = np.mean(trn_data, axis=0)
trn_std = np.std(trn_data, axis=0)
trn_data = (trn_data - trn_mean) / trn_std
val_data = (val_data - trn_mean) / trn_std

trn_data = trn_data.reshape([num_trn, time_steps, num_feat])
val_data = val_data.reshape([num_val, time_steps, num_feat])

settings["trg_num"] = trn_data.shape[-1]

trn_x = trn_data[:, :time_steps]
trn_y = trn_data[:, -1]

val_x = val_data[:, :time_steps]
val_y = val_data[:, -1]

settings["input_shape"] = trn_x.shape[1:]

def make_dataset(feature, y, batch_size=1024):
    ds = tf.data.Dataset.from_tensor_slices((feature, y))
    ds = ds.shuffle(2*batch_size).batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
    return ds
    

trn_ds = make_dataset(trn_x, trn_y)
val_ds = make_daztaset(val_x, val_y)

def get_model(settings):
    seq = []
    seq.append(
        tf.keras.layers.LSTM(settings["units"],
                             recurrent_regularizer=tf.keras.regularizers.l1_l2(
                                 l1=0, l2=0.01),
                             return_sequences=True,
                             activation='tanh',
                             input_shape=settings["input_shape"]))
    for i in range(settings["hidden_layers"]):
        seq.append(
            tf.keras.layers.LSTM(
                settings["units"],
                recurrent_regularizer=tf.keras.regularizers.l1_l2(l1=0,
                                                                  l2=0.01),
                return_sequences=True,
                activation='tanh'))

    seq.append(
        tf.keras.layers.LSTM(settings["units"],
                             recurrent_regularizer=tf.keras.regularizers.l1_l2(
                                 l1=0, l2=0.01),
                             return_sequences=False,
                             activation='tanh'))

    seq.append(tf.keras.layers.Dense(settings["trg_num"]))

    model = tf.keras.models.Sequential(seq)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=settings["lr"]),
        loss="mse")
    return model
    
    
model = get_model(settings)
model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=settings["patience"], mode='min')
ckpt = tf.keras.callbacks.ModelCheckpoint('./checkpoint',
                                          monitor='val_loss',
                                          verbose=0,
                                          save_best_only=True,
                                          save_weights_only=True)

history = model.fit(trn_ds,
    batch_size=settings["batch_size"],
    epochs=settings["epochs_num"],
    shuffle=True,  #默认为True,在每一个batch里面进行shuffle
    verbose=2,
    validation_data=val_ds,
    callbacks=[early_stopping, ckpt])

model.load_weights("./checkpoint")

def myMase(pred, true, m=1):
    '''m是周期长度'''
    return np.mean(np.abs(pred-true))/np.mean(np.abs(true[m:] - true[:-m]))

def myRmse(pred, true):
    return np.sqrt(np.mean(np.square(pred-true)))
    

pd.DataFrame(history.history)[[
    'loss', 'val_loss'
]].plot(figsize=(12, 4), color=['r', 'b'])
plt.grid(True)
#plt.gca().set_ylim(0,60)
plt.xlabel('epoch', fontsize=15)
plt.ylabel('MSE', fontsize=15)
#plt.title('9-16-0.0063-500',fontsize=25)
plt.show()

pred_trn = model.predict(trn_x, verbose=0)
mase1 = myMase(pred_trn, trn_y)
print(mase1)
rmse1 = myRmse(pred_trn, trn_y)
print(rmse1)
print(myRmse(trn_y[:-1], trn_y[1:]))

for i, feat in enumerate(['a','b']):
    tmp = pd.DataFrame(
        np.concatenate([pred_val[:100, i:i + 1], val_y[:100, i:i + 1]], axis=1))
    tmp.columns = ['pred', 'true']

    tmp.plot(figsize=(18, 4), color=['r', 'b'])
    plt.grid(True)
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('value', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(feat, fontsize=20)
    plt.legend(fontsize=15)
    plt.show()
    
i = 98
tmp = pd.DataFrame(
    np.concatenate([pred_val[:100, i:i + 1], val_y[:100, i:i + 1]], axis=1))
tmp.columns = ['pred', 'true']

tmp.plot(figsize=(18, 4), color=['r', 'b'])
plt.grid(True)
plt.xlabel('epoch', fontsize=15)
plt.ylabel('value', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title(feat, fontsize=20)
plt.legend(fontsize=15)
plt.show()