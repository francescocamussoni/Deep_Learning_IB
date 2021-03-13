"""
date: 7-10-2020
File: ej6.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from textwrap import wrap
import seaborn as sns
sns.set(style='whitegrid')

diabetes = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')

###################################
        #Kfold sin correcion
###################################

x = diabetes[:, :-1]
y = diabetes[:, -1]
input_dim=x.shape[1]
neurons=24
epochs=100
batch_size=16
lr=1e-3
rf=1e-1

inputs = keras.layers.Input(shape=(input_dim,))
l1 = keras.layers.Dense(neurons,
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(rf))(inputs)
bn1 = keras.layers.BatchNormalization()(l1)
do1 = keras.layers.Dropout(0.25)(bn1)
l2 = keras.layers.Dense(neurons/2,
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(rf))(do1)
bn2 = keras.layers.BatchNormalization()(l2)
do2 = keras.layers.Dropout(0.25)(bn2)
outputs = keras.layers.Dense(1,
    activation='linear',
    kernel_regularizer=keras.regularizers.l2(rf))do2)
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

model.compile(optimizer=keras.optimizers.Adam(lr=lr),
    loss=keras.losses.BinaryCrossentropy(name='loss', from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name='acc', threshold=0.5)])

model.save_weights('weights_ej6')

acc=[]
val_acc=[]
test_acc=[]
loss=[]
val_loss=[]
test_loss=[]

kf = KFold(n_splits=5)

for train_index, val_index in kf.split(x):
    x_train, x_val = x[train_index], x[val_index]
    media=np.mean(x_train, axis=0)
    std=np.std(x_train, axis=0)
    x_train=(x_train-media)/std
    x_val=(x_val-media)/std
    y_train, y_val = y[train_index], y[val_index]

    model.load_weights('weights_ej6')

    history = model.fit(x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2)

    acc.append(history.history['acc'])
    val_acc.append(history.history['val_acc'])
    loss.append(history.history['acc'])
    val_loss.append(history.history['val_loss'])

acc_media=np.mean(np.array(acc), axis=0)
acc_min=np.min(np.array(acc), axis=0)
acc_max=np.max(np.array(acc), axis=0)
val_acc_media=np.mean(np.array(val_acc), axis=0)
val_acc_min=np.min(np.array(val_acc), axis=0)
val_acc_max=np.max(np.array(val_acc), axis=0)


fig=plt.figure(figsize=(12,6))
plt.subplot(121)
ax1 = plt.gca()
ax1.patch.set_edgecolor('black')
ax1.patch.set_linewidth('1')
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Accuracy', fontsize=14)
ax1.plot(np.arange(epochs), acc_media, color='red', linewidth=3, label='Training acc media')
ax1.plot(np.arange(epochs), acc_min, color='red', label='Training acc min', linestyle='dashdot')
ax1.plot(np.arange(epochs), acc_max, color='red', label='Training acc max', linestyle='dashed')
ax1.fill_between(np. arange(epochs), acc_min, acc_max, color='red', alpha=0.45)
ax1.set_ylim(0.3, 1)
plt.title('Resultados de entrenamiento para datos de training.', fontsize=15, wrap=True)
ax1.legend(loc='best')

plt.subplot(122)
ax1 = plt.gca()
ax1.patch.set_edgecolor('black')
ax1.patch.set_linewidth('1')
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Accuracy', fontsize=14)
ax1.plot(np.arange(epochs), val_acc_media, color='red', linewidth=3, label='val acc media')
ax1.plot(np.arange(epochs), val_acc_min, color='red', label='val acc min', linestyle='dashdot')
ax1.plot(np.arange(epochs), val_acc_max, color='red', label='val acc max', linestyle='dashed')
ax1.fill_between(np. arange(epochs), val_acc_min, val_acc_max, color='red', alpha=0.45)
ax1.set_ylim(0.3, 1)
plt.title('Resultados de entrenamiento para datos de validacion.', wrap=True, fontsize=15)
ax1.legend(loc='best')

plt.suptitle('Lr='+str(lr)+', rf='+str(rf), fontsize=16)
plt.savefig('ej6_sin_correccion.pdf', format='pdf')
plt.show()
plt.close()

###################################
        #Kfold con correcion
###################################
from sklearn.impute import SimpleImputer

x = diabetes[:, :-1]
y = diabetes[:, -1]
x_pregnacies = x[:, 0]
x_aux = x[:, 1:]
imp_mean = SimpleImputer(missing_values=0, strategy='mean')
x_aux = imp_mean.fit_transform(x_aux)
x=np.hstack((x_pregnacies[:,np.newaxis], x_aux))
input_dim=x.shape[1]
neurons=24
epochs=100
batch_size=24
lr=1e-3
rf=1e-1

acc=[]
val_acc=[]
test_acc=[]
loss=[]
val_loss=[]
test_loss=[]

kf = KFold(n_splits=5)

for train_index, val_index in kf.split(x):
    x_train, x_val = x[train_index], x[val_index]
    media=np.mean(x_train, axis=0)
    std=np.std(x_train, axis=0)
    x_train=(x_train-media)/std
    x_val=(x_val-media)/std
    y_train, y_val = y[train_index], y[val_index]

    model.load_weights('weights_ej6')

    history = model.fit(x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2)

    acc.append(history.history['acc'])
    val_acc.append(history.history['val_acc'])
    loss.append(history.history['acc'])
    val_loss.append(history.history['val_loss'])

acc_media=np.mean(np.array(acc), axis=0)
acc_min=np.min(np.array(acc), axis=0)
acc_max=np.max(np.array(acc), axis=0)
val_acc_media=np.mean(np.array(val_acc), axis=0)
val_acc_min=np.min(np.array(val_acc), axis=0)
val_acc_max=np.max(np.array(val_acc), axis=0)


fig=plt.figure(figsize=(12,6))
plt.subplot(121)
ax1 = plt.gca()
ax1.patch.set_edgecolor('black')
ax1.patch.set_linewidth('1')
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Accuracy', fontsize=14)
ax1.plot(np.arange(epochs), acc_media, color='red', linewidth=3, label='Training acc media')
ax1.plot(np.arange(epochs), acc_min, color='red', label='Training acc min', linestyle='dashdot')
ax1.plot(np.arange(epochs), acc_max, color='red', label='Training acc max', linestyle='dashed')
ax1.fill_between(np. arange(epochs), acc_min, acc_max, color='red', alpha=0.45)
ax1.set_ylim(0.3, 1)
plt.title('Resultados de entrenamiento para datos de training.', fontsize=15, wrap=True)
ax1.legend(loc='best')

plt.subplot(122)
ax1 = plt.gca()
ax1.patch.set_edgecolor('black')
ax1.patch.set_linewidth('1')
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Accuracy', fontsize=14)
ax1.plot(np.arange(epochs), val_acc_media, color='red', linewidth=3, label='val acc media')
ax1.plot(np.arange(epochs), val_acc_min, color='red', label='val acc min', linestyle='dashdot')
ax1.plot(np.arange(epochs), val_acc_max, color='red', label='val acc max', linestyle='dashed')
ax1.fill_between(np. arange(epochs), val_acc_min, val_acc_max, color='red', alpha=0.45)
ax1.set_ylim(0.3, 1)
plt.title('Resultados de entrenamiento para datos de validacion.', wrap=True, fontsize=15)
ax1.legend(loc='best')

plt.suptitle('Lr='+str(lr)+', rf='+str(rf), fontsize=16)
plt.savefig('ej6_con_correccion.pdf', format='pdf')
plt.show()
plt.close()
