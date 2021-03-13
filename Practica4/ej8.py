"""
date: 7-10-2020
File: ej8.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from textwrap import wrap
import seaborn as sns
from tensorflow.keras.utils import to_categorical
sns.set(style='whitegrid')

#cargo datos
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=np.reshape(x_train, (x_train.shape[0], np.prod(x_train.shape[1:])))
x_test=np.reshape(x_test, (x_test.shape[0], np.prod(x_test.shape[1:])))
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=10000, stratify=y_train)

x_train = x_train/255
x_val = x_val/255
x_test = x_test/255

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
y_val=to_categorical(y_val)

#######################
        #densa
#######################
input_dim=x_train.shape[1]
lr=1e-4
rf=1e-2
batch_size=100
epochs=50

model = keras.models.Sequential(name='Ejercicio_8_densa')
model.add(keras.layers.Dense(784, input_shape=(input_dim,),
    kernel_regularizer=keras.regularizers.l2(rf),
    activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(100, input_shape=(input_dim,),
    kernel_regularizer=keras.regularizers.l2(rf),
    activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10,
    kernel_regularizer=keras.regularizers.l2(rf),
    activation='linear'))
model.compile(keras.optimizers.Adam(learning_rate=lr),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.CategoricalAccuracy(name='acc')])
model.summary()

history = model.fit(x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(x_val, y_val),
    verbose=2)

loss, acc = model.evaluate(x_test, y_test)
print('El accuracy sobre los datos de validacion fue= ', acc)

fig=plt.figure(figsize=(9,6))
ax1 = plt.gca()
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1=plt.subplot()
ax1.scatter(np.arange(epochs), history.history['loss'], color='red', alpha=0.5, marker='^')
ax1.plot(np.arange(epochs), history.history['loss'], color='red', label='Training loss')
ax1.scatter(np.arange(epochs), history.history['val_loss'], color='blue', alpha=0.5, marker='*')
ax1.plot(np.arange(epochs), history.history['val_loss'], color='blue', label='Validation loss')
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Costo', fontsize=14)
ax1.hlines(min(history.history['loss']), 0, epochs, color='red', linestyles='dashdot', alpha=0.5)
ax1.hlines(min(history.history['val_loss']), 0, epochs, color='blue', linestyles='dashdot', alpha=0.5)
ax1.set_yscale('log')
ax2 = plt.gca()
ax2.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax2=ax1.twinx()
ax2.scatter(np.arange(epochs), history.history['acc'], marker='o', alpha=0.5)
ax2.plot(np.arange(epochs), history.history['acc'], color='green', label='Training acc')
ax2.scatter(np.arange(epochs), history.history['val_acc'], marker='.', alpha=0.5)
ax2.plot(np.arange(epochs), history.history['val_acc'], color='orange', label='Validation acc')
ax2.set_xlabel('Época', fontsize=14), ax2.set_ylabel('Precisión', fontsize=14)
ax2.hlines(max(history.history['acc']), 0, epochs, color='green', linestyles='dashdot', alpha=0.5)
ax2.hlines(max(history.history['val_acc']), 0, epochs, color='orange', linestyles='dashdot', alpha=0.6)
plt.title('Resultados de entrenamiento. Lr='+str(lr)+', rf='+str(rf), fontsize=15)
ax1.legend(loc=7)
ax2.legend(loc='center')
plt.savefig('ej8_densa.pdf', format='pdf')
plt.show()
plt.close()

#######################
        #conv
#######################
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
x_val = np.reshape(x_val, (len(x_val), 28, 28, 1))

input_dim=x_train.shape[1]
lr=1e-4
rf=1e-2
batch_size=100
epochs=50

model_conv = keras.models.Sequential(name='Ejercicio_8_conv')
model_conv.add(keras.layers.Input(shape=x_train[0].shape))
model_conv.add(keras.layers.Conv2D(10,
    (24, 24),
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(rf),
    padding='same'))
model_conv.add(keras.layers.MaxPooling2D((2, 2),
    padding='same'))
model_conv.add(keras.layers.Conv2D(10,
    (12, 12),
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(rf),
    padding='same'))
model_conv.add(keras.layers.Flatten())
model_conv.add(keras.layers.Dense(10,
    activation='linear',
    kernel_regularizer=keras.regularizers.l2(rf)))

model_conv.compile(keras.optimizers.Adam(learning_rate=lr),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.CategoricalAccuracy(name='acc')])
model_conv.summary()

history = model_conv.fit(x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(x_val, y_val),
    verbose=2)

loss, acc = model_conv.evaluate(x_test, y_test)
print('El accuracy sobre los datos de validacion fue= ', acc)

fig=plt.figure(figsize=(9,6))
ax1 = plt.gca()
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1=plt.subplot()
ax1.scatter(np.arange(epochs), history.history['loss'], color='red', alpha=0.5, marker='^')
ax1.plot(np.arange(epochs), history.history['loss'], color='red', label='Training loss')
ax1.scatter(np.arange(epochs), history.history['val_loss'], color='blue', alpha=0.5, marker='*')
ax1.plot(np.arange(epochs), history.history['val_loss'], color='blue', label='Validation loss')
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Costo', fontsize=14)
ax1.hlines(min(history.history['loss']), 0, epochs, color='red', linestyles='dashdot', alpha=0.5)
ax1.hlines(min(history.history['val_loss']), 0, epochs, color='blue', linestyles='dashdot', alpha=0.5)
ax1.set_yscale('log')
ax2 = plt.gca()
ax2.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax2=ax1.twinx()
ax2.scatter(np.arange(epochs), history.history['acc'], marker='o', alpha=0.5)
ax2.plot(np.arange(epochs), history.history['acc'], color='green', label='Training acc')
ax2.scatter(np.arange(epochs), history.history['val_acc'], marker='.', alpha=0.5)
ax2.plot(np.arange(epochs), history.history['val_acc'], color='orange', label='Validation acc')
ax2.set_xlabel('Época', fontsize=14), ax2.set_ylabel('Precisión', fontsize=14)
ax2.hlines(max(history.history['acc']), 0, epochs, color='green', linestyles='dashdot', alpha=0.5)
ax2.hlines(max(history.history['val_acc']), 0, epochs, color='orange', linestyles='dashdot', alpha=0.6)
plt.title('Resultados de entrenamiento. Lr='+str(lr)+', rf='+str(rf), fontsize=15)
ax1.legend(loc=7)
ax2.legend(loc='center')
plt.savefig('ej8_conv.pdf', format='pdf')
plt.show()
plt.close()
