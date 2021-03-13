"""
date: 7-10-2020
File: ej10.py
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
from tensorflow.keras.datasets import cifar10, cifar100
from textwrap import wrap
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#cargo datos
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=10000, stratify=y_train)

media = np.mean(x_train, axis=0)
x_train = (x_train-media)/255
x_val = (x_val-media)/255
x_test = (x_test-media)/255

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
y_val=to_categorical(y_val)

clases=y_train.shape[1]

#############################
        #AlexNet
#############################
lr=1e-3
rf=10
batch_size=100
epochs=100

alexnet = keras.models.Sequential(name='Ejercicio_9_alexnet')
alexnet.add(keras.layers.Input(shape=x_train[0].shape))
alexnet.add(keras.layers.Conv2D(96,
    (3, 3),
    activation='relu',
    strides=4,
    padding='same'))
alexnet.add(keras.layers.BatchNormalization())
alexnet.add(keras.layers.MaxPooling2D((3, 3),
    strides=2,
    padding='same'))
alexnet.add(keras.layers.Conv2D(256,
    (5, 5),
    activation='relu',
    strides=1,
    padding='same'))
alexnet.add(keras.layers.BatchNormalization())
alexnet.add(keras.layers.MaxPooling2D((3, 3),
    strides=2,
    padding='same'))
alexnet.add(keras.layers.Conv2D(384,
    (3, 3),
    activation='relu',
    strides=1,
    padding='same'))
alexnet.add(keras.layers.BatchNormalization())
alexnet.add(keras.layers.Conv2D(184,
    (3, 3),
    activation='relu',
    strides=1,
    padding='same'))
alexnet.add(keras.layers.BatchNormalization())
alexnet.add(keras.layers.Conv2D(256,
    (3, 3),
    activation='relu',
    strides=1,
    padding='same'))
alexnet.add(keras.layers.BatchNormalization())
alexnet.add(keras.layers.MaxPooling2D((3, 3),
    strides=2,
    padding='same'))
alexnet.add(keras.layers.Flatten())
alexnet.add(keras.layers.Dropout(0.8))
alexnet.add(keras.layers.BatchNormalization())
alexnet.add(keras.layers.Dense(1024,
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(rf)))
alexnet.add(keras.layers.Dropout(0.8))
alexnet.add(keras.layers.BatchNormalization())
alexnet.add(keras.layers.Dense(1024,
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(rf)))
alexnet.add(keras.layers.Dense(clases,
    activation='linear',
    kernel_regularizer=keras.regularizers.l2(rf)))

alexnet.compile(keras.optimizers.Adam(learning_rate=lr),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.CategoricalAccuracy(name='acc')])
alexnet.summary()

idg = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=5,
    height_shift_range=5,
    shear_range=0,
    zoom_range=0,
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=False,
)  #cabre me dio una mano con esto

history = alexnet.fit(idg.flow(x_train, y_train, batch_size=batch_size),
    epochs=epochs,
    steps_per_epoch=len(x_train) / batch_size,
    validation_data=(x_val, y_val),
    verbose=2)

loss, acc = alexnet.evaluate(x_test, y_test)
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
plt.savefig('ej10_alexnet.pdf', format='pdf')
plt.show()
plt.close()

#############################
        #VGG16
#############################
lr=1e-3
rf=10
batch_size=100
epochs=50

VGG16 = keras.models.Sequential(name='Ejercicio_9_VGG16')
VGG16.add(keras.layers.Input(shape=x_train[0].shape))
VGG16.add(keras.layers.Conv2D(32,
    (3, 3),
    activation='relu',
    strides=1,
    padding='same'))
VGG16.add(keras.layers.BatchNormalization())
VGG16.add(keras.layers.Conv2D(32,
    (3, 3),
    activation='relu',
    strides=1,
    padding='same'))
VGG16.add(keras.layers.BatchNormalization())
VGG16.add(keras.layers.MaxPooling2D((3, 3),
    strides=2,
    padding='same'))
VGG16.add(keras.layers.Conv2D(64,
    (3, 3),
    activation='relu',
    strides=1,
    padding='same'))
VGG16.add(keras.layers.BatchNormalization())
VGG16.add(keras.layers.Conv2D(64,
    (3, 3),
    activation='relu',
    strides=1,
    padding='same'))
VGG16.add(keras.layers.BatchNormalization())
VGG16.add(keras.layers.MaxPooling2D((3, 3),
    strides=2,
    padding='same'))
VGG16.add(keras.layers.Conv2D(128,
    (3, 3),
    activation='relu',
    strides=1,
    padding='same'))
VGG16.add(keras.layers.BatchNormalization())
VGG16.add(keras.layers.Conv2D(128,
    (3, 3),
    activation='relu',
    strides=1,
    padding='same'))
VGG16.add(keras.layers.BatchNormalization())
VGG16.add(keras.layers.Conv2D(128,
    (3, 3),
    activation='relu',
    strides=1,
    padding='same'))
VGG16.add(keras.layers.BatchNormalization())
VGG16.add(keras.layers.MaxPooling2D((3, 3),
    strides=2,
    padding='same'))
VGG16.add(keras.layers.Conv2D(256,
    (3, 3),
    activation='relu',
    strides=1,
    padding='same'))
VGG16.add(keras.layers.BatchNormalization())
VGG16.add(keras.layers.Conv2D(256,
    (3, 3),
    activation='relu',
    strides=1,
    padding='same'))
VGG16.add(keras.layers.BatchNormalization())
VGG16.add(keras.layers.Conv2D(256,
    (3, 3),
    activation='relu',
    strides=1,
    padding='same'))
VGG16.add(keras.layers.BatchNormalization())
VGG16.add(keras.layers.MaxPooling2D((3, 3),
    strides=2,
    padding='same'))
VGG16.add(keras.layers.Conv2D(256,
    (3, 3),
    activation='relu',
    strides=1,
    padding='same'))
VGG16.add(keras.layers.BatchNormalization())
VGG16.add(keras.layers.Conv2D(256,
    (3, 3),
    activation='relu',
    strides=1,
    padding='same'))
VGG16.add(keras.layers.BatchNormalization())
VGG16.add(keras.layers.Conv2D(256,
    (3, 3),
    activation='relu',
    strides=1,
    padding='same'))
VGG16.add(keras.layers.BatchNormalization())
VGG16.add(keras.layers.MaxPooling2D((3, 3),
    strides=2,
    padding='same'))
VGG16.add(keras.layers.Flatten())
VGG16.add(keras.layers.Dropout(0.5))
VGG16.add(keras.layers.BatchNormalization())
VGG16.add(keras.layers.Dense(6272,
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(rf)))
VGG16.add(keras.layers.Dropout(0.5))
VGG16.add(keras.layers.BatchNormalization())
VGG16.add(keras.layers.Dense(1024,
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(rf)))
VGG16.add(keras.layers.Dropout(0.5))
VGG16.add(keras.layers.BatchNormalization())
VGG16.add(keras.layers.Dense(1024,
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(rf)))
VGG16.add(keras.layers.Dense(clases,
    activation='linear',
    kernel_regularizer=keras.regularizers.l2(rf)))

VGG16.compile(keras.optimizers.Adam(learning_rate=lr),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.CategoricalAccuracy(name='acc')])
VGG16.summary()

idg = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=5,
    height_shift_range=5,
    shear_range=0,
    zoom_range=0,
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=False,
)  #cabre me dio una mano con esto

history = VGG16.fit(idg.flow(x_train, y_train, batch_size=batch_size),
    epochs=epochs,
    steps_per_epoch=len(x_train) / batch_size,
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
plt.savefig('ej10_VGG16.pdf', format='pdf')
plt.show()
plt.close()
