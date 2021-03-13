"""
date: 7-10-2020
File: ej9.py
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

permutation = np.random.permutation(28*28)
x_train_p = x_train.reshape(x_train.shape[0], -1)
x_train_p = x_train_p[:,permutation]

x_test_p = x_test.reshape(x_test.shape[0], -1)
x_test_p = x_test_p[:,permutation]

x_train_p, x_val_p, y_train, y_val = train_test_split(x_train_p,
    y_train,
    test_size=10000,
    stratify=y_train)

x_train_p = x_train_p/255
x_val_p = x_val_p/255
x_test_p = x_test_p/255

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
y_val=to_categorical(y_val)

fig = plt.figure()
ax1 = plt.subplot(121)
ax1.imshow(x_train[0], cmap='binary')
ax1.set_xticks([])
ax1.set_yticks([])
plt.title('Imagen antes de permutar', fontsize=15)
ax2 = plt.subplot(122)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.imshow(x_train_p[0].reshape(x_train[0].shape), cmap='binary')
plt.title('Imagen luego de permutar', fontsize=15)
plt.suptitle('Primer número del set de MNIST', fontsize=16)
fig.tight_layout()
fig.subplots_adjust(top=1)
plt.savefig('ej9_b_a.pdf', format='pdf')
plt.show()

#######################
        #densa
#######################
input_dim=x_train_p.shape[1]
lr=1e-4
rf=1e-2
batch_size=100
epochs=50

model = keras.models.Sequential(name='Ejercicio_8_densa')
model.add(keras.layers.Dense(784, input_shape=(input_dim,),
    kernel_regularizer=keras.regularizers.l2(rf),
    activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5)) #le saque el dropout porque tenia unos efectos raros,
# voy a presentar una imagen en el infrome sobre esto
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

history = model.fit(x_train_p, y_train,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(x_val_p, y_val),
    verbose=2)

loss, acc = model.evaluate(x_test_p, y_test)
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
plt.savefig('ej9_densa.pdf', format='pdf')
plt.show()
plt.close()

#######################
        #conv
#######################
x_train_p = np.reshape(x_train_p, (len(x_train_p), 28, 28, 1))
x_test_p = np.reshape(x_test_p, (len(x_test_p), 28, 28, 1))
x_val_p = np.reshape(x_val_p, (len(x_val_p), 28, 28, 1))

input_dim=x_train_p.shape[1]
lr=5e-4
rf=1e-2
batch_size=100
epochs=50

model_conv = keras.models.Sequential(name='Ejercicio_8_conv')
model_conv.add(keras.layers.Input(shape=x_train_p[0].shape))
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

history = model_conv.fit(x_train_p, y_train,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(x_val_p, y_val),
    verbose=2)

loss, acc = model_conv.evaluate(x_test_p, y_test)
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
plt.savefig('ej9_conv.pdf', format='pdf')
plt.show()
plt.close()
