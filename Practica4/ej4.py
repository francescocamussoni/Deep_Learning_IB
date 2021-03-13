"""
date: 7-10-2020
File: ej4.py
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
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split

#cargo datos y extiendo mi training set (por defecto son 25mil y 25mil)
n_words=10000
review_length=500
emb_dim = 32
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=n_words)

# Preprocesamiento pad + division en val y test
x = np.hstack((x_train, x_test))
y = np.hstack((y_train, y_test))

x = keras.preprocessing.sequence.pad_sequences(x, maxlen=review_length)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=5000, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=5000, stratify=y_train)

##########################
        #embedings
##########################
#red neuronal
neurons=100
batch_size=50
epochs=50
lr=5e-4
rf=1

inputs = keras.Input(shape=x_train[0].shape)
embedding = keras.layers.Embedding(n_words,
    emb_dim,
    input_length=review_length)(inputs)
flatten = keras.layers.Flatten()(embedding)
l1 = keras.layers.Dense(100,
    activation="relu",
    kernel_regularizer=keras.regularizers.l2(rf))(flatten)
do1 = keras.layers.Dropout(rate=0.5)(l1)
l2 = keras.layers.Dense(10,
    activation="relu",
    kernel_regularizer=keras.regularizers.l2(rf))(do1)
do2 = keras.layers.Dropout(rate=0.5)(l2)
concatenated = keras.layers.Concatenate()([flatten, do2])
output = keras.layers.Dense(1,
    activation="linear",
    kernel_regularizer=keras.regularizers.l2(rf))(concatenated)
model=keras.Model(inputs=inputs, outputs=output)
model.compile(optimizer=keras.optimizers.Adam(lr=lr),
    loss=keras.losses.BinaryCrossentropy(name="loss", from_logits=True),
    metrics=["acc"])

model.summary()

history = model.fit(x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    verbose=2)

loss, acc = model.evaluate(x_test, y_test)
print('El accuracy sobre los datos de validacion fue= ', acc)

#plot
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
ax2.legend(loc=10)
plt.savefig('ej4_2.pdf', format='pdf')
plt.show()
plt.close()

##########################
        #conv
##########################
#red neuronal
batch_size=100
epochs=50
lr=1e-4
rf=10

inputs = keras.layers.Input(shape=x_train[0].shape)
embedding = keras.layers.Embedding(n_words,
    emb_dim,
    input_length=review_length)(inputs)
bn1 = keras.layers.BatchNormalization()(embedding)
l1 = keras.layers.Conv1D(filters=16,
    kernel_size=5,
    padding='same',
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(rf)
    )(bn1)
mp1 = keras.layers.MaxPooling1D()(l1)
bn2 = keras.layers.BatchNormalization()(mp1)
l2 = keras.layers.Conv1D(filters=32,
    kernel_size=5,
    padding='same',
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(rf)
    )(bn2)
mp2 = keras.layers.MaxPooling1D()(l2)
bn3 = keras.layers.BatchNormalization()(mp2)
do2 = keras.layers.Dropout(rate=0.5)(bn3)
ft = keras.layers.Flatten()(do2)
output = keras.layers.Dense(1,
    activation="linear",
    kernel_regularizer=keras.regularizers.l2(rf))(ft)
model=keras.Model(inputs=inputs, outputs=output)
model.compile(optimizer=keras.optimizers.Adam(lr=lr),
    loss=keras.losses.BinaryCrossentropy(name="loss", from_logits=True),
    metrics=["acc"])

model.summary()

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs,  batch_size=batch_size, verbose=2)

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
ax2.legend(loc='center left')
plt.savefig('ej4_2_conv_2.pdf', format='pdf')
plt.show()
plt.close()
