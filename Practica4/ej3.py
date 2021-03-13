"""
date: 7-10-2020
File: ej3.py
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
input_dim=10000
(x_train, y_train), (x_test, y_test)=imdb.load_data(num_words=input_dim)
x, y = np.hstack((x_train, x_test)), np.hstack((y_train, y_test))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=5000, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=5000, stratify=y_train)


#necesito tener todos los vectores con la misma longitud.
#solucion de no me acuerdo que libro
def vectorize_sequences(sequences, dimension=10000):
    results=np.zeros((len(sequences), dimension))
    for i, sequences in enumerate(sequences):
        ##El libro propone esto
        #results[i, sequences]=1
        ##Cabre me propuso modificarlo un poco, en vez de hacer un one hot encoder
        ##poner la cantidad de veces que se repite en lugar de un 1
        ##particularmente pienso que deberia andar mejor por sentido común
        ##que tipazo Cabre
        values, counts = np.unique(sequences, return_counts=True)
        results[i, values] = counts
    return results

x_train=vectorize_sequences(x_train, input_dim)
x_test=vectorize_sequences(x_test, input_dim)
x_val=vectorize_sequences(x_val, input_dim)

############################
        #reg_normal
############################
#red neuronal
neurons=100
batch_size=100
epochs=50
lr=1e-4
rf=1e-2

inputs = keras.layers.Input(shape=(input_dim,))
l1 = keras.layers.Dense(neurons,
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(rf))(inputs)
l2 = keras.layers.Dense(neurons/10,
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(rf))(l1)
concatenated=keras.layers.Concatenate()([inputs, l2])
outputs = keras.layers.Dense(1,
    activation='linear',
    kernel_regularizer=keras.regularizers.l2(rf))(concatenated)
model=keras.Model(inputs=inputs, outputs=outputs)

optimizer=keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer,
    loss=keras.losses.BinaryCrossentropy(name='loss', from_logits=True),
    metrics=[tf.keras.metrics.BinaryAccuracy(name='acc', threshold=0.5)])

model.summary()

history=model.fit(x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
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
plt.savefig('ej3_reg_normal.pdf', format='pdf')
plt.show()
plt.close()

############################
        #batch_norm
############################
#red neuronal
neurons=100
batch_size=100
epochs=50
lr=1e-4
rf=1

inputs = keras.layers.Input(shape=(input_dim,))
l1 = keras.layers.Dense(neurons,
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(rf)
    )(inputs)
bn1 = keras.layers.BatchNormalization()(l1)
l2 = keras.layers.Dense(neurons/10,
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(rf)
    )(bn1)
bn2 = keras.layers.BatchNormalization()(l2)
concatenated=keras.layers.Concatenate()([inputs, bn2])
outputs = keras.layers.Dense(1,
    activation='linear',
    kernel_regularizer=keras.regularizers.l2(rf))(concatenated)
model=keras.Model(inputs=inputs, outputs=outputs)

optimizer=keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer,
    loss=keras.losses.BinaryCrossentropy(name='loss', from_logits=True),
    metrics=[tf.keras.metrics.BinaryAccuracy(name='acc', threshold=0.5)])

model.summary()

history=model.fit(x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=batch_size,
    epochs=epochs,
    verbose=2)

loss, acc = model.evaluate(x_val, y_val)
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
plt.savefig('ej3_reg_bn.pdf', format='pdf')
plt.show()
plt.close()

############################
        #drop_out
############################
#red neuronal
neurons=100
batch_size=100
epochs=50
lr=1e-4
rf=1e-2

inputs = keras.layers.Input(shape=(input_dim,))
l1 = keras.layers.Dense(neurons,
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(rf)
    )(inputs)
do1 = keras.layers.Dropout(rate=0.5)(l1)
l2 = keras.layers.Dense(neurons/10,
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(rf)
    )(do1)
do2 = keras.layers.Dropout(rate=0.5)(l2)
concatenated=keras.layers.Concatenate()([inputs, do2])
outputs = keras.layers.Dense(1,
    activation='linear',
    kernel_regularizer=keras.regularizers.l2(rf))(concatenated)
model=keras.Model(inputs=inputs, outputs=outputs)

optimizer=keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer,
    loss=keras.losses.BinaryCrossentropy(name='loss', from_logits=True),
    metrics=[tf.keras.metrics.BinaryAccuracy(name='acc', threshold=0.5)])

model.summary()

history=model.fit(x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=batch_size,
    epochs=epochs,
    verbose=2)

loss, acc = model.evaluate(x_val, y_val)
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
plt.savefig('ej3_reg_do.pdf', format='pdf')
plt.show()
plt.close()
