"""
date: 7-10-2020
File: ej2.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""


############################################################
                    #Ejercicio 2-3
############################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# parámetros
dim_input=np.prod(x_train[0].shape)
epochs=60
batch_size=100
lr=1e-2
rf=1e-4

# pre procesado
x_train=np.reshape(x_train, (x_train.shape[0], np.prod(x_train.shape[1:])))
x_test=np.reshape(x_test, (x_test.shape[0], np.prod(x_test.shape[1:])))

media=x_train.mean(axis=0)
std=x_train.std(axis=0)

x_train=(x_train-media)/std
x_test=(x_test-media)/std

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


# red neuronal
model = keras.models.Sequential(name='ej2_tp4_ej4')

model.add(keras.layers.Dense(100,
        activation='sigmoid',
        input_shape=(dim_input,),
        kernel_regularizer=keras.regularizers.l2(rf)))
model.add(keras.layers.Dense(10,
        kernel_regularizer=keras.regularizers.l2(rf)))

optimizer=keras.optimizers.SGD(learning_rate=lr)
model.compile(optimizer, loss=keras.losses.MSE, metrics=['acc', 'mse'])

model.summary()

history=model.fit(x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=batch_size,
    epochs=epochs,
    verbose=2)

#plot
fig=plt.figure(figsize=(9,6))
ax1 = plt.gca()
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1=plt.subplot()
ax1.scatter(np.arange(epochs), history.history['loss'], color='red', alpha=0.5, marker='^')
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Costo', color='red', fontsize=14)
ax1.plot(np.arange(epochs), history.history['loss'], color='red')
ax1.hlines(min(history.history['loss']), 0, epochs, color='red', linestyles='dashdot')
ax1.set_yscale('log')
ax2 = plt.gca()
ax2.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax2=ax1.twinx()
ax2.scatter(np.arange(epochs), history.history['acc'], alpha=0.5)
ax2.plot(np.arange(epochs), history.history['acc'], color='blue')
ax2.set_xlabel('Época', fontsize=14), ax2.set_ylabel('Precisión', color='blue', fontsize=14)
ax2.hlines(max(history.history['acc']), 0, epochs, color='blue', linestyles='dashdot')
plt.title('Resultados para los datos de training. Lr='+str(lr)+', rf='+str(rf), fontsize=15)
plt.legend()
plt.savefig('ej2_ej3_tp4_training.pdf', format='pdf')
plt.show()
plt.close()

fig=plt.figure(figsize=(9,6))
ax1 = plt.gca()
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1=plt.subplot()
ax1.scatter(np.arange(epochs), history.history['val_loss'], color='red', alpha=0.5, marker='^')
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Costo', color='red', fontsize=14)
ax1.plot(np.arange(epochs), history.history['val_loss'], color='red')
ax1.hlines(min(history.history['val_loss']), 0, epochs, color='red', linestyles='dashdot')
ax1.set_yscale('log')
ax2 = plt.gca()
ax2.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax2=ax1.twinx()
ax2.scatter(np.arange(epochs), history.history['val_acc'], alpha=0.5)
ax2.plot(np.arange(epochs), history.history['val_acc'], color='blue')
ax2.set_xlabel('Época', fontsize=14), ax2.set_ylabel('Precisión', color='blue', fontsize=14)
ax2.hlines(max(history.history['val_acc']), 0, epochs, color='black', linestyles='dashdot')
plt.title('Resultados para los datos de test. Lr='+str(lr)+', rf='+str(rf), fontsize=15)
plt.legend()
plt.savefig('ej2_ej3_tp4_test.pdf', format='pdf')
plt.show()

############################################################
                    #Ejercicio 2-4
############################################################
# red neuronal
dim_input=np.prod(x_train[0].shape)
epochs=60
batch_size=100
lr=1e-3
rf=1e-5

model = keras.models.Sequential(name='ej2_tp4_ej4')

model.add(keras.layers.Dense(100,
        activation='sigmoid',
        input_shape=(dim_input,),
        kernel_regularizer=keras.regularizers.l2(rf)))
model.add(keras.layers.Dense(10,
        kernel_regularizer=keras.regularizers.l2(rf)))

optimizer=keras.optimizers.SGD(learning_rate=lr)
model.compile(optimizer, loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['acc'])

history=model.fit(x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=batch_size,
    epochs=epochs,
    verbose=2)

#plot
fig=plt.figure(figsize=(9,6))
ax1 = plt.gca()
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1=plt.subplot()
ax1.scatter(np.arange(epochs), history.history['loss'], color='red', alpha=0.5, marker='^')
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Costo', color='red', fontsize=14)
ax1.plot(np.arange(epochs), history.history['loss'], color='red')
ax1.hlines(min(history.history['loss']), 0, epochs, color='red', linestyles='dashdot')
ax1.set_yscale('log')
ax2 = plt.gca()
ax2.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax2=ax1.twinx()
ax2.scatter(np.arange(epochs), history.history['acc'], alpha=0.5)
ax2.plot(np.arange(epochs), history.history['acc'], color='blue')
ax2.set_xlabel('Época', fontsize=14), ax2.set_ylabel('Precisión', color='blue', fontsize=14)
ax2.hlines(max(history.history['acc']), 0, epochs, color='blue', linestyles='dashdot')
plt.title('Resultados para los datos de training. Lr='+str(lr)+', rf='+str(rf), fontsize=15)
plt.legend()
plt.savefig('ej2_ej4_tp4_training.pdf', format='pdf')
plt.show()
plt.close()

fig=plt.figure(figsize=(9,6))
ax1 = plt.gca()
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1=plt.subplot()
ax1.scatter(np.arange(epochs), history.history['val_loss'], color='red', alpha=0.5, marker='^')
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Costo', color='red', fontsize=14)
ax1.plot(np.arange(epochs), history.history['val_loss'], color='red')
ax1.hlines(min(history.history['val_loss']), 0, epochs, color='red', linestyles='dashdot')
ax1.set_yscale('log')
ax2 = plt.gca()
ax2.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax2=ax1.twinx()
ax2.scatter(np.arange(epochs), history.history['val_acc'], alpha=0.5)
ax2.plot(np.arange(epochs), history.history['val_acc'], color='blue')
ax2.set_xlabel('Época', fontsize=14), ax2.set_ylabel('Precisión', color='blue', fontsize=14)
ax2.hlines(max(history.history['val_acc']), 0, epochs, color='black', linestyles='dashdot')
plt.title('Resultados para los datos de test. Lr='+str(lr)+', rf='+str(rf), fontsize=15)
plt.legend()
plt.savefig('ej2_ej4_tp4_test.pdf', format='pdf')
plt.show()

############################################################
                    #Ejercicio 2-6
############################################################
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Dataset
x_train = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
y_train = np.array([[1],[0],[0],[1]]) #los defino así para que sea un problema de clasificación binaria
dim_input=x_train.shape[1]

epochs=1000
lr=1e-1

###############################
            #a
###############################
# red neuronal
inputs = keras.layers.Input(shape=(dim_input,))
l1 = keras.layers.Dense(2, activation='tanh')(inputs)
outputs = keras.layers.Dense(1, activation='tanh')(l1)
model=keras.Model(inputs=inputs, outputs=outputs)

optimizer=keras.optimizers.SGD(learning_rate=lr)
model.compile(optimizer, loss=keras.losses.MSE, metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.9)])

model.summary()

history=model.fit(x_train, y_train,
    epochs=epochs,
    verbose=2)

#plot
fig=plt.figure(figsize=(9,6))
ax1 = plt.gca()
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1=plt.subplot()
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Costo', color='red', fontsize=14)
ax1.plot(np.arange(epochs), history.history['loss'], color='red')
ax1.hlines(min(history.history['loss']), 0, epochs, color='red', linestyles='dashdot')
ax1.set_yscale('log')
ax2 = plt.gca()
ax2.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax2=ax1.twinx()
ax2.plot(np.arange(epochs), history.history['binary_accuracy'], color='blue')
ax2.set_xlabel('Época', fontsize=14), ax2.set_ylabel('Precisión', color='blue', fontsize=14)
ax2.hlines(max(history.history['binary_accuracy']), 0, epochs, color='blue', linestyles='dashdot')
plt.title('Resultados para los datos de training. Lr='+str(lr)+', rf='+str(rf), fontsize=15)
plt.legend()
plt.savefig('ej2_ej6_tp4_training.pdf', format='pdf')
plt.show()
plt.close()

###############################
            #b
###############################

# red neuronal
inputs = keras.layers.Input(shape=(dim_input,))
l1 = keras.layers.Dense(1, activation='tanh')(inputs)
concatenated=keras.layers.Concatenate()([inputs, l1])
outputs = keras.layers.Dense(1, activation='tanh')(concatenated)
model=keras.Model(inputs=inputs, outputs=outputs)

optimizer=keras.optimizers.SGD(learning_rate=lr)
model.compile(optimizer,
    loss=keras.losses.MSE,
    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.9)])

history=model.fit(x_train, y_train,
    epochs=epochs,
    verbose=2)

#plot
#plot
fig=plt.figure(figsize=(9,6))
ax1 = plt.gca()
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1=plt.subplot()
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Costo', color='red', fontsize=14)
ax1.plot(np.arange(epochs), history.history['loss'], color='red')
ax1.hlines(min(history.history['loss']), 0, epochs, color='red', linestyles='dashdot')
ax1.set_yscale('log')
ax2 = plt.gca()
ax2.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax2=ax1.twinx()
ax2.plot(np.arange(epochs), history.history['binary_accuracy'], color='blue')
ax2.set_xlabel('Época', fontsize=14), ax2.set_ylabel('Precisión', color='blue', fontsize=14)
ax2.hlines(max(history.history['binary_accuracy']), 0, epochs, color='blue', linestyles='dashdot')
plt.title('Resultados para los datos de training. Lr='+str(lr)+', rf='+str(rf), fontsize=15)
plt.legend()
plt.savefig('ej2_ej6_b_tp4_training.pdf', format='pdf')
plt.show()
plt.close()
