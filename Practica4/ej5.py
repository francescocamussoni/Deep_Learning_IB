"""
date: 7-10-2020
File: ej5.py
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

#generacion datos
n = 100000
x = np.linspace(0, 1, n).reshape((n,1))
y = 4*x*(1-x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=int(0.1*n))
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=int(0.1*n))

#red neuronal
lr=1e-5
rf=0
batch_size=100
epochs=1000

inputs = keras.layers.Input(shape=(1,))
l = keras.layers.Dense(5,
    activation='tanh',
    kernel_regularizer=keras.regularizers.l2(rf))(inputs)
c = keras.layers.Concatenate()([inputs, l])
outputs = keras.layers.Dense(1,
    activation='linear',
    kernel_regularizer=keras.regularizers.l2(rf))(c)
model=keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(lr=lr),
    loss=keras.losses.MeanAbsoluteError(name="loss"),
    metrics=None)

model.summary()

history = model.fit(x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    verbose=2)

y_pred = model.predict(x_val)

fig=plt.figure(figsize=(9,6))
ax1 = plt.gca()
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1=plt.subplot()
#ax1.scatter(np.arange(epochs), history.history['loss'], color='red', alpha=0.5, marker='^')
ax1.plot(np.arange(epochs), history.history['loss'], color='red', label='Training loss')
#ax1.scatter(np.arange(epochs), history.history['val_loss'], color='blue', alpha=0.5, marker='*')
ax1.plot(np.arange(epochs), history.history['val_loss'], color='blue', label='Validation loss')
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Costo', fontsize=14)
ax1.hlines(min(history.history['loss']), 0, epochs, color='red', linestyles='dashdot', alpha=0.5)
ax1.hlines(min(history.history['val_loss']), 0, epochs, color='blue', linestyles='dashdot', alpha=0.5)
ax1.set_yscale('log')
plt.title('Resultados de entrenamiento. Lr='+str(lr)+', rf='+str(rf), fontsize=15)
ax1.legend(loc=7)
plt.savefig('ej5_losses.pdf', format='pdf')
plt.show()
plt.close()

fig=plt.figure(figsize=(9,6))
ax1 = plt.gca()
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1=plt.subplot()
ax1.scatter(x_val, y_val, label='Valor real')
ax1.scatter(x_val, y_pred, label='Predicción', marker='o', alpha=0.5, s=0.8)
ax1.set_xlabel('y', fontsize=14), ax1.set_ylabel('y', fontsize=14)
plt.title('Resultados de entrenamiento. Lr='+str(lr)+', rf='+str(rf), fontsize=15)
ax1.legend(loc='best')
plt.savefig('ej5_y.pdf', format='pdf')
plt.show()
