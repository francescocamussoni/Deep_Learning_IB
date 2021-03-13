"""
date: 7-10-2020
File: ej1.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from tensorflow import keras

x, y = load_boston(return_X_y=True)

def split_train_test(housing, test_ratio, validation_ratio=0):
    print(housing.shape)
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(housing))
    test_set_size = int(len(housing)*test_ratio)
    validation_set_size=int(len(housing)*validation_ratio)
    test_indices = shuffled_indices[:test_set_size]
    validation_indices = shuffled_indices[test_set_size:validation_set_size+test_set_size]
    train_indices = shuffled_indices[test_set_size+validation_set_size:]
    if validation_ratio==0:
        return housing[train_indices], housing[test_indices]
    else:
        return housing[train_indices], housing[test_indices], housing[validation_indices]

test_ratio=0.25
boston_training, boston_test = split_train_test(np.hstack((x, y[:,np.newaxis])), test_ratio)

x_train=boston_training[:,:-1]
y_train=boston_training[:,-1]
x_test=boston_test[:,:-1]
y_test=boston_test[:,-1]

#Preprocesado
media=x_train.mean(axis=0)
std=x_train.std(axis=0)
#norm=np.max(np.abs(x_train), axis=0)
x_train=(x_train-media)/std
x_test=(x_test-media)/std

dim_input=x.shape[1]

rf=1e-4
lr=1e-3
model = keras.models.Sequential(name='RLinear')
model.add(keras.layers.Dense(1, input_shape=(dim_input,),
    kernel_regularizer=keras.regularizers.l2(rf)))
optimizer=keras.optimizers.SGD(learning_rate=lr)
model.compile(optimizer, loss=keras.losses.MSE, metrics=['mse'])
model.summary()

#Entrenamiento
epochs=200

history=model.fit(x_train,
    y_train,
    epochs=epochs,
    validation_data=(x_test, y_test),
    verbose=2)

y_pred = model.predict(x_test)

#plot
fig = plt.figure(figsize=(16,6))
plt.plot(range(epochs), history.history['val_loss'], color='blue')
plt.title('Loss para los datos de validacion. lr='+str(lr)+', rf='+str(rf), fontsize=16)
plt.xlabel('Epocas', fontsize=14)
plt.ylabel('Loss', fontsize=14)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 13)
plt.hlines(np.min(history.history['val_loss']), 0, epochs, linestyle='dashdot')
plt.savefig('Ej1.pdf', format='pdf')
plt.show()

fig = plt.figure(figsize=(16,6))
plt.plot(y_test, y_test, label='Valores reales')
plt.title('Resultados de predicción sobre los datos de validación', fontsize=16)
plt.scatter(y_test, y_pred, color='red', label='Predicciones')
plt.xlabel('Precio', fontsize=14)
plt.ylabel('Precio', fontsize=14)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 13)
plt.savefig('Ej1_2.pdf', format='pdf')
plt.show()
