"""
date: 26-09-2020
File: ej8.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""

from NeuralNetwork import models, activations, losses, metrics, layers, optimizers, regularizers
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
n_clases=np.max(y_train)+1

x_train=np.reshape(x_train, (x_train.shape[0], np.prod(x_train.shape[1:])))
x_test=np.reshape(x_test, (x_test.shape[0], np.prod(x_test.shape[1:])))
maximum = x_train.max()
media = x_train.mean(axis=0)
x_train = (x_train - media)/maximum
x_test = (x_test - media)/maximum


idx=np.arange(y_train.shape[0])
idxt=np.arange(y_test.shape[0])

yy_train=np.zeros((y_train.shape[0], n_clases))
yy_train[idx, y_train[:,0]]=1

yy_test=np.zeros((y_test.shape[0], n_clases))
yy_test[idxt, y_test[:,0]]=1

epochs=30
lr=1e-2
bs=100
rf=5e-3

# Regularizer
reg = regularizers.L2(rf)

# Create model
model = models.Network()
model.add(layers.Dense(in_dim=3072, act=activations.ReLU(), out_dim=100, reg=reg, weight_multiplier=1))
model.add(layers.Dense(act=activations.ReLU(), out_dim=100, reg=reg, weight_multiplier=1))
model.add(layers.Dense(act=activations.Tanh(), out_dim=10, reg=reg, weight_multiplier=1))

# Train network
history=model.fit(x=x_train, y=yy_train, opt=optimizers.SGD(lr=lr, bs=bs), loss=losses.MSE(), metric=metrics.Acc_img, x_test=x_test, y_test=yy_test, epochs=epochs, verbose=True)

if 'val_acc' in history.keys():
    fig=plt.figure(figsize=(8,6))
    plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
    #plt.scatter(np.arange(epochs), history['val_acc'])
    plt.plot(np.arange(epochs), history['val_acc'], color='red')
    plt.xlabel('Época', fontsize=14), plt.ylabel('Precisión', fontsize=14)
    plt.hlines(max(history['val_acc']), 0, epochs, color='black', linestyles='dashdot')
    fig.suptitle(r'Precisión para CIFAR: CCE y función relu+sigmoide. $\alpha$='+str(lr)+'. $\lambda$='+str(rf), fontsize=16)
    plt.savefig('ej8_test_relu.pdf', format='pdf')
    plt.show()

#esto fue para plotear el training
fig=plt.figure(figsize=(9,6))

ax1 =plt.gca()
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1=plt.subplot()
#ax1.scatter(np.arange(epochs), history['loss'])
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Costo', color='red', fontsize=14)
ax1.plot(np.arange(epochs), history['loss'], color='red')
ax1.set_yscale('log')


ax2 = plt.gca()
ax2.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax2=ax1.twinx()
#ax2.scatter(np.arange(epochs), history['acc'])
ax2.plot(np.arange(epochs), history['acc'], color='blue')
ax2.set_xlabel('Época', fontsize=14), ax2.set_ylabel('Precisión', color='blue', fontsize=14)
ax2.hlines(max(history['acc']), 0, epochs, color='black', linestyles='dashdot')

plt.legend()
fig.suptitle(r'Red neuronal de 3 capas densas de neuronas 100, 100 y 10 neuronas). Activacion tgh(). Costo MSE. $\alpha$='+str(lr)+r' y $\lambda$='+str(rf), wrap=True, fontsize=14)
plt.savefig('ej8_relu.pdf', format='pdf')
plt.show()
