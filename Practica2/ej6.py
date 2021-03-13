"""
date: 26-09-2020
File: ej6.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""

import itertools
from NeuralNetwork import models, activations, losses, metrics, layers, optimizers, regularizers
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap


# Dataset
x_train = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
y_train = np.array([[1],[0],[0],[1]]) #los defino así para que sea un problema de clasificación binaria

epochs=1000
lr=1e-1
bs=4
rf=0
rf2=0

# Regularizer
reg1 = regularizers.L2(rf)
reg2 = regularizers.L1(rf2)

# Create model
model = models.Network()
model.add(layers.Dense(out_dim=2, act=activations.Tanh(), in_dim=x_train.shape[1], seed=1, reg=reg1))
model.add(layers.Dense(out_dim=1, act=activations.Tanh(), reg=reg2, seed=1))

# Train network
history=model.fit(x=x_train, y=y_train, opt=optimizers.SGD(lr=lr, bs=bs), loss=losses.CCE(), metric=metrics.Acc_xor, x_test=None, y_test=None, epochs=epochs, verbose=False)


if 'val_acc' in history.keys():
    fig=plt.figure(figsize=(8,6))
    plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
    #plt.scatter(np.arange(epochs), history['val_acc'])
    plt.plot(np.arange(epochs), history['val_acc'], color='red')
    plt.xlabel('Época', fontsize=14), plt.ylabel('Precisión', fontsize=14)
    plt.hlines(max(history['val_acc']), 0, epochs, color='black', linestyles='dashdot')
    fig.suptitle(r'Precisión para CIFAR: CCE y función relu+sigmoide. $\alpha$='+str(lr)+'. $\lambda$='+str(rf), fontsize=16)
    plt.savefig('test6_test.pdf', format='pdf')


#esto fue para plotear el training
fig=plt.figure(figsize=(9,6))

ax1 =plt.gca()
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1=plt.subplot()
#ax1.scatter(np.arange(epochs), history['loss'])
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Costo', color='red', fontsize=14)
ax1.plot(np.arange(epochs), history['loss'], color='red')
#ax1.set_yscale('log')


ax2 = plt.gca()
ax2.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax2=ax1.twinx()
#ax2.scatter(np.arange(epochs), history['acc'])
ax2.plot(np.arange(epochs), history['acc'], color='blue')
ax2.set_xlabel('Época', fontsize=14), ax2.set_ylabel('Precisión', color='blue', fontsize=14)
ax2.hlines(max(history['acc']), 0, epochs, color='black', linestyles='dashdot')

plt.legend()
fig.suptitle(r'Red neuronal de 2 capasde neuronas, 1 densa y una de concatenación. Activacion tgh(). Costo CCE. $\alpha$='+str(lr)+r', $\lambda_1$='+str(rf)+r' y $\lambda_2$='+str(rf2), wrap=True, fontsize=14)
plt.savefig('ej7_cce_concat.pdf', format='pdf')
plt.show()
