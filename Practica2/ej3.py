"""
date: 26-09-2020
File: ej3.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10

def sigmoid(x):
    return 1/(1+np.exp(-x))

def foward(x, w1, w2, batch_size):
    y1=x.dot(w1)
    s1=sigmoid(y1)
    s1p=np.hstack([np.ones((batch_size,1)), s1])
    s2=s1p.dot(w2)
    return y1, s1, s1p, s2

def scores(x, w1, w2):
    s1=sigmoid(x.dot(w1))
    s1p=np.hstack([np.ones((x.shape[0],1)), s1])
    s2=s1p.dot(w2)
    return s2

def accuracy(s2, yb):
    y_pred=np.argmax(s2, axis=1)
    y_true=np.argmax(yb, axis=1)
    return (y_true==y_pred).mean()

def loss(w1, w2, s2, yb, rf):
    mse_i=np.sum((s2-yb)**2, axis=1)
    mse=np.mean(mse_i)
    return mse+0.5*rf*((w1**2).sum()+(w2**2).sum())

def grad(w1, w2, batch_size, xb, yb, y1, s1p, s2, rf):
    grad=2/batch_size*(s2-yb)
    dw2=s1p.T.dot(grad)+rf*w2
    grad=grad.dot(w2.T)
    grad=grad[:,1:]
    grad_sig=sigmoid(y1)*(1-sigmoid(y1))
    grad=grad*grad_sig
    dw1=xb.T.dot(grad)+rf*w1
    return dw1, dw2


def fit(x, y, x_test=None, y_test=None, n_neuronas=100, lr=1e-3, rf=1e-5, epochs=1, batch_size=100, seed=None, verbose=False):
    n_samples=y.shape[0]
    n_atribs=x[0,].shape[0]
    rnd=np.random.RandomState(seed)
    w1=1e-3*rnd.randn(n_atribs+1, n_neuronas)
    w1[0,:]=0
    w2=1e-3*rnd.randn(n_neuronas+1, n_clases)
    w2[0,:]=0
    niters=int(n_samples/batch_size)

    history={}
    history['loss']=np.zeros(epochs)
    history['acc']=np.zeros(epochs)

    if x_test is not None:
        history['val_acc']=np.zeros(epochs)

    for e in range(epochs):
        randomize = np.arange(n_samples)
        np.random.shuffle(randomize)
        x = x[randomize]
        y = y[randomize]
        log_loss=0
        log_acc=0
        for it in range(niters):
            id_batch=idx[it*batch_size: (it+1)*batch_size]
            xb=np.hstack([np.ones((batch_size,1)), x[id_batch]])
            yb=y[id_batch]
            y1, s1, s1p, s2=foward(xb, w1, w2, batch_size)
            log_loss+=loss(w1, w2, s2, yb,rf)
            log_acc+=accuracy(s2, yb)
            dw1, dw2=grad(w1, w2, batch_size, xb, yb, y1, s1p, s2, rf)
            w1-=lr*dw1
            w2-=lr*dw2
        history['loss'][e]=log_loss/niters
        history['acc'][e]=log_acc/niters
        if verbose:
            if x_test is not None:
                xt=np.hstack([np.ones((x_test.shape[0], 1)), x_test])
                s2=scores(xt, w1, w2)
                val_acc=accuracy(s2,y_test)
                history['val_acc'][e]=val_acc
                print('Epoca {:03d}: Precisión training: {:.3f}, Costo: {:.3f}, Precision test: {:.3f}'.format(e+1, history['acc'][e], history['loss'][e], history['val_acc'][e]))
            else:
                print('Epoca {:03d}: Precisión training: {:.3f}, Costo: {:.3f}'.format(e+1, history['acc'][e], history['loss'][e]))
    return history

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# parámetros
n_clases=np.max(y_train)+1
n_neuronas=100
idx=np.arange(y_train.shape[0])
idxt=np.arange(y_test.shape[0])
epochs=20
batch_size=100
lr=1e-3
rf=1e-1

# pre procesado
x_train=np.reshape(x_train, (x_train.shape[0], np.prod(x_train.shape[1:])))
x_train=(x_train-np.mean(x_train, axis=0))
x_test=np.reshape(x_test, (x_test.shape[0], np.prod(x_test.shape[1:])))
x_test=(x_test-np.mean(x_test, axis=0))

yy_train=np.zeros((y_train.shape[0], n_clases))
yy_train[idx, y_train[:,0]]=1

yy_test=np.zeros((y_test.shape[0], n_clases))
yy_test[idxt, y_test[:,0]]=1

history=fit(x_train, yy_train, x_test, yy_test, n_neuronas=n_neuronas, lr=lr, rf=rf, epochs=epochs, batch_size=batch_size, verbose=True)
