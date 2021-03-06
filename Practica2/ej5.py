"""
date: 26-09-2020
File: ej5.py
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

def relu(x):
    return np.where(x<0, 0, x)

def foward(x, w1, w2, batch_size):
    y1=x.dot(w1)
    s1=relu(y1)
    s1p=np.hstack([np.ones((batch_size,1)), s1])
    s2=s1p.dot(w2)
    s3=sigmoid(s2)
    return y1, s1, s1p, s2, s3

def scores(x, w1, w2):
    s1=relu(x.dot(w1))
    s1p=np.hstack([np.ones((x.shape[0],1)), s1])
    s2=s1p.dot(w2)
    s3=sigmoid(s2)
    return s3

def accuracy(s2, yb):
    y_pred=np.argmax(s2, axis=1)
    y_true=np.argmax(yb, axis=1)
    return (y_true==y_pred).mean()

def loss_mse(w1, w2, s3, yb, rf):
    mse_i=np.sum((s3-yb)**2, axis=1)
    mse=np.mean(mse_i)
    return mse+0.5*rf*((w1**2).sum()+(w2**2).sum())

def grad_mse(w1, w2, batch_size, xb, yb, y1, s1p, s2, s3, rf):
    grad=2/batch_size*(s3-yb)
    grad_sig=sigmoid(s2)*(1-sigmoid(s2))
    grad=grad*grad_sig
    dw2=s1p.T.dot(grad)+rf*w2
    grad=grad.dot(w2.T)
    grad=grad[:,1:]
    grad_relu=np.where(y1<0, 0, 1)
    grad=grad*grad_relu
    dw1=xb.T.dot(grad)+rf*w1
    return dw1, dw2

def loss_softmax(w1, w2, s3, yb, rf):
    maxval=-np.amax(s3, axis=1)
    s3_exp=np.exp(s3+maxval[:,np.newaxis])
    s3_exp_y=s3_exp[np.arange(batch_size), np.argmax(yb, axis=1)]
    sum_s3_exp=np.sum(s3_exp, axis=1)
    cce=-np.log(s3_exp_y/sum_s3_exp[:,np.newaxis])
    return cce.mean()+0.5*rf*((w1**2).sum()+(w2**2).sum()), s3_exp, sum_s3_exp

def grad_softmax(w1, w2, batch_size, xb, yb, y1, s1p, s2, s3, s3_exp, sum_s3_exp, rf):
    grad=s3_exp/sum_s3_exp[:,np.newaxis]
    grad[np.arange(batch_size), np.argmax(yb, axis=1)]-=1
    grad_sig=sigmoid(s2)*(1-sigmoid(s2))
    grad=grad*grad_sig
    dw2=s1p.T.dot(grad)+rf*w2
    grad=grad.dot(w2.T)
    grad=grad[:,1:]
    grad_relu=np.where(y1<0, 0, 1)
    grad=grad*grad_relu
    dw1=xb.T.dot(grad)+rf*w1
    return dw1, dw2

def fit(x, y, x_test=None, y_test=None, n_neuronas=100, lr=1e-3, rf=1e-5, epochs=1, batch_size=100, metric='mse', seed=None, verbose=False):
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
            y1, s1, s1p, s2, s3=foward(xb, w1, w2, batch_size)
            log_acc+=accuracy(s2, yb)
            if metric=='mse':
                log_loss+=loss_mse(w1, w2, s3, yb, rf)
                dw1, dw2=grad_mse(w1, w2, batch_size, xb, yb, y1, s1p, s2, s3, rf)
            elif metric=='softmax':
                loss_aux, s3_exp, sum_s3_exp=loss_softmax(w1, w2, s3, yb, rf)
                log_loss+=loss_aux
                dw1, dw2=grad_softmax(w1, w2, batch_size, xb, yb, y1, s1p, s2, s3, s3_exp, sum_s3_exp, rf)
            else:
                print('Metrica no soportada')
                return
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
lr=5e-4
rf=1e-4

# pre procesado
x_train=np.reshape(x_train, (x_train.shape[0], np.prod(x_train.shape[1:])))
x_train=(x_train-np.mean(x_train, axis=0))
x_test=np.reshape(x_test, (x_test.shape[0], np.prod(x_test.shape[1:])))
x_test=(x_test-np.mean(x_test, axis=0))

yy_train=np.zeros((y_train.shape[0], n_clases))
yy_train[idx, y_train[:,0]]=1

yy_test=np.zeros((y_test.shape[0], n_clases))
yy_test[idxt, y_test[:,0]]=1

history=fit(x_train, yy_train, x_test, yy_test, metric='softmax', n_neuronas=n_neuronas, lr=lr, rf=rf, epochs=epochs, batch_size=batch_size, verbose=True)
