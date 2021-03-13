import numpy as np
from .layers import Input, Dense, ConcatInput

class Network():
    def __init__(self):
        self.layers=[]
    def add(self, layer):
        if len(self.layers)==0:
            self.layers.append(Input(in_dim=layer.in_dim, out_dim=layer.in_dim))
            self.layers.append(layer)
        elif isinstance(layer, Dense):
            self.layers.append(Dense(in_dim=self.layers[-1].out_dim, out_dim=layer.out_dim, act=layer.act, seed=layer.seed, reg=layer.reg))
        else:
            self.layers.append(ConcatInput(self.layers[0], self.layers[-1]))

    def fit(self, x, y, loss, opt, metric, x_test=None, y_test=None,  epochs=10, verbose=False):
        nsamples=x.shape[0]
        self.opt=opt
        self.loss=loss
        self.metric=metric
        lr_0=self.opt.lr

        history={}
        history['loss']=np.zeros(epochs)
        history['acc']=np.zeros(epochs)

        if x_test is not None:
            history['val_acc']=np.zeros(epochs)

        for e in range(epochs):
            self.opt(x, y, self)
            history['loss'][e]=self.loss(self.predict(x), y)
            history['acc'][e]=self.metric(self.predict(x), y)

            if verbose:
                if x_test is not None:
                    history['val_acc'][e]=self.metric(self.predict(x_test), y_test)
                    print('Epoca {:03d}: Precisión training: {:.3f}, Costo: {:.3f}, Precision test: {:.3f}'.format(e+1, history['acc'][e], history['loss'][e], history['val_acc'][e]))
                else:
                    print('Epoca {:03d}: Precisión training: {:.3f}, Costo: {:.3f}'.format(e+1, history['acc'][e], history['loss'][e]))
        return history

    def forward(self, x):
        s_i=x
        y=[0]
        s=[s_i]
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], Dense):
                y_i=self.layers[i].dot(s_i)
                s_i=self.layers[i](y_i)
                y.append(y_i)
                s.append(s_i)
            elif isinstance(self.layers[i], ConcatInput):
                s_i=self.layers[i](x, s_i)
                y.append(0)
                s.append(s_i)
        return y, s

    def scores(self, x):
        s_i=x
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], Dense):
                y_i=self.layers[i].dot(s_i)
                s_i=self.layers[i](y_i)
            elif isinstance(self.layers[i], ConcatInput):
                s_i=self.layers[i](x, s_i)
        return s_i

    def backward(self, x, y):
        y_i, s_i=self.forward(x)
        grad_i=self.loss.gradient(s_i[-1], y)
        for i in range(len(self.layers)-1, 0, -1):
            if isinstance(self.layers[i], Dense):
                #necesito estas dos cosas para el calculo de dw y del gradiente local respectivamente
                s_im1=s_i[i-1] #s sub i menos 1
                #con esos datos ya puedo calcular el resto ed las cosas
                grad_loc_i=self.layers[i].act.gradient(y_i[i]) #gradiente local segun al fncion de activacion de la layer
                grad_i*=grad_loc_i #gradiente de la funcion de activacion es el gradiente local por el gradiente glboal
                #tengo que agregarle una columna de nos al s_jm1
                s_im1=np.hstack((np.ones((s_im1.shape[0],1)), s_im1))
                dW_i=s_im1.T@grad_i #obtengo el gradiente de los pesos de esta capa
                self.layers[i].W=self.opt.update_weights(self.layers[i].W, self.layers[i].update_weights(dW_i))
                #el gradiente sigue para el proximo backward
                grad_i=(grad_i@self.layers[i].W.T)[:,1:]
            elif isinstance(self.layers[i], ConcatInput):
                #solamente le tengo que sacar la parte del input
                grad_i=grad_i[:,self.layers[i].in_input_dim:]

    def predict(self, x):
        return self.scores(x)
