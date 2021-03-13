import numpy as np

class Loss():
    def __call__(self):
        pass
    def gradient(self):
        pass

class MSE(Loss): #s=scores yt=y_true
    def __call__(self, s, yt):
        return (np.sum((s-yt)**2))/yt.shape[0]
    def gradient(self, s, yt):
        return 2*(s-yt)/yt.shape[0]

class CCE(Loss):#s=scores yt=y_true #despues me di cuenta que si utilizaba la forma vectorizada servía tanto para imagenes como para xor
    #es decir, para clasificado binario o no, pero así corrí todos los ejercicios (el 8 lo hice con MSE).
    def __call__(self, s, yt):
        s=1/(1+np.exp(-s))
        return -np.sum(yt*np.log(s)+(1-yt)*np.log(1-s))/yt.shape[0]
    def gradient(self, s, yt):
        return (1/(1+np.exp(-s))-yt)/yt.shape[0]
