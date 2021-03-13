import numpy as np

class Activation():
    def __call__(self):
        pass
    def gradient(self):
        pass

class ReLU(Activation):
    def __call__(self, x):
        return np.where(x<0, 0, x)
    def gradient(self, x):
        return np.where(x<0, 0, 1)

class Tanh(Activation):
    def __call__(self, x):
        return np.tanh(x)
    def gradient(self, x):
        return 1-np.tanh(x)**2

class Sigmoid(Activation):
    def __call__(self, x):
        return 1/(1-np.exp(-x))
    def gradient(self, x):
        return (1/(1-np.exp(-x)))*(1-1/(1-np.exp(-x)))

class Linear(Activation):
    def __call__(self, x):
        return x
    def gradient(self, x):
        return x/x
