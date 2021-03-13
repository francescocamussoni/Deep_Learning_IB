import numpy as np

class Optimizer(): #lr=learning rate, bs=batch_size
    def __init__(self, lr):
        self.lr=lr
    def __call__(self):
        pass
    def update_weights(self):
        pass

class SGD(Optimizer):
    def __init__(self, lr=1e-3, bs=0):
        super().__init__(lr)
        self.bs=bs
    def __call__(self, x, y, model):
        idx=np.arange(x.shape[0])
        np.random.shuffle(idx)
        if not self.bs:
            model.backward(x[idx], y[idx])
        else:
            batches=np.array_split(idx, idx.shape[0]/self.bs)
            for batch in batches:
                model.backward(x[batch], y[batch])
    def update_weights(self, W, dW):
        W-=self.lr*dW
        return W
