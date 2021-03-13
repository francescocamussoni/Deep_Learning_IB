import numpy as np

class Regularizer():
    def __init__(self, rf):
        self.rf=rf
    def __call__(self):
        pass
    def gradient(self):
        pass

class L2(Regularizer):
    def __init__(self, rf=1e-3):
        super().__init__(rf)
    def __call__(self, W):
        return 0.5*self.rf*np.sum(W**2)
    def gradient(self, W):
        return self.rf*W

class L1(Regularizer):
    def __init__(self, rf=1e-3):
        super().__init__(rf)
    def __call__(self, W):
        return self.rf*np.sum(np.absolute(W))
    def gradient(self, W):
        return self.rf*np.sign(W)
