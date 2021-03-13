import numpy as np

class BaseLayer():
    def __init__(self):
        pass
    def get_output_shape(self):
        pass

class Input(BaseLayer):
    def __init__(self, in_dim, out_dim):
        self.in_dim=in_dim
        self.out_dim=out_dim
    def get_output_shape(self):
        return self.out_dim
        return

class ConcatInput(BaseLayer): #voy a tomar de convencion que la primer layer siempre es la que viene de antes (la que llamamos S)
    def __init__(self, layer1=None, layer2=None):
        if layer1!=None: #esto es por implementacion, es para que sea mas facil utilizar desde el main, cn esto no hace falta aclarar tantas cosas a la hora de crear la red
            self.in_input_dim=layer1.get_output_shape()
            self.in_previous_dim=layer2.get_output_shape()
            self.out_dim=self.in_input_dim+self.in_previous_dim
    def __call__(self, x, s):
        return np.hstack((x, s))
    def get_output_shape(self):
        return self.out_dim

class WLayer(BaseLayer):
    def __init__(self, in_dim, out_dim, act, reg):
        self.act=act
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.reg=reg
    def get_input_shape(self):
        return self.in_dim
    def get_output_shape(self):
        return self.out_dim
    def get_weights(self):
        return self.W
    def update_weights(self, dw_j):
        return dw_j+self.reg.gradient(self.W)

class Dense(WLayer):
    def __init__(self, out_dim, act, reg, weight_multiplier=1, seed=42, in_dim=None):
        super().__init__(in_dim, out_dim, act, reg)
        self.seed=seed
        if in_dim!=None: #esto es por implementacion, es para que sea mas facil utilizar desde el main, cn esto no hace falta aclarar tantas cosas a la hora de crear la red
            self.rnd = np.random.RandomState(seed)
            self.W=weight_multiplier*self.rnd.randn(self.in_dim + 1, self.out_dim)*2/np.sqrt(self.in_dim+1+self.out_dim)
            self.W[0,:]=0 #inicializo el bias en cero
    def __call__(self, x):
        return self.act(x)
    def dot(self, x):
        xp=np.hstack((np.ones((x.shape[0], 1)), x))
        return xp@self.W
