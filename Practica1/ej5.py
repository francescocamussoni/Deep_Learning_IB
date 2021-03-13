"""
date: 30-08-2020
File: ej5.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from keras.datasets import cifar10
from keras.datasets import mnist

class LinearClassifier():
    def __init__(self, alfa, lbda):
        self.alfa=alfa
        self.lbda=lbda
        self.accuracy=[]
        self.costo=[]
    def fit(self, x, y, x_test, y_test, n_epocs, n_batch_size):
        n_categories=np.amax(y_train)+1
        self.W=np.random.uniform(0, 1, (n_categories, np.prod(x_train.shape[1:])))
        print('Epoca 0 completada')
        self.costo.append(np.sum([self.L_i(x[i], y[i]) for i in range(len(x))])/len(x))
        self.accuracy.append(self.predict(x_test, y_test))
        print('Costo: ', self.costo[0])
        for i in range(n_epocs):
            randomize = np.arange(len(x))
            np.random.shuffle(randomize)
            x = x[randomize]
            y = y[randomize]
            x_batch = [x[k:k+n_batch_size] for k in range(0, len(x), n_batch_size)]
            y_batch = [y[k:k+n_batch_size] for k in range(0, len(x), n_batch_size)]
            for x_batch, y_batch in zip(x_batch, y_batch):
                dW=np.sum([self.loss_gradient(x_batch[i], y_batch[i]) for i in range(len(x_batch))], axis=0)/len(x_batch)+self.lbda*self.W*np.linalg.norm(self.W)**2
                self.W=self.W-self.alfa*dW
            print('Epoca '+str(i+1)+ ' completada')
            self.accuracy.append(self.predict(x_test,y_test))
            self.costo.append(np.sum([self.L_i(x[i], y[i]) for i in range(len(x))])/len(x))
            print('Costo: ', self.costo[i+1])
        return self.costo, self.accuracy

    def loss_gradient(self, x, y):
        pass

    def L_i(self, x, y):
        pass

    def predict(self, x, y):
        pass

class Linear_SM(LinearClassifier):
    def __init__(self, alfa, lbda):
        super().__init__(alfa, lbda)

    def fit(self, x, y, x_test, y_test, n_epocs, n_batch_size):
        super().fit(x, y, x_test, y_test, n_epocs, n_batch_size)
        return self.costo, self.accuracy

    def loss_gradient(self, x, y):
            scores=self.W.dot(x)
            constant=np.argmax(scores)
            loss_i=-np.log(np.exp(scores[y]+constant)/np.sum(np.exp(scores+constant)))
            suma=np.sum(np.exp(scores+constant))
            dW_vector=np.array([np.exp(scores[y])/suma if i!=y else np.exp(scores[y])/suma-1 for i in range(len(scores))])
            return (dW_vector.reshape(len(dW_vector),1) @ x.reshape(1, len(x)))*loss_i

    def L_i(self, x, y):
        scores=self.W.dot(x)
        constant=np.amax(scores)
        return -np.log(np.exp(scores[y]+constant)/np.sum(np.exp(scores+constant)))

    def predict(self, x, y):
        y_predicted=[np.argmax(self.W.dot(x[i])) for i in range(len(x))]
        result=sum(int(y_predicted[i]==y[i]) for i in range(len(y)))*100/len(y)
        print('El accuracy fue de: ', result)
        return result

class Linear_VSM(LinearClassifier):
    def __init__(self, alfa, lbda):
        super().__init__(alfa, lbda)


    def fit(self, x, y, x_test, y_test, n_epocs, n_batch_size):
        super().fit(x, y, x_test, y_test, n_epocs, n_batch_size)
        return self.costo, self.accuracy

    def loss_gradient(self, x, y):
        scores = self.W.dot(x)
        margins = np.maximum(0, scores-scores[y]+1)
        margins[y]=0
        margins=[1 if margins[i] else 0 for i in range(len(margins))]
        loss_i=np.sum(margins)
        dW_vector=np.array([margins[i] if i!=y else -sum(margins) for i in range(len(margins))])
        return (dW_vector.reshape(len(dW_vector),1) @ x.reshape(1, len(x)))*loss_i

    def L_i(self, x, y):
        scores = self.W.dot(x)
        margins = np.maximum(0, scores-scores[y]+1)
        margins[y]=0
        return np.sum(margins)

    def predict(self, x, y):
        y_predicted=[np.argmax(self.W.dot(x[i])) for i in range(len(x))]
        result=sum(int(y_predicted[i]==y[i]) for i in range(len(y)))*100/len(y)
        print('El accuracy fue de: ', result)
        return result

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=np.reshape(x_train, (x_train.shape[0], np.prod(x_train.shape[1:])))
x_train=(x_train-255/2)*2/255
x_train=np.hstack((np.ones((x_train.shape[0],1)), x_train))
x_test=np.reshape(x_test, (x_test.shape[0], np.prod(x_test.shape[1:])))
x_test=(x_test-255/2)*2/255
x_test=np.hstack((np.ones((x_test.shape[0],1)), x_test))

alfa=0.001
lbda=0.0005

n_epocs=10
batch_size=100
VSM=Linear_SM(alfa, lbda)
costo, accuracy = VSM.fit(x_train, y_train, x_test, y_test, n_epocs, batch_size) #tuve que mandar el x_train e y_train al fit por la consigna de plotear por epoca, yo entinedo que el accuracy
#siempre se mide sobre el test set.

fig=plt.figure(figsize=(16,6))
ax1 = plt.gca()
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1=plt.subplot(121)
ax1.scatter(np.arange(n_epocs+1), costo)
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Costo', fontsize=14)
ax1.plot(np.arange(n_epocs+1), costo, color='red')
ax1.set_yscale('log')
ax2 = plt.gca()
ax2.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax2=plt.subplot(122)
ax2.scatter(np.arange(n_epocs+1), accuracy)
ax2.plot(np.arange(n_epocs+1), accuracy, color='red')
ax2.set_xlabel('Época', fontsize=14), ax2.set_ylabel('Precisión', fontsize=14)
ax2.hlines(max(accuracy), 0, 10, color='black', linestyles='dashdot')
fig.suptitle(r'Metodo SoftMax para CIFAR, $\alpha$='+str(alfa)+', $\lambda$='+str(lbda), fontsize=16)
plt.savefig('SM_mnist.pdf', format='pdf')


########################################################################################################################################
#Ejercicio 5 VECTORIZADO
########################################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from keras.datasets import cifar10
from keras.datasets import mnist

class LinearClassifier():
    def __init__(self, alfa, lbda):
        self.alfa=alfa
        self.lbda=lbda
        self.accuracy=[]
        self.costo=[]
    def fit(self, x, y, x_test, y_test, n_epocs, n_batch_size):
        n_categories=np.amax(y_train)+1
        self.rnd = np.random.RandomState(42)
        self.W=1e-3*self.rnd.randn(np.prod(x_train.shape[1:]), n_categories)
        for i in range(n_epocs):
            randomize = np.arange(len(x))
            np.random.shuffle(randomize)
            x = x[randomize]
            y = y[randomize]
            x_batch = [x[k:k+n_batch_size] for k in range(0, len(x), n_batch_size)]
            y_batch = [y[k:k+n_batch_size] for k in range(0, len(x), n_batch_size)]
            costo_sum=0
            acc_sum=0
            it=0
            for x_batch, y_batch in zip(x_batch, y_batch):
                costo, dW=self.loss_gradient(x_test, y_test)
                self.W=self.W-self.alfa*dW
                costo_sum+=costo
                it+=1
                acc_sum+=self.predict(x_test, y_test)
            print('Epoca '+str(i+1)+ ' completada')
            self.accuracy.append(acc_sum/it)
            self.costo.append(costo_sum/it)
            print('Costo: ', self.costo[i])
            print('Accuracy: ', self.accuracy[i])
        return self.costo, self.accuracy

    def loss_gradient(self, x, y):
        pass

    def L_i(self, x, y):
        pass

    def predict(self, x, y):
        pass

class Linear_SM(LinearClassifier):
    def __init__(self, alfa, lbda):
        super().__init__(alfa, lbda)

    def fit(self, x, y, x_test, y_test, n_epocs, n_batch_size):
        super().fit(x, y, x_test, y_test, n_epocs, n_batch_size)
        return self.costo, self.accuracy

    def loss_gradient(self, x_batch, y_batch):
        r=(self.W*self.W).sum()

        scores=x_batch.dot(self.W) #de vuelta, tendra dimension (numero de imagenes, numero de categorias)

        if not isinstance(y_batch, list):
            y_batch=list(y_batch)

        maxval=-np.amax(scores, axis=1)
        scores_exp=np.exp(scores+maxval[:,np.newaxis])
        scores_exp_y=scores_exp[np.arange(x_batch.shape[0]), y_batch]
        sum_scores_exp=np.sum(scores_exp, axis=1)

        l=-np.log(scores_exp_y/sum_scores_exp[:,np.newaxis])

        loss=l.mean()+0.5*self.lbda*r

        #empieza el calculo de gradiente
        aux=scores_exp/sum_scores_exp[:,np.newaxis]
        aux[np.arange(x_batch.shape[0]), y_batch]-=1
        dW=x_batch.T.dot(aux) #bueno x_batch tiene dimension (num imagenes, atributos). La transpuesta (atributos, num_imagenes).
        #binary es (num imagenes, categorias) entonces el producto es (atributos, categorias) -> como W!!! al hacer este producto
        #estoy sumando la derivada dw Wij para cada x, como hice antes en un for.
        dW/= x_batch.shape[0] #esto es por el 1/N en la definicion de loss, en la derivada tambien entra.
        dW+=self.lbda*self.W #falto la parte de regularizacion

        return loss, dW

    def predict(self, x, y):
        y_predicted=np.argmax(x.dot(self.W), axis=1)
        result=np.sum(y_predicted==y)/len(y)
        return result

class Linear_VSM(LinearClassifier):
    def __init__(self, alfa, lbda):
        super().__init__(alfa, lbda)


    def fit(self, x, y, x_test, y_test, n_epocs, n_batch_size):
        super().fit(x, y, x_test, y_test, n_epocs, n_batch_size)
        return self.costo, self.accuracy

    def loss_gradient(self, x_batch, y_batch):
        r=(self.W*self.W).sum()

        scores=x_batch.dot(self.W) #de vuelta, tendra dimension (numero de imagenes, numero de categorias)

        if not isinstance(y_batch, list):
            y_batch=list(y_batch)

        scores_y = scores[np.arange(x_batch.shape[0]), y_batch] #aca basicamente estoy obteniendo un vector en donde cada fila es una imagen
        #y cada valor es el resultado del score de la imagen ganadora, eso estoy haciendo con ese index.
        margins = scores - scores_y[:, np.newaxis]+1 #aca estoy obteniendo una matriz en donde cada fila es una imagen y cada
        #columna el resultado de restar el score de la categoria respondiente (columna i -> categoria i) con el score de la categoria ganadora
        #con el delta, obvio

        #calculamos el max(0, scores)
        margins=np.maximum(0, margins) #aplico la funcion hige

        #la categoria ganadora tiene que ser cero, no se sumaba, esto era para que el error tienda a cero en un clasificador perfecto
        margins[np.arange(x_batch.shape[0]), y_batch]=0 #mismo index que use antes

        l=margins.sum(axis=1) #aca estoy sumando por fila, es decir, por imagen, estoy aplicando la definición de loss para cada imagen
        #entonces l es un vector que en cada fila contiene el loss de cada imagen

        #el 0.5 esta para evitar el 2 en el gradiente
        loss=l.mean()+0.5*self.lbda*r

        #empieza el calculo de gradiente
        binary = margins.copy()
        binary[binary>0]=1 #aca donde habia un numero distinto de cero lo remplazo por 1, esto tiene que ver con como es el gradiente
        #de esta funcion, en la posicion de la categoria ganadora tiene que haber la suma de las cateogrias no ganadoras que le ganaron
        #en la posición de las categorias no ganadoras tiene que haber un 1 si es que le ganaron, esto es porque despues se multiplica por x
        #y ese es el gradiente
        row_sum=binary.sum(axis=1) #estoy contando todas las categorias que dieron mal por imagen (por fila)
        binary[np.arange(x_batch.shape[0]), y_batch]=-row_sum #en la posicion de la categoria ganadora pongo lo que escribi antes
        dW=x_batch.T.dot(binary) #bueno x_batch tiene dimension (num imagenes, atributos). La transpuesta (atributos, num_imagenes).
        #binary es (num imagenes, categorias) entonces el producto es (atributos, categorias) -> como W!!! al hacer este producto
        #estoy sumando la derivada dw Wij para cada x, como hice antes en un for.
        dW/= x_batch.shape[0] #esto es por el 1/N en la definicion de loss, en la derivada tambien entra.
        dW+=self.lbda*self.W #falto la parte de regularizacion

        return loss, dW

    def predict(self, x, y):
        y_predicted=np.argmax(x.dot(self.W), axis=1)
        result=np.sum(y_predicted==y)/len(y)
        return result

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=np.reshape(x_train, (x_train.shape[0], np.prod(x_train.shape[1:])))
x_train=(x_train-255/2)*2/255
x_train=np.hstack((np.ones((x_train.shape[0],1)), x_train))
x_test=np.reshape(x_test, (x_test.shape[0], np.prod(x_test.shape[1:])))
x_test=(x_test-255/2)*2/255
x_test=np.hstack((np.ones((x_test.shape[0],1)), x_test))

alfa=5e-3
lbda=5e-3

n_epocs=10
batch_size=100
VSM=Linear_SM(alfa, lbda)
costo, accuracy = VSM.fit(x_train, y_train, x_test, y_test, n_epocs, batch_size) #tuve que mandar el x_train e y_train al fit por la consigna de plotear por epoca, yo entinedo que el accuracy
#siempre se mide sobre el test set.

fig=plt.figure(figsize=(16,6))
ax1 = plt.gca()
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1=plt.subplot(121)
ax1.scatter(np.arange(n_epocs), costo)
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Costo', fontsize=14)
ax1.plot(np.arange(n_epocs), costo, color='red')
ax1.set_yscale('log')
ax2 = plt.gca()
ax2.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax2=plt.subplot(122)
ax2.scatter(np.arange(n_epocs), accuracy)
ax2.plot(np.arange(n_epocs), accuracy, color='red')
ax2.set_xlabel('Época', fontsize=14), ax2.set_ylabel('Precisión', fontsize=14)
ax2.hlines(max(accuracy), 0, n_epocs, color='black', linestyles='dashdot')
fig.suptitle(r'Metodo SoftMax para MNIST, $\alpha$='+str(alfa)+', $\lambda$='+str(lbda), fontsize=16)
plt.savefig('SoftMax_mnist.pdf', format='pdf')
