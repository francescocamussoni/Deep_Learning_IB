"""
date: 30-08-2020
File: ej4.py
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

def split_train_test(set, test_ratio): #Como los datos los genere ordenadamente por clase aca los ordeno aleatoriamente y de paso parto mi set en training y test set
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(set)) #obtengo indices random
    test_set_size = int(len(set) * test_ratio) #obtengo la cantidad de indices necesarios para mi housing test
    test_indices = shuffled_indices[:test_set_size] #obtengo dichos indices
    train_indices = shuffled_indices[test_set_size:] #obtengo los indices restantes para el training
    return set[train_indices], set[test_indices]

class KNN():
    def __init__(self, x_train, y_train):
        self.X=x_train #agarro el array cuadrado de imagenes y lo pongo lineal, para cada una de las imagenes
        self.Y=y_train

    def Prediccion(self, x_test, k=1, norm=2):
        def Calculo(self, x_test, k, norm):
            distance=np.linalg.norm(self.X-x_test, norm, axis=1)#calculo la norma entre mi imagen a predecir y todo el training sample
            distance=np.concatenate((distance.reshape(len(distance), 1), self.Y.reshape(len(y_train),1)), axis=1) #obtengo un array de distnacias y de resultados
            distance=distance[np.argsort(distance[:, 0])] #ordeno dicho array según las distancias, moviendo adecuadamente los resultados
            result = np.argmax(np.bincount([int(distance[i,1]) for i in range(k)]))
            return result #esto lo encontre por internet porque no sabia que hacer cuando se repetian la cantidad de resultados
            # se me habia ocurrido medir la distnacia de cara uno las categorias pero era un poco complicado y tampoco me parecia el mejor de los criterios
            # en internet decía que también se seleccionaba simplemente al azar. Estoy abierto a sugerencias
        if x_test.size>2: #esto tiene sentido si le mando un conjunto de valores para predecir
            y_predicted=np.zeros(len(x_test)) #para inicializar nomas
            for i in range(len(x_test)): #busque hacerlo de alguna forma matricial pero no encontre, estaría bueno poder hacerlo así, asi no tenemos que loopear tanto, si hay alguna forma me gustaria saberla
                y_predicted[i]=Calculo(self, x_test[i], k, norm)
        else:
            y_predicted=Calculo(self, x_test.ravel(), k, norm)
        return y_predicted #devuelvo

n_clases=8
n_pp_clase=50

x_total=[]
y_total=[]
for i in range(n_clases):
    x_total.append(np.random.multivariate_normal([np.random.uniform(-10,10), np.random.uniform(-10,10)],
                                             np.eye(2)*np.random.uniform(0.5,3.5), n_pp_clase))
    y_total.append([i]*n_pp_clase)
x_total=np.reshape(x_total, (n_clases*n_pp_clase,2))
y_total=np.reshape(y_total, (n_clases*n_pp_clase,1))
set=np.concatenate((x_total, y_total), axis=1)

training_set, test_set = split_train_test(set, 0.1)
x_train, y_train, x_test, y_test = training_set[:,:2], training_set[:,-1], test_set[:,:2], test_set[:,-1]
knn=KNN(x_train, y_train)
y_predicted=knn.Prediccion(x_test, 10)
results=sum(int(y_predicted[i]==y_test[i]) for i in range(len(y_predicted)))
print('El accuracy fue de: ', results/len(y_test)*100)

k=[1,3,7]
for i in range(len(k)):


    ## bueno ahora vamos con lo del ploteo
    xx, yy = np.meshgrid(np.arange(-10, 10.01, 0.1), np.arange(-10, 10.01, 0.1))
    z=knn.Prediccion(np.c_[xx.ravel(), yy.ravel()], k[i])
    z = z.reshape(xx.shape)

    fig=plt.figure(figsize=(12,8))
    cs=plt.pcolormesh(xx, yy, z, cmap='Pastel2')
    cbar = fig.colorbar(cs)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Zona de clasificación para '+str(k[i])+' vecinos cercanos', fontsize=16)
    # Plot also the training points
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='Dark2', edgecolor='black', alpha=0.5)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('Clasificación para '+str(k[i])+' vecinos cercanos.', fontsize=19)
    plt.xlabel('Característica 1', fontsize=16)
    plt.ylabel('Característica 2', fontsize=16)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
    plt.contour(xx, yy, z, cmap='tab20', alpha=0.5, linewidths=0.5)
    plt.savefig('ej4_'+str(k[i])+'.pdf', format='pdf')
    plt.show()
    plt.close()
