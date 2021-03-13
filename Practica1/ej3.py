"""
date: 30-08-2020
File: ej3.py
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

class KNN():
    def __init__(self, x_train, y_train):
        self.X=np.reshape(x_train, (x_train.shape[0], np.prod(x_train.shape[1:]))) #agarro el array cuadrado de imagenes y lo pongo lineal, para cada una de las imagenes
        self.Y=y_train

    def Prediccion(self, x_test, k=1, norm=2):
        def Calculo(self, x_test, k, norm):
            distance=np.linalg.norm(self.X-x_test.astype(np.int16), norm, axis=1)#calculo la norma entre mi imagen a predecir y todo el training sample
            distance=np.concatenate((distance.reshape(len(distance), 1), self.Y.reshape(len(y_train),1)), axis=1) #obtengo un array de distnacias y de resultados
            distance=distance[np.argsort(distance[:, 0])] #ordeno dicho array según las distancias, moviendo adecuadamente los resultados
            result = np.argmax(np.bincount([int(distance[i,1]) for i in range(k)]))
            return result #(esto esta por que despues lo uso para plotear algunas cosas)

        if len(np.shape(x_test))>2: #esto tiene sentido si le mando un conjunto de valores para predecir
            y_predicted=np.zeros(len(x_test)) #para inicializar nomas
            index=np.zeros(len(x_test))
            for i in range(len(x_test)): #busque hacerlo de alguna forma matricial pero no encontre, estaría bueno poder hacerlo así, asi no tenemos que loopear tanto, si hay alguna forma me gustaria saberla
                y_predicted[i]=Calculo(self, x_test[i].ravel(), k, norm)
        else:
            y_predicted=Calculo(self, x_test.ravel(), k, norm)
        return y_predicted


(x_train, y_train), (x_test, y_test) = mnist.load_data()

n=20
k=1
norm=2
knn=KNN(x_train, y_train)
y_predicted=knn.Prediccion(x_test[:n], k, norm)
print('predicciones: ', y_predicted)
print('valores reales: ', y_test[:n])
print('Son iguales?: ', np.array_equal(y_predicted, y_test[:n]))

#Esto fue solo para plotear y presentar una imagen en el informe, habia modificado un poco la clase para que ademas de devolverme la prediccion me devuelva el indice
#del vecino mas cercano, borre eso porque molestaba para el funcionamiento normal, no fue mas que agregar index=np.argmin(distance) y poner ese valor a una variable index[i].
# fig = plt.figure(figsize=(15,6))
# ax1=plt.subplot(251)
# ax1.imshow(x_test[0])
# ax1.set_title('Número '+str(y_test[0]), fontsize=14)
# ax1.set_xticks([])
# ax1.set_yticks([])
# ax2=plt.subplot(252)
# ax2.imshow(x_test[1])
# ax2.set_title('Número '+str(y_test[1]), fontsize=14)
# ax2.set_xticks([])
# ax2.set_yticks([])
# ax3=plt.subplot(253)
# ax3.imshow(x_test[2])
# ax3.set_title('Número '+str(y_test[2]), fontsize=14)
# ax3.set_xticks([])
# ax3.set_yticks([])
# ax4=plt.subplot(254)
# ax4.imshow(x_test[3])
# ax4.set_title('Número '+str(y_test[3]), fontsize=14)
# ax4.set_xticks([])
# ax4.set_yticks([])
# ax4=plt.subplot(255)
# ax4.imshow(x_test[4])
# ax4.set_title('Número '+str(y_test[4]), fontsize=14)
# ax4.set_xticks([])
# ax4.set_yticks([])
# ax5=plt.subplot(256)
# ax5.imshow(x_train[index[0]])
# ax5.set_title('Número '+str(y_train[int(index[0])])+' más cercano', fontsize=14)
# ax5.set_xticks([])
# ax5.set_yticks([])
# ax6=plt.subplot(257)
# ax6.imshow(x_train[index[1]])
# ax6.set_title('Número '+str(y_train[int(index[1])])+' más cercano', fontsize=14)
# ax6.set_xticks([])
# ax6.set_yticks([])
# ax7=plt.subplot(258)
# ax7.imshow(x_train[index[2]])
# ax7.set_title('Número '+str(y_train[int(index[2])])+' más cercano', fontsize=14)
# ax7.set_xticks([])
# ax7.set_yticks([])
# ax8=plt.subplot(259)
# ax8.imshow(x_train[index[3]])
# ax8.set_title('Número '+str(y_train[int(index[3])])+' más cercano', fontsize=14)
# ax8.set_xticks([])
# ax8.set_yticks([])
# ax9=plt.subplot(2,5,10)
# ax9.imshow(x_train[index[4]])
# ax9.set_title('Número '+str(y_train[int(index[4])])+' más cercano', fontsize=14)
# ax9.set_xticks([])
# ax9.set_yticks([])
# plt.suptitle('Primeros 5 números del test set y su vecino más cercano', fontsize=16)
# plt.savefig('numeros.pdf', format='pdf')
# plt.show()
