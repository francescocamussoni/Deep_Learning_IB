"""
date: 30-08-2020
File: ej2.py
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

def Generacion_puntos(center, n_points, n_dim):
    return np.random.multivariate_normal(center,np.eye(n_dim)*np.random.uniform(0.5,1.5), n_points)

class Noiser():
    def __init__(self, mn, mx):
        self.__minV=mn
        self.__maxV=mx
    def __call__(self, value):
        return value+np.random.uniform(self.__minV, self.__maxV) #bueno aca agarro el valor del call y le pongo el ruido que definí

def Nearest(X, centers):
    Y=np.zeros(len(X))
    for i in range(len(X)):
        dist=[np.linalg.norm(X[i]-centers[j]) for j in range(len(centers))]
        Y[i]=np.argmin(dist)
    return Y

def Mass_center(X, Y):
    X0=[X[i] for i in range(len(X)) if Y[i]==0]
    X1=[X[i] for i in range(len(X)) if Y[i]==1]
    X2=[X[i] for i in range(len(X)) if Y[i]==2]
    X3=[X[i] for i in range(len(X)) if Y[i]==3]

    return [np.average(X0, axis=0), np.average(X1, axis=0), np.average(X2, axis=0), np.average(X3, axis=0)]



#generación de clusters, así es como los tengo definidos YO a mano, los centros los ubico 'a mano' pero les agrego una componente random
#espero que a esto haga referencia que la inicializacion sea random, por que no entiendo como hacer para ubicar 2 cosas solapadas si
#fuese totalmente random.
ruido=np.vectorize(Noiser(-0.5, 0.5)) #voy a agregar este ruido que va a ser lo que le agregue la componente 'random' al problema
center0=ruido([1, -2, -4]) #defino centros
center1=ruido([-4, 3, 1])
center2=ruido([5, 3, -2])
center3=ruido([-3, 2, 0])

n_points=50 #defino num points
n_dim=3#defino dimensiones

X_clusters=np.array([Generacion_puntos(center0, n_points, n_dim), #lo defino así solo para poder plotear cluster original
   Generacion_puntos(center1, n_points, n_dim),
   Generacion_puntos(center2, n_points, n_dim),
   Generacion_puntos(center3, n_points, n_dim)])

#muchas lineas de ploteo sry not sry...........
fig = plt.figure(figsize=(18,6))
ax1= fig.add_subplot(1,4,1)
ax1.scatter(X_clusters[0,:,0],X_clusters[0,:,1],color='red', alpha=0.5, label='Grupo 1')
ax1.scatter(X_clusters[1,:,0],X_clusters[1,:,1],color='blue', alpha=0.5, label='Grupo 2')
ax1.scatter(X_clusters[2,:,0],X_clusters[2,:,1],color='yellow', alpha=0.5, label='Grupo 3')
ax1.scatter(X_clusters[3,:,0],X_clusters[3,:,1],color='green', alpha=0.5, label='Grupo 4')
ax1.set_xlabel('Caracteristica 1'), ax1.set_ylabel('Caracteristica 2')
ax2= fig.add_subplot(1,4,2)
ax2.scatter(X_clusters[0,:,1],X_clusters[0,:,2],color='red', alpha=0.5, label='Grupo 1')
ax2.scatter(X_clusters[1,:,1],X_clusters[1,:,2],color='blue', alpha=0.5, label='Grupo 2')
ax2.scatter(X_clusters[2,:,1],X_clusters[2,:,2],color='yellow', alpha=0.5, label='Grupo 3')
ax2.scatter(X_clusters[3,:,1],X_clusters[3,:,2],color='green', alpha=0.5, label='Grupo 4')
ax2.set_xlabel('Caracteristica 2'), ax2.set_ylabel('Caracteristica 3')
ax3= fig.add_subplot(1,4,3)
ax3.scatter(X_clusters[0,:,2],X_clusters[0,:,0],color='red', alpha=0.5, label='Grupo 1')
ax3.scatter(X_clusters[1,:,2],X_clusters[1,:,0],color='blue', alpha=0.5, label='Grupo 2')
ax3.scatter(X_clusters[2,:,2],X_clusters[2,:,0],color='yellow', alpha=0.5, label='Grupo 3')
ax3.scatter(X_clusters[3,:,2],X_clusters[3,:,0],color='green', alpha=0.5, label='Grupo 4')
ax3.set_xlabel('Caracteristica 3'), ax3.set_ylabel('Caracteristica 1')
ax4 = fig.add_subplot(1, 4, 4, projection='3d')
ax4.scatter(X_clusters[0,:,0],X_clusters[0,:,1],X_clusters[0,:,2], color='red', label='Grupo 1')
ax4.scatter(X_clusters[1,:,0],X_clusters[1,:,1],X_clusters[1,:,2], color='blue', label='Grupo 2')
ax4.scatter(X_clusters[2,:,0],X_clusters[2,:,1],X_clusters[2,:,2], color='yellow', label='Grupo 3')
ax4.scatter(X_clusters[3,:,0],X_clusters[3,:,1],X_clusters[3,:,2], color='green', label='Grupo 4')
ax4.set_xlabel('Caracteristica 1'), ax4.set_ylabel('Caracteristica 2'), ax4.set_zlabel('Caracteristica 3')
ax1.legend(), ax2.legend(), ax3.legend(), ax4.legend()
fig.suptitle('Clusters originales')
plt.savefig('clusteroriginales.pdf', format='pdf')
plt.show()
plt.close()
#muchas lineas de ploteo sry not sry...........

X=np.concatenate((X_clusters[0],X_clusters[1],X_clusters[2],X_clusters[3])) #ahora los tengo todos mezclados y perdi cual es cual

#generación de centros,

#la primera definición es porque buscaba aproposito que el algoritmo funcione mal y la segunda quería ver como intentaba separar cosas solapadas
# centers=[np.random.uniform(-1,1, size=3),
#         np.random.uniform(0.5,2, size=3),
#         np.random.uniform(-5,-3, size=3),
#         np.random.uniform(3,6, size=3)]

centers=[ruido([1, -2, -4]), ruido([-4, 3, 1]), ruido([5, 3, -2]), ruido([-3, 2, 0])]


n_steps=40
for i in range(n_steps):
    Y=Nearest(X, centers)
    centers=Mass_center(X, Y)
    #muchas lineas de ploteo, nuevamente
    if i==0 or i%10==0:
        X0=[X[i,0] for i in range(len(X)) if Y[i]==0]
        X1=[X[i,0] for i in range(len(X)) if Y[i]==1]
        X2=[X[i,0] for i in range(len(X)) if Y[i]==2]
        X3=[X[i,0] for i in range(len(X)) if Y[i]==3]
        Y0=[X[i,1] for i in range(len(X)) if Y[i]==0]
        Y1=[X[i,1] for i in range(len(X)) if Y[i]==1]
        Y2=[X[i,1] for i in range(len(X)) if Y[i]==2]
        Y3=[X[i,1] for i in range(len(X)) if Y[i]==3]
        Z0=[X[i,2] for i in range(len(X)) if Y[i]==0]
        Z1=[X[i,2] for i in range(len(X)) if Y[i]==1]
        Z2=[X[i,2] for i in range(len(X)) if Y[i]==2]
        Z3=[X[i,2] for i in range(len(X)) if Y[i]==3]
        fig = plt.figure(figsize=(18,6))
        ax1= fig.add_subplot(1,4,1)
        ax1.scatter(X0, Y0, color='red', alpha=0.5, label='Grupo 1')
        ax1.scatter(X1, Y1, color='blue', alpha=0.5, label='Grupo 2')
        ax1.scatter(X2, Y2, color='yellow', alpha=0.5, label='Grupo 3')
        ax1.scatter(X3, Y3, color='green', alpha=0.5, label='Grupo 4')
        ax1.set_xlabel('Caracteristica 1'), ax1.set_ylabel('Caracteristica 2')
        ax2= fig.add_subplot(1,4,2)
        ax2.scatter(Y0, Z0,color='red', alpha=0.5, label='Grupo 1')
        ax2.scatter(Y1, Z1,color='blue', alpha=0.5, label='Grupo 2')
        ax2.scatter(Y2, Z2,color='yellow', alpha=0.5, label='Grupo 3')
        ax2.scatter(Y3, Z3,color='green', alpha=0.5, label='Grupo 4')
        ax2.set_xlabel('Caracteristica 2'), ax2.set_ylabel('Caracteristica 3')
        ax3= fig.add_subplot(1,4,3)
        ax3.scatter(Z0, X0,color='red', alpha=0.5, label='Grupo 1')
        ax3.scatter(Z1, X1,color='blue', alpha=0.5, label='Grupo 2')
        ax3.scatter(Z2, X2,color='yellow', alpha=0.5, label='Grupo 3')
        ax3.scatter(Z3, X3,color='green', alpha=0.5, label='Grupo 4')
        ax3.set_xlabel('Caracteristica 3'), ax3.set_ylabel('Caracteristica 1')
        ax4 = fig.add_subplot(1, 4, 4, projection='3d')
        ax4.scatter(X0, Y0, Z0, color='red', label='Grupo 1')
        ax4.scatter(X1, Y1, Z1, color='blue', label='Grupo 2')
        ax4.scatter(X2, Y2, Z2, color='yellow', label='Grupo 3')
        ax4.scatter(X3, Y3, Z3, color='green', label='Grupo 4')
        ax4.set_xlabel('Caracteristica 1'), ax4.set_ylabel('Caracteristica 2'), ax4.set_zlabel('Caracteristica 3')
        ax1.legend(), ax2.legend(), ax3.legend(), ax4.legend()
        fig.suptitle('Iteración '+str(i))
        plt.savefig('clusteriteracion'+str(i)+'.pdf', format='pdf')
        plt.show()
        plt.close()
