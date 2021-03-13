"""
date: 30-08-2020
File: ej1.py
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

def Evaluacion(coefs, X):
    return sum([coefs[i]*X[i] for i in range(len(X))])
def Generacion_puntos(n_puntos, n_coefs):
    return np.array([np.random.uniform(0,10) for i in range(n_points*n_coefs)]).reshape(n_points, n_coefs)

class Noiser():
    def __init__(self, mn, mx):
        self.__minV=mn #los pongo privados porque sí
        self.__maxV=mx #same
    def __call__(self, value):
        return value+np.random.uniform(self.__minV, self.__maxV) #bueno aca agarro el valor del call y le pongo el ruido que definí


n_points=40
n_promedios=100
err=[]
num_coefs=[]
coefs=[1, 1] #el primer coeficiente es la ordenada al origen
for i in range(n_points-1):
    err_aux=0
    for j in range(n_promedios):
        X=Generacion_puntos(n_points, len(coefs)-1) #obtengo puntos random y los reshapeo convenientemente
        X=np.insert(X, 0, np.ones(n_points), axis=1) #para tener en cuenta la ordenada al origen
        Y=[Evaluacion(coefs, X[i]) for i in range(len(X))] #obtengo la coordenada Y de estos puntos aplicando la función de coeficientes definida
        ruido=np.vectorize(Noiser(-0.5, 0.5)) #aca defino mi funcion ruido vectorizada con el ruido 0.1 y 1 así queda como en el ejemplo del practico
        Y=ruido(Y) #Le aplico el ruido a mis valores de salida
        Xt=X.T #Asi no calculo muchas veces al pedo
        A=np.matmul(np.matmul(np.linalg.inv(np.matmul(Xt, X)),Xt), Y) #calculo los coeficientes por definicion
        err_aux+=(np.linalg.norm(A-coefs)/np.sqrt(len(coefs))) #obtengo el error para esta cantidad de coeficientes
    err.append(err_aux/n_promedios) #como sume una cantidad de veces = n_promedios, divido por esto
    num_coefs.append(len(coefs)) #agrego la cantidad de coeficientes para este error
    if len(coefs)==2: #esto es solo para plotear el caso bidimensional
        Y_calc=sum([A[j]*X[:,j] for j in range(len(coefs))])#calculo el valor obtenido
        plt.scatter(X[:,1], Y, color='blue', alpha=0.9, label='Valores medidos') #ploteo las mediciones
        plt.plot(X[:,1], Y_calc, color='red', linewidth=1.5, label='Ajuste lineal') #ploteo el modelo
        plt.xlabel('Variable independiente')
        plt.ylabel('Variable dependiente')
        plt.title('Caso 1D')
        plt.savefig('Caso1d.pdf', format='pdf')
        plt.legend()
        plt.show()
    coefs.append(1)
n_points+=20
plt.scatter(num_coefs, err)
plt.yscale('log')
plt.title('Error en función de la dimensión')
plt.ylabel('Norma del error normalizada')
plt.xlabel('Dimensión del problema')
plt.savefig('ej1.pdf', format='pdf')
plt.show()
