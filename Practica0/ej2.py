"""
date: 16-08-2020
File: ej2.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""


import numpy as np

def media_mano(data): #calculo de la media a mano
    return sum(data)/1000 #sumo y divido por la cantidad de elementos

def desviacion(data):
    media=media_mano(data) #calculo la media con mi funcion anterior
    suma_parcial=np.array([(data[i]-media)**2 for i in range(len(data))]) #calculo la desviacion por definicion
    return np.sqrt(np.sum(suma_parcial)/(len(data))) #devuelvo

#calculo usando librerias de numpy
data= np.random.gamma(3,2,1000)
print('media con numpy', np.mean(data))
print('desviacion estandar con numpy', np.std(data))
histograma=np.histogram(data, bins=10, range=None, normed=None, weights=None, density=None) #calculo del histograma
print(histograma) #me daba curiosidad como lo mostraba

#verificacion a mano
print('media hecha a mano', media_mano(data))
print('desviación estandar hecha a mano', desviacion(data))

