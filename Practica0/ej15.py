"""
date: 16-08-2020
File: ej15.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""

import numpy as np

class Noiser():
    def __init__(self, mn, mx):
        self.__minV=mn #los pongo privados porque sí
        self.__maxV=mx #same
    def __call__(self, value):
        return value+np.random.uniform(self.__minV, self.__maxV) #bueno aca agarro el valor del call y le pongo el ruido que definí

ruido=np.vectorize(Noiser(0.1, 1)) #aca defino mi funcion ruido vectorizada con el ruido 0.1 y 1
x=np.linspace(1,10,10) #genero vector...
print(x) #muestro vector original
print(ruido(x)) #con ruido



