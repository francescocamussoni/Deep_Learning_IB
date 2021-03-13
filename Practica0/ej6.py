"""
date: 16-08-2020
File: ej6.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""

class Lineal():
    def __init__(self, a, b): #same as before
        self.ordenada=a
        self.pendiente=b
    def __cal__(self, x):
        return self.pendiente*x+self.ordenada
class Exponencial(Lineal): #aca voy a heredar la inicializacion de la linea
    def __init__(self, a, b):
        super().__init__(a, b) #heredo las caracteristicas de lineal
    def __call__(self, x):
        return self.ordenada*x**self.pendiente #evaluo

asd=Exponencial(2,3) #verifico
print(asd(2)) #da bien
