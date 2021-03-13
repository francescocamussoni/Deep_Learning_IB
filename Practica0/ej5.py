"""
date: 16-08-2020
File: ej5.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""

class Lineal():
    def __init__(self, a, b): #para inicializarlo debo pasarle una pendiente y una ordenada al origen
        self.ordenada=a
        self.pendiente=b
    def __call__(self, val):#para evaluarlo en cualquier punto
        return self.pendiente*val+self.ordenada

a=Lineal(2, 3) #inicializo
print(a(5)) #pruebo la evaluacion, da bien


