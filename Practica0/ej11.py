"""
date: 16-08-2020
File: ej11.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""

import matplotlib.pyplot as plt
import numpy as np

def f(x, y):
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 -y ** 2)

n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X, Y = np.meshgrid(x, y) #esto es para poder usar la funcion de dos variables
plt.figure(figsize=(8, 5), dpi=100)
plt.contourf(X, Y, f(X, Y), alpha=.75, cmap='hot', levels=6) #aca ploteo mis curvas de nivel con X e Y como variables independientes y Z como variable dependiente pongo 6 curvas de nivel para igualar la imagen y con el alpha lo hago un poquito mas transparente, el mapa de colores es hot para que sea como el ejemplo
C = plt.contour(X, Y, f(X, Y), colors='black', linewidths=1) #agrego contornos tambien en 6 niveles si no estaria mostrando mas o menos contornos que mapas de colores
plt.clabel(C, inline=True, fontsize=6, fmt='%.3f') #agrego los labels del valor de nivel a cada contorno, por eso antes lo defini como C=... cambio el formato para mostrar los decimales del ejemplo
plt.savefig('ej11.png', format='png')
plt.show()
