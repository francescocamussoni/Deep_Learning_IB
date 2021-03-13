"""
date: 16-08-2020
File: ej10.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
n = 10
x = np.linspace(-3, 3, 4 * n)
y = np.linspace(-3, 3, 3 * n)

X, Y = np.meshgrid(x, y)
plt.figure(figsize=(8, 5), dpi=100)
colormap=plt.imshow(f(X, Y), cmap='bone') #configuro el color del colormap
plt.axis('off') # para borrar los ejes
plt.gca().invert_yaxis() #para que quede como en la imagen
plt.colorbar(colormap) # estoy creando el colorbar con la configuracion de ejes deseada
plt.savefig('ej10.png', format='png')
plt.show()

