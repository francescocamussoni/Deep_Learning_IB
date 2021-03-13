"""
date: 16-08-2020
File: ej12.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""

import numpy as np
import matplotlib.pyplot as plt

n = 1024
X = np.random.normal(0,1,n)
Y = np.random.normal(0,1,n)
plt.figure(figsize=(8, 5), dpi=100)
colors=np.arctan2(Y,X) #vi que habia como una distribucion de colores que depende del angulo, como sabemos para pasar de cartesianas a polares, mas en particular, al angulo, es con la arcotangente de Y/X. Googleando encontre que para que esto se visualice en el grafico debo definir un 'rango' de valores que entra como argumento en plt.scatter
plt.scatter(X,Y, c=colors, cmap='jet', alpha=0.75, s=30, linewidths=0.3, edgecolors='black') #aca ploteo usando el 'rango' de colores calculado anteriormente y utilizando el campo de colores que más coincida con el ejemplo. Le agregue un cierto espesor, le puse un ciderto tamaño y le baje la opacidad para que sea lo mas parecido posible
plt.xlim([-1.5,1.5]) #como es una distribución normal, a medida que te alejas del centro se iba perdiendo la densidad de puntos y no es lo que se muestra en la figura asique limite la region ploteo
plt.ylim([-1.5,1.5])
plt.xticks([]) #le saque los ticks
plt.yticks([])
plt.show()
plt.figure(figsize=(8, 5), dpi=100)
plt.scatter(X,Y)
plt.xlim([-1.5,1.5]) #como es una distribución normal, a medida que te alejas del centro se iba perdiendo la densidad de puntos y no es lo que se muestra en la figura asique limite la region ploteo
plt.ylim([-1.5,1.5])
plt.xticks([]) #le saque los ticks
plt.yticks([])
plt.savefig('ej12.png', format='png')
plt.show()



