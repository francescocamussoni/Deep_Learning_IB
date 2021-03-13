"""
date: 16-08-2020
File: ej4.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""

import numpy as np
import matplotlib.pyplot as plt

def raices(a, b=0, c=0): #por enunciado necesito a
    return np.array([[(np.sqrt(complex(b**2-4*a*c))-b)/(2*a)],[(-b-np.sqrt(complex(b**2-4*a*c)))/(2*a)]]) #devuelvo el resultado de las raices, recuerdo que puede ser complejo
   #en ese caso tengo que especificar que el argumento de sqrt es complejo para que no me tire NaN

def parabola(a, b=0, c=0): #por enunciado necesito a
	x=np.arange(-4.5, 4, 0.01) #genero un vector para plotear
	raiz=raices(a,b,c) #calculo las raices segun el argumento de la funcion
	plt.figure(figsize=(8, 5), dpi=100) #genero la figura
	plt.subplot(111) #va a tener 1 solo plot..
	if not [i for i in raiz if np.iscomplex(i)]: #si es compleja al pedo mostrar algo
		plt.plot(x, a*x**2+b*x+c) #ploteo el resultado
		if raiz[0]!=raiz[1]: #puedo tener dos opciones, o que sea una raiz multiple o 2
			plt.scatter(raices(a, b, c), np.array([0, 0])) #ubico mis dos raices en x correspondiente y en y=0, pedia en rojo pero me gusta mas el azul que viene por defecto
			plt.annotate('$x_1$='+str(np.real(raiz[0])), xy=(np.real(raiz[0]), 0), xytext=(np.real(raiz[0]), 5), fontsize=12, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")) #anoto
			plt.annotate('$x_2$='+str(np.real(raiz[1])), xy=(np.real(raiz[1]), 0), xytext=(np.real(raiz[1]), 5), fontsize=12, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")) #anoto
		else: #si es una raiz multiple...
			plt.scatter(raices(a, b, c), np.array([0, 0]))
			plt.annotate('$x$='+str(np.real(raiz[0])), xy=(np.real(raiz[0]), 0), xytext=(np.real(raiz[0]), 5), fontsize=12, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
		plt.savefig('ej4.png', format='png') # la guardo para ver que onda como queda mi anotación
		plt.show()
	else: #si tiene dos raices complejas conjugadas..
		print('tiene raices complejas')

parabola(1, 7, 12)
