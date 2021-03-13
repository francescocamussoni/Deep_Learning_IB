"""
date: 16-08-2020
File: ej13.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class R2():
    def __init__(self, x, y): #la clase R2 simula un espacio bidimensional con coordenadas cartesianas x e y
        self.X=x
        self.Y=y
    def __str__(self): #defino un printeo
        return 'El valor en x es \n{}\n y el valor en Y es \n{}\n'.format(self.X, self.Y)
    def __add__(self, other): #defino la adición
        return R2(self.X+other.X, self.Y+other.Y)
    def __sub__(self, other): #defino la resta
        return R2(self.X-other.X, self.Y-other.Y)
    def __truediv__(self, val): #defino la división
        return R2(self.X/val, self.Y/val)
    def __mul__(self, val): #defino la multiplicación
        return R2(self.X*val, self.Y*val)

class pez():
    def __init__(self, r, v): #defino la clase pez que tiene una posicion y una velocidad en el espacio
        self.r=r
        self.v=v

class cardumen(): #defino la clase cardumen
    def __init__(self, n, maxvel, maxdist, size_tab):
        self.peces=pez(R2(np.random.rand(n)*size_tab-size_tab/2,np.random.rand(n)*size_tab-size_tab/2), R2(np.random.rand(n)*maxvel*2-maxvel,np.random.rand(n)*2*maxvel-maxvel)) #inicializo una cantidad dada de peces con velocidades random dentro del limite y posiciones dentro del tablero
        self.__maxvel=maxvel #defino maxima velocidad
        self.__maxdist=maxdist #defino maxima distnacia entre peces
        self.size=n #por comodidad
        self.size_tab=size_tab

    def doStep(self): #esto va a realizar los step
        def dist(p1x, p2x, p1y, p2y):
            return np.sqrt((p1x-p2x)**2+(p1y-p2y)**2) #dist entre dos puntos

        def norm(x, y):
            return np.sqrt(x**2+y**2) #calcula la norma de un vector en el espacio

        dt=0.1 #necesito algún paso temporal
        v1=(R2(sum(self.peces.r.X), sum(self.peces.r.Y))/self.size-self.peces.r)/8 #calculo la primera def de vel
        v3=(R2(sum(self.peces.v.X), sum(self.peces.v.Y))/self.size-self.peces.v)/8 #la segunda def de velocidades
        v2=R2(np.zeros(self.size), np.zeros(self.size)) #lo voy a tener que ir construyendo
        for i in range(self.size): #lo lamento tuve que usar un for
            for j in range(self.size): #mejor dicho dos, para comparar cada pez con todo el resto
                d=dist(self.peces.r.X[i], self.peces.r.X[j], self.peces.r.Y[i], self.peces.r.Y[j]) #calculo la distancia entre dichos peces
                if i!=j and d<self.__maxdist: #si me quedan muy cerca corrijo
                    v2.X[i], v2.Y[i]=v2.X[i]+(self.peces.r.X[i]-self.peces.r.X[j])/d, v2.Y[i]+(self.peces.r.Y[i]-self.peces.r.Y[j])/d

        self.peces.v=self.peces.v+v1+v2+v3 #actualizo velocidad en el cardumen
        for i in range(self.size):
            norma=norm(self.peces.v.X[i], self.peces.v.Y[i]) #calculo el modulo de la velocidad actualizada
            if norma>self.__maxvel: #si es mayor
                norma=self.__maxvel/norma #realizo una correcion
                self.peces.v.X[i], self.peces.v.Y[i]= self.peces.v.X[i]*norma, self.peces.v.Y[i]*norma #actualizo

        self.peces.r=self.peces.r+self.peces.v*dt #actualiza posicion
        for i in range(self.size): #todo esto es para mantenerlo en el tablero
            if self.peces.r.X[i]>self.size_tab/2 and self.peces.v.X[i]>0:
                self.peces.v.X[i]=-self.peces.v.X[i]
            elif self.peces.r.X[i]<-self.size_tab/2 and self.peces.v.X[i]<0:
                self.peces.v.X[i]=-self.peces.v.X[i]
            if self.peces.r.Y[i]>self.size_tab/2 and self.peces.v.Y[i]>0:
                self.peces.v.Y[i]=-self.peces.v.Y[i]
            elif self.peces.r.Y[i]<-self.size_tab/2 and self.peces.v.Y[i]<0:
                self.peces.v.Y[i]=-self.peces.v.Y[i]

    def print(self): #printeo
        for i in range(self.size):
            print('El pez numero {} esta en ({:.2f},{:.2f}) con velocidad ({:.2f},{:.2f})'.format(i, self.peces.r.X[i], self.peces.r.Y[i], self.peces.v.X[i], self.peces.v.Y[i]))

niter=500
numpeces=40
velmax=4
distmax=2
size_tablero=40
c=cardumen(numpeces,velmax,distmax,size_tablero)
colors = cm.rainbow(np.linspace(0, 1, numpeces))


x = [c.peces.r.X[i] for i in range(c.size)]
y = [c.peces.r.Y[i] for i in range(c.size)]
plt.scatter(x,y, color=colors)
plt.ylim(-size_tablero/2, size_tablero/2)
plt.xlim(-size_tablero/2, size_tablero/2)
plt.savefig('peces1.png', format='png')
plt.show()

for i in range(niter):
    c.doStep()
c.print()

x = [c.peces.r.X[i] for i in range(c.size)]
y = [c.peces.r.Y[i] for i in range(c.size)]
plt.scatter(x,y, color=colors)
plt.ylim(-size_tablero/2, size_tablero/2)
plt.xlim(-size_tablero/2, size_tablero/2)
plt.savefig('peces2.png', format='png')
plt.show()
