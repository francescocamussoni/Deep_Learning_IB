"""
date: 16-08-2020
File: ej14.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""

import numpy as np

def check(grupo): #funcion que chequea si hay repetidos
    if len(grupo) != len(set(grupo)): #si el largo del grupo original, con el largo del grupo por fechas de cumpleaños sin repetir es distinto quiere decir que hay repetidos
        return 1 #devuelvo 1 como 'true'
    else:
        return 0 #de lo contrario no hay repetidos 'false'

tamaño=np.linspace(10,60,6) #me creo un vectorcito con las personas, primero 10, desp 20 etc
cant_grupos=1000 #para cada tamaño de grupo tengo esta cantidad de personas

prob=[]
for j in range(6): #un for PUAJ no se me ocurrio como hacerlo general sin tener que usar un for
    grupos=[np.random.randint(365, size=int(tamaño[j])) for i in range(cant_grupos)] #aca lo que hago es generar los grupos de personas para el tamaño especifico
    repetidos=[check(grupos[i]) for i in range(cant_grupos)].count(1)  #luego me fijo en que cantidad de grupos se repitio algun cumple
    prob.append(repetidos*100/cant_grupos) #guardo el resultado de la probabilidad segun el tamaño de grupo
    print('para un tamaño de grupo {} se obtiene una probabilidad de repetición de {}'.format(tamaño[j], prob[j])) #voy printeando resultados

