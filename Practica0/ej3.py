"""
date: 16-08-2020
File: ej3.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""

import numpy as np
def raices(a, b=0, c=0): #si no me dan b y c son 0, a me lo tienen que dar si o si
    return np.array([[(np.sqrt(complex(b**2-4*a*c))-b)/(2*a)],[(-b-np.sqrt(complex(b**2-4*a*c)))/(2*a)]]) #como np.sqrt no me devuelve un resultado complejo si el argumento no es complejo tengo que aclararcelo por las dudas, si no me devolvería NaN

print(raices(1,1,3)) #imprimo el resultado
