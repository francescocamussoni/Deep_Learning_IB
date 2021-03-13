"""
date: 16-08-2020
File: ej1.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""

import numpy as np
A=np.array([[1, 0, 1], [2, -1, 1], [-3, 2, -2]]) #defino matriz de coeficientes
b=np.array([[-2], [1], [-1]]) #defino vector columna de resultados
print(np.linalg.solve(A, b)) #aguante numpy
