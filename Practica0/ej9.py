"""
date: 16-08-2020
File: ej9.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""

#esto es porque lo habia hecho en archivos separados, este era el ej9.py

import p0_lib
from p0_lib.circunferencia import PI, area
from p0_lib.elipse import area
from p0_lib.rectangulo import area as area_rect

print('el area del rectangulo de lado 4 es', area_rect(4))
print('el area de la elpise de semiejes 3 y 1 es', area(3,1))
print('el area del circulo de radio 3 es', area(3))
