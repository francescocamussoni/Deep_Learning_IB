"""
date: 16-08-2020
File: ej7.py
Author : Francesco Camussoni
Email: camussonif@gmail.com francesco.camussoni@ib.edu.ar
GitHub: https://github.com/francescocamussoni
GitLab: https://gitlab.com/francescocamussoni
Description:
"""

#esto es porque lo habia hecho en archivos separados, este se llamaba ej7.py

import circunferencia #importo circunferencia
from circunferencia import PI as pi  #de circunferencia importo esto en particular
from circunferencia import area #same
print(circunferencia.area(3)) #verifico que es lo mismo esto que
print(area(3)) #esto, al menos en valor
print(circunferencia.PI) #same
print(pi)
if circunferencia.area is area: #aca me fijo si es el mismo objeto o no
    print('son el mismo objeto')
else:
    print('no son el mismo objeto')

if circunferencia.PI is pi: #aca me fijo si es el mismo objeto o no
    print('son el mismo objeto')
else:
    print('no son el mismo objeto')
