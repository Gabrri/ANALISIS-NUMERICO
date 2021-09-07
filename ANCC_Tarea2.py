#Importamos las librerias

import numpy as np
import scipy
import math
from scipy import integrate
from scipy.integrate import quad

#Juan Camilo Rodriguez y Gabriela Cortes Mejia


# Inicializamos las variables
# dz =[0,5, 0,1, 0,05, 0,01]

m_u = 1
I = 1
L = 1
a= -1
b= 1

p_i = 3.1416

#Biot-Savart

def f1(x):
    B_S = (m_u*I)/(4*p_i)*(math.sin(p_i)/x**2)
    return B_S

x= np.linspace(a,b)
y= f1(x)

Integra = quad(f1, -1,-0.1)
Integra2= quad(f1, 0.1,1)

print(Integra)
print(Integra2)

#Trapezoide y simpson

Tr = integrate.trapz(y, x)
Simp= integrate.simpson(y, x)
print(Tr)
print(Simp)
