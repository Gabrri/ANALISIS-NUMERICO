import numpy as np
import scipy
import math
from matplotlib import pyplot as plt
from scipy import integrate

m_u = 1
I = 1
a= 0.1 #deberia ser -1
b= 1
L= 10
p_i = 3.1416

#Biot-Savart

def BS(z,x):
    B_S = (m_u*I)/(4*p_i)*(x/(z**2+x**2)**3/2)
    return B_S

N = int((b-a)/0.5)
X = np.linspace(a, b, N, endpoint=True)


def comp_trapz(sup,inf,delta,X):
    Y=[]
    a=0
    for i in range (0,len(X)-1,1):
        points = np.arange(start=X[i], stop=X[i+1], step=(X[i+1]-X[i])/2)
        y=[]
        for j in points:
            y.append(BS(0,j))
            a = integrate.trapz(y,points,dx=0.5)
        Y.append(a)
    Y.append(a)
    return(Y)

T_1 = comp_trapz(b,a,0.5,X)


plt.figure(1)
plt.plot(X,T_1, marker=".", color="red")
plt.xlabel("x", fontsize = 10)
plt.ylabel("Magnitud del Campo", fontsize = 10)
plt.title("Grafica metodos Trapezoide y Trapezoide Compuesto", fontsize = 10)
plt.legend(['dz=0.5','dz=0.1','dz=0.05','dz=0.01', 'Standart'])
plt.savefig("Grafica_A.PNG")
plt.grid()
plt.show()
