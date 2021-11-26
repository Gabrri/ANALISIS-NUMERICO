#Importamos las librerias
import numpy as np
import scipy
import math
from matplotlib import pyplot as plt
from scipy import integrate

#Juan Camilo Rodriguez y Gabriela Cortes Mejia

################# PUNTO A#########################################

# Inicializamos las variables

m_u = 1
I = 1
a= 0.01 #deberia ser -1
b=5
L= 1
p_i = 3.1416

#Biot-Savart

def BS(z,x):
    B_S = (m_u*I)/(4*p_i)*(x/(z**2+x**2)**3/2)
    return B_S

#TRAPEZOIDE STANTARD
# dz = 2L, n= 2L/dz
x_stan = np.linspace(a,b)
XL_s = []
for x in x_stan: XL_s.append(x/L)

#TRAPEZOIDE COMPUESTO
N_1= int(2/0.5)
N_2= int(2/0.1)
N_3= int(2/0.05)
N_4= int(2/0.01)

x_1= np.linspace(a,b,num=N_1)
x_2= np.linspace(a,b,num=N_2)
x_3= np.linspace(a,b,num=N_3)
x_4= np.linspace(a,b,num=N_4)

def comp_trapz(x):
    Y=[]
    a=0
    for i in range (0,len(x)-1,1):
        points = np.arange(start=x[i], stop=x[i+1], step=(x[i+1]-x[i])/3)
        y=[]
        for j in points:
            y.append(BS(0,j))
        a = integrate.trapz(y,points)
        Y.append(a)
    Y.append(a)
    return(Y)

TT_1 = comp_trapz(x_1)
TT_2 = comp_trapz(x_2)
TT_3 = comp_trapz(x_3)
TT_4 = comp_trapz(x_4)
TT_str = comp_trapz(x_stan)


#GRAFICA

plt.figure(1)
plt.plot(x_1,TT_1, marker=".", color="red")
plt.plot(x_2,TT_2, marker=".", color="blue")
plt.plot(x_3,TT_3, marker=".", color="green")
plt.plot(x_4,TT_4, marker=".", color="purple")
plt.plot(x_stan,TT_str, marker=".", color="yellow")
plt.xlabel("x", fontsize = 10)
plt.ylabel("Magnitud del Campo", fontsize = 10)
plt.title("Grafica metodos Trapezoide y Trapezoide Compuesto", fontsize = 10)
plt.legend(['dz=0.5','dz=0.1','dz=0.05','dz=0.01', 'Standart'])
plt.savefig("Grafica_A.PNG")
plt.grid()
plt.show()

########################## PUNTO B ####################################################

def comp_simps(x):
    Y=[]
    a=0
    for i in range (0,len(x)-1,1):
        points = np.arange(start=x[i], stop=x[i+1], step=(x[i+1]-x[i])/2)
        y=[]
        for j in points:
            y.append(BS(0,j))
        a = integrate.simpson(y,points,dx=10)
        Y.append(a)
    Y.append(a)
    return(Y)
SS_1 = comp_simps(x_1)
SS_2= comp_simps(x_2)
SS_3= comp_simps(x_3)
SS_4= comp_simps(x_4)
SS_stan=comp_simps(x_stan)


#GRAFICA

plt.figure(2)
plt.plot(XL_1,SS_1, marker=".", color="red")
plt.plot(x_2,SS_2, marker=".", color="blue")
plt.plot(x_3,SS_3, marker=".", color="green")
plt.plot(x_4,SS_4, marker=".", color="purple")
plt.plot(x_stan,SS_stan, marker=".", color="yellow")
plt.xlabel("x", fontsize = 10)
plt.ylabel("Magnitud del Campo", fontsize = 10)
plt.title("Grafica metodos Simpson y Simpson Compuesto", fontsize = 10)
plt.legend(['dz=0.5','dz=0.1','dz=0.05','dz=0.01', 'Standart'])
plt.savefig("Grafica_B.PNG")
plt.grid()
plt.show()


########################## PUNTO C ####################################################

def BS_C (x,L):
    num = (m_u*I)/(4*p_i)
    denom = x*np.sqrt(x**2+L**2)
    resul = num*(1/denom)*(2*L)
    return(resul)


a_1=0.01
b_1=6
x_C= np.linspace(a_1,b_1,num=N_4)

XL=[]
for x in x_C:XL.append(x/L)

def comp_simps_1(x,L):
    Y=[]
    a=0
    for i in range (0,len(x)-1,1):
        points = np.arange(start=x[i], stop=x[i+1], step=(x[i+1]-x[i])/3)
        y=[]
        for j in points:
            y.append(BS_C(j,L))
        a = integrate.simpson(y,points)
        Y.append(a)
    Y.append(a)
    return(Y)

c_2=comp_simps_1(XL,10)
c = comp_simps_1(x_4,1)

plt.figure(3)
plt.plot(XL,c, marker=".", color="red")
plt.plot(XL,c_2, marker=".", color="purple")
plt.xlabel("x/L", fontsize = 10)
plt.ylabel("Magnitud del Campo", fontsize = 10)
plt.title("Grafica metodos Simpson con L= 0 y L=10", fontsize = 10)
plt.legend(['dz=0.01 L=0','dz=0.01 L=10'])
plt.savefig("Grafica_C.PNG")
plt.grid()
plt.show()
