#Importamos las librerias
import numpy as np
import scipy
import math
from matplotlib import pyplot as plt
from scipy import integrate
from scipy.integrate import quad

#Juan Camilo Rodriguez y Gabriela Cortes Mejia

################# PUNTO A#########################################

# Inicializamos las variables

m_u = 1
I = 1
L = 1
a= -1
b= 1
p_i = 3.1416

#Calculando los pasos de diferente tamaño
# dz =[0,5, 0,1, 0,05, 0,01]
N_1= int((b-a)/0.5)
N_2= int((b-a)/0.1)
N_3= int((b-a)/0.05)
N_4= int((b-a)/0.01)

#Biot-Savart

def f1(x):
    B_S = (m_u*I)/(4*p_i)*(math.sin(p_i)/x**2)
    return B_S

x_1= np.linspace(a,b,num=N_1)
x_2= np.linspace(a,b,num=N_2)
x_3= np.linspace(a,b,num=N_3)
x_4= np.linspace(a,b,num=N_4)

y_1= f1(x_1)
y_2= f1(x_2)
y_3= f1(x_3)
y_4= f1(x_4)

#Integra = quad(f1, -1,-0.1)
#Integra2= quad(f1, 0.1,1)

#print(Integra)
#print(Integra2)

#Trapezoide y simpson

Tr_1 = integrate.trapz(y_1, x_1)
Tr_2 = integrate.trapz(y_2, x_2)
Tr_3 = integrate.trapz(y_3, x_3)
Tr_4 = integrate.trapz(y_4, x_4)

print("Aproximaciones (Trapezoide) con diferentes Deltas de z: ")
print(Tr_1)
print(Tr_2)
print(Tr_3)
print(Tr_4)

p=[1,2,3,4]
dz =[0.5, 0.1, 0.05, 0.01]

#GRAFICA
#trapexoide
Puntos_A_Trapz =[Tr_1, Tr_2, Tr_3, Tr_4]

plt.figure(1)
plt.scatter(dz, Puntos_A_Trapz)
plt.plot(dz,Puntos_A_Trapz, marker="o", color="red")
plt.xlabel("dz", fontsize = 10)
plt.ylabel("Trapezoidal Rule Points", fontsize = 10)
plt.title("Grafica comparando metdos Biot-Savart, Trapezoidal y Trapezoidal Composite", fontsize = 15)
plt.legend(['Trapezoidal'])
plt.savefig("Grafica_A.PNG")
plt.grid()

########################## PUNTO B ####################################################

#Usando Simpson
Si_1 = integrate.simpson(y_1, x_1)
Si_2 = integrate.simpson(y_2, x_2)
Si_3 = integrate.simpson(y_3, x_3)
Si_4 = integrate.simpson(y_4, x_4)

print("Aproximaciones (Simpson) con diferentes Deltas de z: ")
print(Si_1)
print(Si_2)
print(Si_3)
print(Si_4)


#GRAFICA
#trapexoide
Puntos_A_Simp =[Si_1, Si_2, Si_3, Si_4]
p=[1,2,3,4]
dz =[0.5, 0.1, 0.05, 0.01]

plt.figure(2)
plt.scatter(dz, Puntos_A_Simp)
plt.plot(dz,Puntos_A_Simp, marker="o", color="blue")
plt.xlabel("dz", fontsize = 10)
plt.ylabel("Simpson Rule Points", fontsize = 10)
plt.title("Grafica comparando metdos Biot-Savart, Simpson y Simpson Composite", fontsize = 15)
plt.legend(['Simpson'])
plt.savefig("Grafica_B.PNG")
plt.grid()

plt.show()
