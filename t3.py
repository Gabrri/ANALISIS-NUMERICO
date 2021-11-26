import numpy as np
import matplotlib.pyplot as plt

# Set constants
m = 100000
k = 2e-6


# Define problem and its exact solution
def disease_spread(t, y):
    return k*(m-y)*y


def exact_solution(t, y0):
    c = (m - y0)/(m*y0)
    y = m/(1 + m*c*np.exp(-k*m*t))
    return y


# Define numerical approximations to the solution
def euler(f, a, b, w0, h=0.5):
    t = [a]
    w = [w0]
    while t[-1] < b:
        ti = t[-1]
        wi = w[-1]
        ti1 = ti + h
        wi1 = wi + h*f(ti, wi)
        w.append(wi1)
        t.append(ti1)
    return np.array(t), np.array(w)


def modified_euler(f, a, b, w0, h=0.5):
    t = [a]
    w = [w0]
    while t[-1] < b:
        wi = w[-1]
        ti = t[-1]
        ti1 = ti + h
        wi1 = wi + h/2*(f(ti, wi) + f(ti1, wi + h*f(ti, wi)))
        t.append(ti1)
        w.append(wi1)
    return np.array(t), np.array(w)


def midpoint(f, a, b, w0, h=0.5):
    t = [a]
    w = [w0]
    while t[-1] < b:
        wi = w[-1]
        ti = t[-1]
        ti1 = ti + h
        wi1 = wi + h*f(ti + h/2, wi + h/2*f(ti, wi))
        t.append(ti1)
        w.append(wi1)
    return np.array(t), np.array(w)


def runge_kutta(f, a, b, w0, h=0.5):
    t = [a]
    w = [w0]
    while t[-1] < b:
        wi = w[-1]
        ti = t[-1]
        ti1 = ti + h
        k1 = h*f(ti, wi)
        k2 = h*f(ti + h/2, wi + k1/2)
        k3 = h*f(ti + h/2, wi + k2/2)
        k4 = h*f(ti1, wi + k3)
        wi1 = wi + (k1 + 2*k2 + 2*k3 + k4)/6
        t.append(ti1)
        w.append(wi1)
    return np.array(t), np.array(w)


# Find numerical approximations and exact solution
t, w_euler = euler(disease_spread, 0, 30, 1000)
t, w_modified_euler = modified_euler(disease_spread, 0, 30, 1000)
t, w_midpoint = midpoint(disease_spread, 0, 30, 1000)
t, w_runge_kutta = runge_kutta(disease_spread, 0, 30, 1000)
y = exact_solution(t, 1000)

print('Euler: w(30) =', w_euler[-1])
print('Modified Euler: w(30) =', w_modified_euler[-1])
print('Midpoint: w(30) =', w_midpoint[-1])
print('Runge-Kutta: w(30) =', w_runge_kutta[-1])
print('Exact solution: y(30) =', y[-1])

print('\n')

# Find absolute errors
abs_err_euler = np.abs(w_euler - y)
abs_err_modified_euler = np.abs(w_modified_euler - y)
abs_err_midpoint = np.abs(w_midpoint - y)
abs_err_runge_kutta = np.abs(w_runge_kutta - y)

print('Absolute error for Euler:', abs_err_euler[-1])
print('Absolute error for Modified Euler:', abs_err_modified_euler[-1])
print('Absolute error for Midpoint:', abs_err_midpoint[-1])
print('Absolute error for Runge-Kutta:', abs_err_runge_kutta[-1])

print('\n')

# Find relative errors
relative_err_euler = abs_err_euler/np.abs(y)
relative_err_modified_euler = abs_err_modified_euler/np.abs(y)
relative_err_midpoint = abs_err_midpoint/np.abs(y)
relative_err_runge_kutta = abs_err_runge_kutta/np.abs(y)

print('Relative error for Euler:', relative_err_euler[-1])
print('Relative error for Modified Euler:', relative_err_modified_euler[-1])
print('Relative error for Midpoint:', relative_err_midpoint[-1])
print('Relative error for Runge-Kutta:', relative_err_runge_kutta[-1])

# Find absolute errors, varying h
hs = np.linspace(0.0005, 0.5, 100)
abs_euler_h = []
abs_modified_euler_h = []
abs_midpoint_h = []
abs_runge_kutta_h = []
for h in hs:
    th, euler_h = euler(disease_spread, 0, 100, 1000, h)
    th, modified_euler_h = modified_euler(disease_spread, 0, 100, 1000, h)
    th, midpoint_h = midpoint(disease_spread, 0, 100, 1000, h)
    th, runge_kutta_h = runge_kutta(disease_spread, 0, 100, 1000, h)
    yh = exact_solution(th, 1000)
    abs_euler_h.append(abs(euler_h[-1] - yh[-1]))
    abs_modified_euler_h.append(abs(modified_euler_h[-1] - yh[-1]))
    abs_midpoint_h.append(abs(midpoint_h[-1] - yh[-1]))
    abs_runge_kutta_h.append(abs(runge_kutta_h[-1] - yh[-1]))

# Plot results
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

ax1.plot(t, w_euler, label='Euler',linestyle='--',color='g')
ax1.plot(t, w_modified_euler, label='Euler Modificado',linestyle='--',color='b')
ax1.plot(t, w_runge_kutta, label='Runge-Kutta',linestyle='--',color='r')
ax1.plot(t, y, label='Solución Exacta',linestyle='--',color='m')
ax1.set_title('$w(t)$ vs $t$')
ax1.set_xlabel('Tiempo(Dias)')
ax1.set_ylabel('Poblacion infectada')
ax1.legend()

ax2.plot(t, abs_err_euler, label='Euler',linestyle='--',color='g')
ax2.plot(t, abs_err_modified_euler, label='Euler Modificado',linestyle='--',color='b')
ax2.plot(t, abs_err_runge_kutta, label='Runge-Kutta',linestyle='--',color='r')
ax2.set_title('$|w(t)-y(t)|$ vs $t$')
ax2.set_xlabel('Tiempo(Dias)')
ax2.set_ylabel('Error Absoluto')
ax2.legend()

ax3.plot(t, relative_err_euler, label='Euler',linestyle='--',color='g')
ax3.plot(t, relative_err_modified_euler, label='Euler Modificado',linestyle='--',color='b')
ax3.plot(t, relative_err_runge_kutta, label='Runge-Kutta',linestyle='--',color='r')
ax3.set_title('$\\frac{|w(t)-y(t)|}{|y(t)|}$ vs $t$')
ax3.set_xlabel('Tiempo (Dias)')
ax3.set_ylabel('Error Relativo')
ax3.legend()

ax4.plot(hs, abs_euler_h, label='Euler',linestyle='--',color='g')
ax4.plot(hs, abs_modified_euler_h, label='Euler Modificado',linestyle='--',color='b')
ax4.plot(hs, abs_runge_kutta_h, label='Runge-Kutta',linestyle='--',color='r')
ax4.set_title('$|w(t)-y(t)|$ vs $h$')
ax4.set_xlabel('h')
ax4.set_ylabel('Error Absoluto')
ax4.legend()

plt.show()
