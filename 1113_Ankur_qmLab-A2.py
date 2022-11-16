import numpy as np
import matplotlib.pyplot as plt
from IVP_Module import *
from scipy.integrate import simps

#Constants 

h_cut = 1.05457182e-34
m = 9.1e-31
L = 2e-10

k = (h_cut**2)/(2*m*(L**2))


def func_x(x,y_vec,e):
    ans_vec=np.zeros((2))
    ans_vec[0]=y_vec[1]
    ans_vec[1]= -(e**2)*(np.pi**2)*y_vec[0]
    return ans_vec


x_vals = np.linspace(-1/2,1/2,1000)

for i in range(1,11):
    initial_conds = [0,i]
    result1 = RK_4(func_x, initial_conds, x_vals, e = 8)
    norm = simps((result1.T[0])**2, x_vals)
    

    plt.plot(x_vals, (result1.T[0])/np.sqrt(norm), label = f'u(-1/2)={i}')

plt.plot(x_vals, np.sqrt(2)*np.sin(8*np.pi*x_vals), label = 'Analytic Solution')
plt.grid()
plt.legend()
plt.title('Solution of Schrodinger Eq. for e = 8')
plt.xlabel(r'$\xi$')
plt.ylabel(r'$U(\xi)$')
plt.show()

for i in range(1,11):
    initial_conds = [0,i]
    result2 = RK_4(func_x, initial_conds, x_vals, e = 11)
    norm = simps((result2.T[0])**2, x_vals)

    plt.plot(x_vals, (result2.T[0])/np.sqrt(norm), label = f'u(-1/2)={i}')

plt.plot(x_vals, np.sqrt(2)*-np.cos(11*np.pi*x_vals), label = 'Analytic Solution')
plt.grid()
plt.legend()
plt.title('Solution of Schrodinger Eq. for e = 11')
plt.xlabel(r'$\xi$')
plt.ylabel(r'$U(\xi)$')
plt.show()


initial_conds = [0,1]

e = 0.89
x = 1
while abs(x) > 0.5e-10:

    e = e + 0.01
    result = RK_4(func_x, initial_conds, x_vals, e)
    plt.plot(x_vals, (result.T[0]), label = f'e={e}')

    x = result.T[0][-1]

plt.grid()
plt.legend()
plt.title('Solution of Schrodinger Eq. for variable e(Ground State)')
plt.xlabel(r'$\xi$')
plt.ylabel(r'$U(\xi)$')
plt.show()

print('E1=')
print((e*k*(np.pi**2))/1.6e-19, 'eV')



e = 1.89
x = 1
while abs(x) > 0.5e-10:

    e = e + 0.01
    result = RK_4(func_x, initial_conds, x_vals, e)
    plt.plot(x_vals, (result.T[0]), label = f'e={e}')

    x = result.T[0][-1]

plt.grid()
plt.legend()
plt.title('Solution of Schrodinger Eq. for variable e(First Excited State)')
plt.xlabel(r'$\xi$')
plt.ylabel(r'$U(\xi)$')
plt.show()

print('E2=')
print(((e**2)*k*(np.pi**2))/1.6e-19, 'eV')




