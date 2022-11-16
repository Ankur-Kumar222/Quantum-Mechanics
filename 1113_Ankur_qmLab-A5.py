import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import pandas as pd

def f(x, e):
    return 2*(e - (1/2)*(x**2))

def e(n, delta = 0):
    return n + (1/2) + delta

def numerov(func, u0, x_max, points, e, n, delta):
    
    x_range = np.linspace(0, x_max, points)
    h = x_range[1] - x_range[0]
    u_values = np.zeros(len(x_range))
    u_values[0] = u0

    c_values = np.ones(len(x_range)) + np.multiply((h**2)/12, func(x_range, e(n, delta)))

    if u0 == 0:
        u_values[1] = u0 + h
    else:
        u_values[1] = ((6 - 5*c_values[0])/c_values[1])*u0


    for i in range(1, len(x_range)-1):
        u_values[i+1] = (1/c_values[i+1])*((12-10*c_values[i])*u_values[i] - c_values[i-1]*u_values[i-1])

    extended_u = []

    if u0 == 0:

        for i in range(1, len(x_range)):
            extended_u.append(-1*(u_values[-i]))

        for i in range(0, len(x_range)):
            extended_u.append(u_values[i])

    else:
        for i in range(1, len(x_range)):
            extended_u.append(1*(u_values[-i]))

        for i in range(0, len(x_range)):
            extended_u.append(u_values[i])

    extended_u = np.array(extended_u)

    extended_x_vals = np.linspace(-x_max, x_max, 2*points-1)

    norm = (simps((extended_u)**2, extended_x_vals))
    u_list = (extended_u)/np.sqrt(norm)


    return extended_x_vals, u_list

delta_e = [1e-2, 1e-4, 1e-6, 1e-8]

#Ground State
for i in range(0, len(delta_e)):
    x_vals, result = numerov(f, 1, np.sqrt(1), 100, e, 0, delta_e[i])
    plt.scatter(x_vals, result, label = f'delta_e_{i}', s = 5)

x_vals_analyitc, result_analytic = numerov(f, 1, np.sqrt(1), 100, e, 0, 0)
plt.plot(x_vals_analyitc, result_analytic, label = 'Analytical Solution')
plt.title('Ground State - Variation of Delta_e')
plt.xlabel(r'$\xi$')
plt.ylabel(r'$U(\xi)$')
plt.grid()
plt.legend()
plt.show()

#First Three Excited States
initial_conds = [0, -1]
for i in range(1, len(initial_conds)+1):

    x_vals, result = numerov(f, initial_conds[i-1], np.sqrt(2*i+1), 100, e, i, 1e-6)
    plt.scatter(x_vals, result, label = 'Numerov Solution', s = 10, color = 'red')
    x_vals_analytic, result_analytic = numerov(f, initial_conds[i-1], np.sqrt(2*i+1), 100, e, i, 0)
    plt.plot(x_vals_analytic, result_analytic, label = 'Analytical Solution')
    
    plt.title(f'N = {i} ')
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$U(\xi)$')
    plt.legend()
    plt.grid()
    plt.show()


x_vals, result = numerov(f, 0, np.sqrt(7), 100, e, 3, 1e-6)
plt.scatter(x_vals, -1*result, label = 'Numerov Solution', s = 10, color ='red')
x_vals_analytic, result_analytic = numerov(f, 0, np.sqrt(7), 100, e, 3, 0)
plt.plot(x_vals_analytic, -1*result_analytic, label = 'Analytical Solution')

plt.title('N = 3')
plt.xlabel(r'$\xi$')
plt.ylabel(r'$U(\xi)$')
plt.legend()
plt.grid()
plt.show()

#Probability Densities

initial_conds = [1, 0, -1]
for i in range(0, len(initial_conds)):

    x_vals, result = numerov(f, initial_conds[i], np.sqrt(2*i+1), 100, e, i, 1e-6)
    plt.scatter(x_vals, result**2, label = f'Numerov Solution N = {i}', s = 10, color = 'red')
    x_vals_analytic, result_analytic = numerov(f, initial_conds[i], np.sqrt(2*i+1), 100, e, i, 0)
    plt.plot(x_vals_analytic, result_analytic**2, label = f'Analytical Solution N = {i}')
    
    
x_vals, result = numerov(f, 0, np.sqrt(7), 100, e, 3, 1e-6)
plt.scatter(x_vals, (-1*result)**2, label = f'Numerov Solution N = 3', s = 10, color ='red')
x_vals_analytic, result_analytic = numerov(f, 0, np.sqrt(7), 100, e, 3, 0)
plt.plot(x_vals_analytic, (-1*result_analytic)**2, label = f'Analytical Solution N = 3')

plt.title('Probability Densities')
plt.xlabel(r'$\xi$')
plt.ylabel(r'|$U^2(\xi)$|')
plt.legend()
plt.grid()
plt.show()

#Energy Values

def energy(n, omega, delta = 0):

    e = 1.6e-19
    h_cut =  1.05457182e-34
    return ((n + 1/2 + delta) * h_cut * omega)/e

energy_calc = []
energy_analytic = []
n = []

for i in range(0,4):
    energycalc = energy(i, 5.5e14, delta = 1e-6)
    energy_calc.append(energycalc)

    energyanalytic = energy(i, 5.5e14, delta = 0)
    energy_analytic.append(energyanalytic)

    n.append(i)

data = {

  'n': n,
  'Calculated Energies(eV)': energy_calc,
  'Analytical Energies(eV)': energy_analytic
  
}

df = pd.DataFrame(data)
print(df)

#PROBABILITY

x_vals_prob, result_prob =  numerov(f, 1, np.sqrt(9), 150, e, 0, 0)

slice_x = x_vals_prob[99:200]
slice_result = result_prob[99:200]

Probability = simps((slice_result)**2, slice_x)

print('Probability = ', 1 - Probability)
