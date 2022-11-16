import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IVP_Module import *
from scipy.integrate import simps
from scipy.optimize import newton
from scipy.stats import linregress


def energies_eV(e, m, L, h_cut = 1.05457182e-34):

    return (e*((h_cut**2)/(2*m*(L**2))))/(1.6e-19)


def func_x(x,y_vec,e):
    ans_vec=np.zeros((2))
    ans_vec[0]=y_vec[1]
    ans_vec[1]= -(e)*y_vec[0]
    return ans_vec

e_range = np.linspace(0,300,300)
x_vals = np.linspace(-1/2,1/2,1000)

U_r = []

for i in e_range:
    initial_conds = [0,1]
    result = RK_4(func_x, initial_conds,x_vals , i)
    norm = simps((result.T[0])**2, x_vals)
    norm_result = (result.T[0])/np.sqrt(norm)
    U_r.append(norm_result.T[-1])


plt.scatter(e_range, U_r)
plt.grid()
plt.xlabel('e')
plt.ylabel('U(r)')
plt.title('U(r) Vs e')
plt.show()

index_1 = []
index_2 = []

for i in range(len(U_r)-1):
    if U_r[i] * U_r[i+1] < 0:
        index_1.append(i)
        index_2.append(i+1)
    else:
        pass


zeros = []

def U_final(e):
    result = RK_4(func_x, initial_conds,x_vals , e)
    norm = simps((result.T[0])**2, x_vals)
    norm_result = (result.T[0])/np.sqrt(norm)

    return norm_result.T[-1]

for i, j in zip(index_1,index_2):
    zero = newton(U_final, x0 = i, x1 = j, fprime = None)
    zeros.append(zero)


for i in range(len(zeros)):
    result = RK_4(func_x, initial_conds,x_vals , zeros[i])
    norm = simps((result.T[0])**2, x_vals)
    norm_result = (result.T[0])/np.sqrt(norm)

    plt.plot(x_vals, norm_result, label = f'e = {zeros[i]}')
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$\psi(\xi)$')
    plt.title('First Five States')
    plt.legend()
    plt.grid()
plt.show()


curve = linregress([i**2 for i in range(1, len(zeros)+1)], zeros)

print("Slope = ", curve[0])

n_sq = [i**2 for i in range(1, len(zeros)+1)]
plt.scatter(n_sq, zeros, label ='e values', color = 'green')
plt.plot(n_sq, curve[0]*np.array([i**2 for i in range(1, len(zeros)+1)]),label = 'Curve Fitted Line')
plt.xlabel(r'$n^2$')
plt.ylabel('e')
plt.title(r'e Vs $n^2$')
plt.grid()
plt.legend()
plt.show()


for i in range(len(zeros)):
    result = RK_4(func_x, initial_conds,x_vals , zeros[i])
    norm = simps((result.T[0])**2, x_vals)
    norm_result = (result.T[0])/np.sqrt(norm)
    norm_result_sq = norm_result**2

    plt.plot(x_vals, norm_result_sq, label = f'n = {i}')
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$|\psi^2(\xi)|$')
    plt.title('First Five Probability Densities')
    plt.legend()
    plt.grid()
plt.show()

calc_energy_e = []
analytical_energy_e = []
n_state = []

for i in range(0,5):

    energy_calc = energies_eV(zeros[i], 9.1e-31, 5e-10)
    calc_energy_e.append(energy_calc)

    analytical_energy = energies_eV(((i+1)**2)*(np.pi**2), 9.1e-31, 5e-10)
    analytical_energy_e.append(analytical_energy)

    n_state.append(i+1)

print('Well of Width 5 Angstrom(Electron)')
print('------------------------------------------')
data = {

  'n': n_state,
  'Calculated Energy(eV)': calc_energy_e,
  'Analytical Energy(eV)': analytical_energy_e, 

}

df = pd.DataFrame(data)
print(df)

calc_energy_e = []
analytical_energy_e = []
n_state = []

for i in range(0,5):

    energy_calc = energies_eV(zeros[i], 9.1e-31, 10e-10)
    calc_energy_e.append(energy_calc)

    analytical_energy = energies_eV(((i+1)**2)*(np.pi**2), 9.1e-31, 10e-10)
    analytical_energy_e.append(analytical_energy)

    n_state.append(i+1)

print('Well of Width 10 Angstrom(Electron)')
print('------------------------------------------')
data = {

  'n': n_state,
  'Calculated Energy(eV)': calc_energy_e,
  'Analytical Energy(eV)': analytical_energy_e, 

}

df = pd.DataFrame(data)
print(df)

calc_energy_p = []
analytical_energy_p = []
n_state = []

for i in range(0,5):

    energy_calc = energies_eV(zeros[i], 1.67e-27, 5e-15)
    calc_energy_p.append(energy_calc)

    analytical_energy = energies_eV(((i+1)**2)*(np.pi**2), 1.67e-27, 5e-15)
    analytical_energy_p.append(analytical_energy)

    n_state.append(i+1)

print('Well of Width 5 Fermimeter(Proton)')
print('------------------------------------------')
data = {

  'n': n_state,
  'Calculated Energy(eV)': calc_energy_p,
  'Analytical Energy(eV)': analytical_energy_p, 

}

df = pd.DataFrame(data)
print(df)

ground_state = RK_4(func_x, initial_conds,x_vals , zeros[0])
norm1 = simps((ground_state.T[0])**2, x_vals)
norm_result1 = (ground_state.T[0])/np.sqrt(norm1)

norm2 = simps((ground_state.T[1])**2, x_vals)
norm_result2 = (ground_state.T[1])/np.sqrt(norm2)

U = np.array(norm_result1)
U_sq = np.array(norm_result1)**2
U_sq_xi = U_sq*x_vals

xi_sq = np.array(x_vals)**2
U_sq_xi_sq = U_sq*xi_sq

u_dash = np.array(norm_result2)
uu_dash = U*u_dash

p_sq = U_sq*zeros[0]

h_cut = 1.05457182e-34
expect_x = simps(U_sq_xi, x_vals)
expect_x_sq = simps(U_sq_xi_sq,x_vals)
expect_p = simps(uu_dash,x_vals)*h_cut
expect_p_sq = simps(p_sq,x_vals)*(h_cut**2)

sigma_x = np.sqrt(expect_x_sq - (expect_x**2))
sigma_p = np.sqrt(expect_p_sq - (expect_p**2))

if sigma_x*sigma_p > h_cut/2:

    print('Uncertainity Principle is Satisfied')

else:
    print('Uncertainity Principle Fails')

#Probability
new_x_vals = x_vals[249:749]
new_U = U[249:749]

Probability = simps(new_U**2, new_x_vals)
print('Probability = ', Probability*100, '%')
















    

