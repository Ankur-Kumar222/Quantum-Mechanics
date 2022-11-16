from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import pandas as pd


def matrix_formation(a, b, N, f, alpha):

    x_range = np.linspace(a, b, N+2)
    h = x_range[1] - x_range[0]

    X = x_range[1: -1]
    g = f(X, alpha)

    l = np.zeros(len(X))
    d = np.zeros(len(X))
    u = np.zeros(len(X))
    
    for i in range(0, len(X)):
        d[i] = -(-2/h**2) - 1*g[i]
        l[i] = -(1/h**2)
        u[i] = -(1/h**2)
        
    diagonal = np.diag(d, k = 0)
    off_diag_l = np.diag(l[:-1], k = -1)
    off_diag_u = np.diag(u[:-1], k = 1)

    matrix = diagonal + off_diag_l + off_diag_u

    return matrix, X

def pot_col(x, alpha):
    potential = []
    for j in x:
            val = 2*((1/j))
            potential.append(val)

    return potential

def pot_screen(x, alpha):
    potential = []
    for j in x:
            val = 2*((1/j)*np.exp(-j/alpha))
            potential.append(val)

    return potential


def normalize(x, u): 
    return u/np.sqrt(simps(u**2, x))

def V_r(x):
    
    return -2/x

def V_screening(x, alpha):

    return (-2/x)*np.exp(-x/alpha)


r_range = np.linspace(0.1, 2, 500)
alpha_vals = [2, 5, 10, 20, 100]

plt.plot(r_range, V_r(r_range), label = 'V(Coulomb) for l=0 ')
for i in alpha_vals:
    plt.plot(r_range, V_screening(r_range, i), label = f'V(screening) for alpha = {i}')

plt.title('Potential Plots')
plt.xlabel(r"$\xi$")
plt.ylabel("V")
plt.legend()
plt.grid()
plt.show()

ground_state_vals = []

for i in alpha_vals:

    matrix, X = matrix_formation(0, 200, 1000, pot_screen, i)
    e , vec = eigh(matrix)

    print(f'Energy Eigen Values(alpha={i}) = ', e[:5])

    ground_state_vals.append(e[0])


ground_state_energy = np.multiply(ground_state_vals, 13.6)

data = {

  'alpha': alpha_vals,
  'Ground_State_Energy(eV)': ground_state_energy

}

df = pd.DataFrame(data)
print(df)


x_range = np.linspace(0, 20, 1000)

matrix1, X1 = matrix_formation(0, 20, 1000, pot_col, 0)
e1 , vec1 = eigh(matrix1)
norm1 = normalize(X1, vec1.T[0])

plt.scatter(X1, norm1,s = 5, label = 'Coulomb Potential')

for i in alpha_vals:

    matrix2, X2 = matrix_formation(0, 20, 1000, pot_screen, i)
    e2 , vec2 = eigh(matrix2)
    norm2 = normalize(X2, vec2.T[0])
    plt.plot(X2, norm2, label = f'Screening Potential for alpha = {i}')

plt.title('Ground State WaveFunction')
plt.xlabel(r"$\xi$")
plt.ylabel("K_nl")
plt.legend()
plt.grid()
plt.show()


matrix1, X1 = matrix_formation(0, 20, 1000, pot_col, 0)
e1 , vec1 = eigh(matrix1)
norm1 = normalize(X1, vec1.T[0])

plt.scatter(X1, norm1**2,s = 5, label = 'Coulomb Potential')

for i in alpha_vals:

    matrix2, X2 = matrix_formation(0, 20, 1000, pot_screen, i)
    e2 , vec2 = eigh(matrix2)
    norm2 = normalize(X2, vec2.T[0])
    plt.plot(X2, norm2**2, label = f'Screening Potential for alpha = {i}')

plt.title('Ground State Probability Density')
plt.xlabel(r"$\xi^2$")
plt.ylabel("K_nl")
plt.legend()
plt.grid()
plt.show()

plt.plot(alpha_vals, ground_state_energy)
plt.scatter(alpha_vals, ground_state_energy, color ="red")
plt.title('Ground State Energy Vs Alpha')
plt.xlabel(r"Alpha")
plt.ylabel("E0(eV)")
plt.grid()
plt.show()