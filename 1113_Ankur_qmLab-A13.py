from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import pandas as pd

def pot(x, alpha):
    return x**2 + (2/3)*alpha*x**3

def matrix_formation(a, b, N, f, alpha):

    x_range = np.linspace(a, b, N+1)
    h = x_range[1] - x_range[0]

    X = x_range[1: -1]

    l = np.zeros(len(X))
    d = np.zeros(len(X))
    u = np.zeros(len(X))
    potential = np.zeros(len(X))
    
    for i in range(0, len(X)):
        d[i] = -(-2/h**2)
        l[i] = -(1/h**2)
        u[i] = -(1/h**2)
        potential = f(X, alpha)
        
    diagonal = np.diag(d, k = 0)
    off_diag_l = np.diag(l[:-1], k = -1)
    off_diag_u = np.diag(u[:-1], k = 1)

    V = np.diag(potential, k = 0)

    matrix = diagonal + off_diag_l + off_diag_u + V

    return matrix, X

def perturbation(n, alpha):
    return 2*n + 1 - (1/8)*(alpha**2)*(15*(2*n+1)**2+7)

def normalize(x_vals, y_vals):
    norm = simps(abs(y_vals**2), x_vals)
    norm_result = y_vals/np.sqrt(norm)

    return x_vals, norm_result



x_values = np.linspace(-5, 5, 500)
alpha_vals = [0, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]

for i in alpha_vals:
    plt.plot(x_values, pot(x_values, i), label = f'Alpha={i}')
plt.legend()
plt.grid()
plt.title(r'Potential Energy as a function of $\xi$')
plt.xlabel(r'\xi')
plt.ylabel(r'V(\xi)')
plt.show()


n_vals = np.array([0,1,2,3,4,5,6,7,8,9])


for i in alpha_vals:

    matrix, X = matrix_formation(-5, 5, 500, pot, i)

    e , vec = eigh(matrix)

    perturbation_temp = perturbation(n_vals, i)

    data = {

  'n': n_vals,
  'Calculated eigen value': e[:10],
  'Perturbation eigen value': perturbation_temp,

    }

    df = pd.DataFrame(data)
    print(f'Alpha={i}')
    print('------------------------------------------------------')
    print('Ground State Energy = ', 197.3/2*np.sqrt(100/940)*e[0], 'MeV')
    print(df)

    plt.plot(n_vals , e[:10], label = f'alpha={i}')

plt.legend()
plt.grid()
plt.title(r'E(n) as a function of n')
plt.xlabel(r'E(n)')
plt.ylabel(r'n')
plt.show()


for i in alpha_vals:

    for j in range(0,5):

        matrix, X = matrix_formation(-5, 5, 500, pot, i)

        e , vec = eigh(matrix)

        plt.plot(X, normalize(X, vec.T[j])[1], label = f'n = {j}')
    plt.legend()
    plt.grid()
    plt.title(f'Wave Function Alpha={i}')
    plt.xlabel(r'\xi')
    plt.ylabel(r'U(\xi)')
    plt.show()

for i in alpha_vals:

    for j in range(0,5):

        matrix, X = matrix_formation(-5, 5, 500, pot, i)

        e , vec = eigh(matrix)

        plt.plot(X, (normalize(X, vec.T[j])[1])**2, label = f'n = {j}')
    plt.legend()
    plt.grid()
    plt.title(f'Probability Density Alpha={i}')
    plt.xlabel(r'\xi')
    plt.ylabel(r'U(\xi)')
    plt.show()


for i in range(0,2):

    for j in alpha_vals:

        matrix, X = matrix_formation(-5, 5, 500, pot, j)

        e , vec = eigh(matrix)

        plt.plot(X, normalize(X, vec.T[i])[1], label = f'alpha = {j}')
    plt.legend()
    plt.grid()
    plt.title(f'Wave Function n={i}')
    plt.xlabel(r'\xi')
    plt.ylabel(r'U(\xi)')
    plt.show()

for i in range(0,2):

    for j in alpha_vals:

        matrix, X = matrix_formation(-5, 5, 500, pot, j)

        e , vec = eigh(matrix)

        plt.plot(X, (normalize(X, vec.T[i])[1])**2, label = f'alpha = {j}')
    plt.legend()
    plt.grid()
    plt.title(f'Probability Density n={i}')
    plt.xlabel(r'\xi')
    plt.ylabel(r'U(\xi)')
    plt.show()