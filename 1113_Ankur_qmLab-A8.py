from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

def matrix_formation(a, b, N, f):

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
        potential = f(X)
        
    diagonal = np.diag(d, k = 0)
    off_diag_l = np.diag(l[:-1], k = -1)
    off_diag_u = np.diag(u[:-1], k = 1)

    V = np.diag(potential, k = 0)

    matrix = diagonal + off_diag_l + off_diag_u + V

    return matrix, X

def pot(x):
    potential = []
    for j in x:
        if j >= 1/2 and j <= -1/2:
            potential.append(np.inf)

        else:
            potential.append(0)
    return potential

def analytic(x, n):
    L = 1
    
    if n % 2 == 0:
        return x, (-1)*np.sqrt(2/L)*np.sin((n*np.pi*x)/L)

    else:
        return x, (1)*np.sqrt(2/L)*np.cos((n*np.pi*x)/(L))

def normalize(x, u):

    norm = simps(u**2, x)

    return u/np.sqrt(norm)

matrix, X = matrix_formation(-1/2, 1/2, 100, pot)

e , vec = eigh(matrix)

print("First Ten Eigen Values")
print(e[:10])


for i in range(1, 5):

    plt.scatter(X, normalize(X, vec.T[i-1]), label = 'Numerical Solution', color = 'red', s = 10)

    if i == 3 or i == 4:

        plt.plot(X, (-1)*analytic(X, i)[1], label = 'Analytical Solution')
        plt.grid()
        plt.legend()
        plt.title(f'Wave Function for n={i}')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.show()
    
    else:
        plt.plot(X, (1)*analytic(X, i)[1], label = 'Analytical Solution')
        plt.grid()
        plt.legend()
        plt.title(f'Wave Function for n={i}')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.show()

for i in range(1, 5):

    plt.scatter(X, (normalize(X, vec.T[i-1]))**2, label = f'Numerical Probability Density, i={i}', s = 10)
    
    plt.plot(X, (analytic(X, i)[1])**2, label = f'Analytical Probability Density, i={i}')
plt.grid()
plt.legend()
plt.title(f'Probability Density')
plt.xlabel('x')
plt.ylabel('|u(x)^2|')
plt.show()