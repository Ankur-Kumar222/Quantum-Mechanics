from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.special import assoc_laguerre


def matrix_formation(a, b, N, f, l):

    x_range = np.linspace(a, b, N+2)
    h = x_range[1] - x_range[0]

    X = x_range[1: -1]
    g = f(X, l)

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

def pot(x, l):
    potential = []
    for j in x:
            val = 2*(1/j- l*(l+1)/(2*j**2))
            potential.append(val)

    return potential


def normalize(x, u): 
    return u/np.sqrt(simps(u**2, x))


def V_r(x):
    
    return -2/x

def V_eff(x, l):

    return -2/x + ((l*(l+1))/(x**2))

def psi_R(r, n, l):
    
    return ((2*r/n)**(l)*assoc_laguerre(2*r/n, n-l-1, 2*l-1))/np.exp(r/n)*r


r_range = np.linspace(0.1, 2, 500)
l = [1,2,3]

plt.plot(r_range, V_r(r_range), label = 'V(r)')
for i in l:
    plt.plot(r_range, V_eff(r_range, i), label = f'V(eff) for l = {i}')

plt.title('V(r) and V(eff)')
plt.xlabel("r'")
plt.ylabel("V")
plt.legend()
plt.grid()
plt.show()

matrix, X = matrix_formation(0, 200, 1000, pot, 0)

e , vec = eigh(matrix)

print('first ten energy eigenvalues for l=0')
print(e[:10])

for i in range(0,4):

    norm_vec = normalize(X, vec.T[i])
    plt.scatter(X[:300], norm_vec[:300], label = f'n={i+1}', s = 10)

plt.title('First Four Radial Wavefunctions for l=0')
plt.xlabel("r'")
plt.ylabel("K_nl")
plt.legend()
plt.grid()
plt.show()


for i in range(0,2):
    matrix, X = matrix_formation(0, 300, 1000, pot, i+1)

    e , vec = eigh(matrix)

    print(f'first ten energy eigenvalues for l={i+1}')
    print(e[:10])

#Density Plot 1

matrix1, X1 = matrix_formation(0, 200, 1000, pot, 0)

e , vec = eigh(matrix1)

norm_vec = normalize(X1, vec.T[0])
plt.scatter(X1[:100], norm_vec[:100]**2, s = 10, label = 'l=0')
plt.title('Probability Density for n=1')
plt.xlabel("r'")
plt.ylabel("K_nl^2")
plt.legend()
plt.grid()
plt.show()

#Density Plot 2

matrix2, X2 = matrix_formation(0, 200, 1000, pot, 1)

e , vec2 = eigh(matrix2)

norm_vec = normalize(X1, vec.T[1])
plt.scatter(X1[:200], norm_vec[:200]**2, s = 10, label = 'l=0')

norm_vec = normalize(X2, vec2.T[1])
plt.scatter(X1[:200], norm_vec[:200]**2, s = 10, label = 'l=1')

plt.title('Probability Density for n=2')
plt.xlabel("r'")
plt.ylabel("K_nl^2")
plt.legend()
plt.grid()
plt.show()

#Density Plot 3

matrix3, X3 = matrix_formation(0, 200, 1000, pot, 2)

e , vec3 = eigh(matrix3)

norm_vec = normalize(X1, vec.T[2])
plt.scatter(X1[:400], norm_vec[:400]**2, s = 10, label = 'l=0')

norm_vec = normalize(X2, vec2.T[2])
plt.scatter(X2[:400], norm_vec[:400]**2, s = 10, label = 'l=1')

norm_vec = normalize(X3, vec3.T[2])
plt.scatter(X3[:400], norm_vec[:400]**2, s = 10, label = 'l=2')

plt.title('Probability Density for n=3')
plt.xlabel("r'")
plt.ylabel("K_nl^2")
plt.legend()
plt.grid()
plt.show()
