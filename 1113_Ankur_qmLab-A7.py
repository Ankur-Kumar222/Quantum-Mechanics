import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
import pandas as pd
from scipy.optimize import newton, fsolve


def alpha(x, e):

    return 2*(e - (1/2)*(x**2))

def numerov(f, x_range, u0, e, key):

    h = x_range[1] - x_range[0]
    if key == 1:
        h = -h
    
    U_i = np.zeros(len(x_range))
    C_i = np.ones(len(x_range)) + np.multiply((h**2)/12, f(x_range, e))

    U_i[0] = u0

    if u0 == 0:
        U_i[1] = u0 + h
    else:
        U_i[1] = ((6 - 5*C_i[0])/C_i[1])*u0

    for i in range(1, len(x_range)-1):
        U_i[i+1] = (1/C_i[i+1])*((12-10*C_i[i])*U_i[i] - C_i[i-1]*U_i[i-1])

    return x_range, U_i, C_i

def turning_points(x_range, e, alpha):

    q = alpha(x_range, e)
    for i in range(1, len(x_range)):

        if q[i-1]*q[i] < 0:
            return x_range[i], i


def e_shooting(n_node, E_min, E_max, x_range, initial_cond):

    N_node = 100

    while N_node != n_node:

        I = []
        E = (E_min+E_max)/2

        u = numerov(alpha, x_range, initial_cond, E, key = 0)[1]

        
        for i in range(1, len(u)):

            if (u[i-1]*u[i]) < 0:

                I.append(i)
        
        N_node = len(I)
        
        if N_node > int((n_node)/2):

            E_max = E

        elif N_node < int((n_node)/2):

            E_min = E
        else:

            return E, E+0.1

        

def phi(e):

    turning_pt = turning_points(x_range, e, alpha)

    xL, uL, cL = numerov(alpha, np.linspace(0,turning_pt[0],len(x_range)), 1, e, key = 0)

    xR, uR, cR = numerov(alpha, np.linspace(x_range[-1], turning_pt[0], len(x_range)), 0, e, key = 1)

    uR_new = (uR/uR[-1])*uL[-1]

    return (uL[-2]+uR_new[-2]-((12*cL[-1]-10)*uL[-1]))/(x_range[1]-x_range[0])



def extending_func(x_range, n_node, initial_cond):

    guess1, guess2= e_shooting(n_node, 0, 5, x_range, initial_cond)
    zero = newton(phi, x0 = guess1, x1 = guess2, fprime = None)
    
    turning_pt = turning_points(x_range, zero, alpha)

    
    uR = numerov(alpha, np.linspace(0,turning_pt[0],len(x_range)), 1, zero, key = 0)

    uL = numerov(alpha, np.linspace(x_range[-1], turning_pt[0], len(x_range)), 0, zero, key = 1)


    uL_new = (uL[1]/uL[1][-1])*uR[1][-1]

    xLL = np.flip(uL[0])
    uLL = np.flip(uL_new)

    u_final = np.concatenate((uR[1], uLL))
    x_final = np.concatenate((uR[0], xLL))
   
    if u_final[0] == 0:

        flipped_u = -np.flip(u_final)
        extended_u = np.concatenate((flipped_u[:-1], u_final))

        flipped_x  = -np.flip(x_final)
        extended_x = np.concatenate((flipped_x[:-1], x_final))

    else:
        flipped_u  = np.flip(u_final)
        extended_u = np.concatenate((flipped_u[:-1], u_final))

        flipped_x  = -np.flip(x_final)
        extended_x = np.concatenate((flipped_x[:-1], x_final))

    return extended_x, extended_u, zero


x_range = np.linspace(0,2,100)

x1, u1, z1= extending_func(x_range, 0, 1)

n = [1]
e = [0.5]

data = {

  'n': n,
  'Calculated eigen value': z1,
  'Analytical eigen value': e,

}

df = pd.DataFrame(data)
print(df)

plt.scatter(x1, u1, label = 'N=1', s = 5)
plt.title('Wave Function for n = 1')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.grid()
plt.legend()
plt.show()

plt.scatter(x1, u1**2, label = 'N=1', s = 5)
plt.title('Probability Density for n = 1')
plt.xlabel('x')
plt.ylabel('|u(x)^2|')
plt.grid()
plt.legend()
plt.show()
















