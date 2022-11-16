import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import simps
from scipy.special import assoc_laguerre


def alpha(x, e, l):
    return e + 2/x - (l*(l+1)/x**2)


def numerov(func, u0, x_range, e, l):

    h = x_range[1] - x_range[0]
    h = -h
   
    u_values = np.zeros(len(x_range))
    u_values[0] = u0
    u_values[1] = u0 + h

    c_values = np.ones(len(x_range)) + np.multiply((h**2)/12, func(x_range, e, l))
    for i in range(1, len(x_range)-1):
        u_values[i+1] = (1/c_values[i+1])*((12-10*c_values[i])*u_values[i] - c_values[i-1]*u_values[i-1])

    return u_values


e_range = np.linspace(-2,0,3000)
x_range = np.linspace(10e-14, 200, 1000)

def U_r_vals(l):
    U_r = []

    for i in e_range:

        result = numerov(alpha, 0, x_range, i, l)
        norm = simps((result)**2, x_range)
        norm_result = (result)/np.sqrt(norm)
        U_r.append(norm_result[-1])

    return U_r


def index(l):

    U_r = U_r_vals(l)

    index_turn = []
    
    for i in range(len(U_r)-1):
        if U_r[i] * U_r[i+1] < 0:

            index_turn.append(i)
        else:

            pass

    return index_turn


l_vals = [0, 1, 2]

def phi(e, x_vals, l):

    u = numerov(alpha, 0, x_vals, e, l)

    return u[-1]

x_range_1 = np.linspace(10e-14, 100, 1000)
x_range_2 = np.linspace(10e-14, 200, 1000)


def eigen_values(x_1, x_2, l):

    zero_list = []
    turn = index(l)

    for i in range(0,10):

        if i <5:

            zero = fsolve(phi, e_range[turn[i]], args = (x_1, l_vals[l]))
            zero_list.append(zero[0])

        else:
            zero = fsolve(phi, e_range[turn[i]], args = (x_2, l_vals[l]))
            zero_list.append(zero[0])

    return zero_list

first = eigen_values(x_range_1, x_range_2, 0)
print('first ten energy eigenvalues for l = 0')
print('-------------------------------------------')
print(first)
print('-------------------------------------------')

print('first ten energy eigenvalues for l = 1')
print('-------------------------------------------')
second = eigen_values(x_range_1, x_range_2, 1)
print(second)
print('-------------------------------------------')

print('first ten energy eigenvalues for l = 2')
print('-------------------------------------------')
third = eigen_values(x_range_1, x_range_2, 2)
print(third)
print('-------------------------------------------')

def Analytic(r, n, l):
    return ((2*r/n)**(l)*assoc_laguerre(2*r/n, n-l-1, 2*l+1))/np.exp(r/n)*r

reverse_x = np.linspace(60, 10e-14, 500)

for i in range(0,4):
    result = numerov(alpha, 0, reverse_x, first[i], 0)
    norm = simps((np.flip(result))**2, np.flip(reverse_x))
    norm_result = (np.flip(result))/np.sqrt(norm)
    plt.scatter(np.flip(reverse_x), norm_result, label = f'n={i+1}', s = 10)

plt.title('First Four Radial Wavefunctions for l=0')
plt.xlabel("r'")
plt.ylabel("K_nl(r')")
plt.legend()
plt.grid()
plt.show()


#Density Plot 1
result = numerov(alpha, 0, reverse_x, first[0], 0)
norm = simps((np.flip(result))**2, np.flip(reverse_x))
norm_result = (np.flip(result))/np.sqrt(norm)
plt.scatter(np.flip(reverse_x), norm_result**2, label = 'l=0', s = 10)
plt.title('Probability Density for n=1')
plt.xlabel("r'")
plt.ylabel("K_nl(r')^2")
plt.legend()
plt.grid()
plt.show()

#Density Plot 2
result = numerov(alpha, 0, reverse_x, first[1], 0)
norm = simps((np.flip(result))**2, np.flip(reverse_x))
norm_result = (np.flip(result))/np.sqrt(norm)

plt.scatter(np.flip(reverse_x), norm_result**2, label = 'l=0', s = 10)

result = numerov(alpha, 0, reverse_x, second[1], 1)
norm = simps((np.flip(result))**2, np.flip(reverse_x))
norm_result = (np.flip(result))/np.sqrt(norm)

plt.scatter(np.flip(reverse_x), norm_result**2, label = 'l=1', s = 10)

plt.title('Probability Density for n=2')
plt.xlabel("r'")
plt.ylabel("K_nl(r')^2")
plt.legend()
plt.grid()
plt.show()

#Density Plot 3
result = numerov(alpha, 0, reverse_x, first[2], 0)
norm = simps((np.flip(result))**2, np.flip(reverse_x))
norm_result = (np.flip(result))/np.sqrt(norm)

plt.scatter(np.flip(reverse_x), norm_result**2, label = f'l=0', s = 10)

result = numerov(alpha, 0, reverse_x, second[2], 1)
norm = simps((np.flip(result))**2, np.flip(reverse_x))
norm_result = (np.flip(result))/np.sqrt(norm)

plt.scatter(np.flip(reverse_x), norm_result**2, label = f'l=1', s = 10)

result = numerov(alpha, 0, reverse_x, third[2], 2)
norm = simps((np.flip(result))**2, np.flip(reverse_x))
norm_result = (np.flip(result))/np.sqrt(norm)

plt.scatter((np.flip(reverse_x))[2:], (norm_result**2)[2:], label = f'l=2', s = 10)

plt.title('Probability Density for n=3')
plt.xlabel("r'")
plt.ylabel("K_nl(r')^2")
plt.legend()
plt.grid()
plt.show()

#Probability for 2s

p_range = [0.5, 1, 1.5, 3, 4, 5, 10, 20, 30, 40, 50, 60]

# N = 2, l = 0

two_s = numerov(alpha, 0, reverse_x, first[1], 0)
norm = simps((np.flip(two_s))**2, np.flip(reverse_x))

norm_result = (np.flip(two_s))/np.sqrt(norm)
final_x = np.flip(reverse_x)

prob = []
for i in p_range:

    probability = simps(norm_result**2, np.linspace(10e-14, i, 500))
    prob.append(probability)

    print(probability)

plt.plot(p_range, prob)
plt.scatter(p_range, prob, color = 'red', s = 20)
plt.title('Probability for 2s')
plt.xlabel("P")
plt.ylabel("Probability")
plt.grid()
plt.show()