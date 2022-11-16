import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import pandas as pd

def waveFunction_x(x_vals, b):

    y_vals = []

    for i in range(len(x_vals)):
        if abs(x_vals[i]) < b:
            y_vals.append(1/np.sqrt(2*b))

        else:
            y_vals.append(0)

    return np.array(y_vals)

x_range = np.linspace(-5, 5, 1000)

plt.plot(x_range, abs((waveFunction_x(x_range, 1))**2))
plt.title('Probability Density(x-space) at t=0')
plt.xlabel('x')
plt.ylabel('|y^2(x)|')
plt.grid()
plt.show()

k_range = np.linspace(-20, 20, 1000)

j = complex(0, 1)

def waveFunction_k(k_vals, x_vals, b):
    y_vals = []

    for i in range(len(x_vals)):
        y = waveFunction_x(x_vals, b)* np.exp(-j*k_vals[i]*x_vals)  
        integral = (1/np.sqrt(2*np.pi))*simps(y, x_vals)
        y_vals.append(integral)
    y_vals = np.array(y_vals)

    return y_vals
        
def normalize(x_vals, y_vals):
    norm = simps(abs(y_vals**2), x_vals)
    norm_result = y_vals/np.sqrt(norm)

    return x_vals, norm_result

norm_y = normalize(k_range, waveFunction_k(k_range, x_range, 1))[1]

plt.plot(k_range, abs(norm_y**2))
plt.title('Probability Density(p-space) at t=0')
plt.xlabel('p')
plt.ylabel('|y^2(p)|')
plt.grid()
plt.show()
plt.show()

def inverse_func_x(k_vals, x_vals, t):

    inverse_x = []

    for i in range(len(k_vals)):
        inv = norm_y* np.exp((j)*((k_vals*x_vals[i])-(k_vals**2)*t))
        integral = (1/np.sqrt(2*np.pi))*simps(inv, k_vals)
        inverse_x.append(integral)

    inverse_x = np.array(inverse_x)

    return inverse_x

t_range = np.linspace(0, 2, 10)

for t in (t_range):

    norm_inv = normalize(x_range, inverse_func_x(k_range, x_range, t))[1]
    plt.plot(x_range, abs(norm_inv**2), label = f't={t}')

plt.title('Probability Density Time Evolution(Box Wave Function)')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.grid()
plt.legend()
plt.show()

#Probability
prob_list = []
for t in (t_range):

    norm_inv = normalize(x_range, inverse_func_x(k_range, x_range, t))[1]
    norm_inv_sq = abs(norm_inv**2)

    probability = simps(norm_inv_sq[450:550], x_range[450:550])
    prob_list.append(probability)

data = {

  't': t_range,
  'Probability': prob_list

}

df = pd.DataFrame(data)
print(df)

#Gaussian
def gaussian_x(x_vals, sigma):
    
    return (1/(np.pi*sigma))**(1/4)*np.exp(-x_vals**2/(2*sigma**2))

x_range = np.linspace(-5, 5, 1000)
k_range = np.linspace(-20, 20, 1000)

def gaussian_k(k_vals, x_vals, sigma):
    y_vals = []

    for i in range(len(x_vals)):
        y = gaussian_x(x_vals, sigma)* np.exp(-j*k_vals[i]*x_vals)  
        integral = (1/np.sqrt(2*np.pi))*simps(y, x_vals)
        y_vals.append(integral)
    y_vals = np.array(y_vals)

    return y_vals
        
norm_y_gauss = normalize(k_range, gaussian_k(k_range, x_range, 1))[1]

def inverse_gaussian_x(k_vals, x_vals, t):

    inverse_x = []

    for i in range(len(k_vals)):
        inv = norm_y_gauss* np.exp((j)*((k_vals*x_vals[i])-(k_vals**2)*t))
        integral = (1/np.sqrt(2*np.pi))*simps(inv, k_vals)
        inverse_x.append(integral)

    inverse_x = np.array(inverse_x)

    return inverse_x

for t in (t_range):

    norm_inv_gauss = normalize(x_range, inverse_gaussian_x(k_range, x_range, t))[1]
    plt.plot(x_range, abs(norm_inv_gauss**2), label = f't={t}')

plt.title('Probability Density Time Evolution(Gaussian Wave Function)')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.grid()
plt.legend()
plt.show()