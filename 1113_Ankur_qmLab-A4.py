import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

def alpha(x):
    return -(1 + x**2)

def second_guess(h):

    return 1 + ((h**2)/2) + 3*(h**4)/24


def numerov(func, u0, u1, x_range):

    h = x_range[1] - x_range[0]
    u_values = np.zeros(len(x_range))
    u_values[0] = u0
    u_values[1] = u1(h)

    c_values = np.ones(len(x_range)) + np.multiply((h**2)/12, func(x_range))
    for i in range(1, len(x_range)-1):
        u_values[i+1] = (1/c_values[i+1])*((12-10*c_values[i])*u_values[i] - c_values[i-1]*u_values[i-1])

    return u_values

x_vals = np.linspace(0, 1, 100)

Numerov_Result = numerov(alpha, 1, second_guess, x_vals)
plt.scatter(x_vals, Numerov_Result, label = 'Numerov_Method', color = 'red', s = 15)

def func_x(x,y_vec):
    ans_vec=np.zeros((2))
    ans_vec[0]=y_vec[1]
    ans_vec[1]= (1+x**2)*y_vec[0]
    return ans_vec

Analytical_Result = solve_ivp(func_x, (0,1), (1,0), t_eval = x_vals)

plt.plot(x_vals, Analytical_Result.y[0], label = 'Analytical Result')
plt.grid()
plt.legend()
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Solution of Differential Equation(Validation)')
plt.show()

print('N=2')
print('------------------------------------------')

x_vals_1 = np.linspace(0, 1, 3)
u_num_1 =  numerov(alpha, 1, second_guess, x_vals_1)
u_inbuilt_1 = (solve_ivp(func_x, (0,1), (1,0), t_eval = x_vals_1)).y[0]
E_i_1 = list(np.subtract(u_inbuilt_1, u_num_1))
E_i_1_abs =  [abs(ele) for ele in E_i_1]

data = {

  'x_i': x_vals_1,
  'u_num': u_num_1,
  'u_inbuilt': u_inbuilt_1, 
  'E_i' : E_i_1_abs

}

df = pd.DataFrame(data)
print(df)
print('------------------------------------------')

print('N=4')
print('------------------------------------------')

x_vals_2 = np.linspace(0, 1, 5)
u_num_2 =  numerov(alpha, 1, second_guess, x_vals_2)
u_inbuilt_2 = (solve_ivp(func_x, (0,1), (1,0), t_eval = x_vals_2)).y[0]
E_i_2 = list(np.subtract(u_inbuilt_2, u_num_2))
E_i_2_abs =  [abs(ele) for ele in E_i_2]

data = {

  'x_i': x_vals_2,
  'u_num': u_num_2,
  'u_inbuilt': u_inbuilt_2, 
  'E_i' : E_i_2_abs

}

df = pd.DataFrame(data)
print(df)
print('------------------------------------------')

markers = ["o", "v", "s", "p", "*", "+"]
for i in range(1,7):
    x_vals = np.linspace(0,1,(2**i)+1)
    Numerov= numerov(alpha, 1, second_guess, x_vals)
    plt.scatter(x_vals, Numerov, label = f'k={i}', marker = markers[i-1], s = 20)
    plt.plot(x_vals, Numerov, linestyle = 'dashed')
 
Analytical= solve_ivp(func_x, (0,1), (1,0), t_eval = np.linspace(0,1,100))
plt.plot(np.linspace(0,1,100), Analytical.y[0], label = 'Analytical Solution')
plt.grid()
plt.legend()
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title(r'Solution of Differential Equation(For Different N = $2^k$)')
plt.show()