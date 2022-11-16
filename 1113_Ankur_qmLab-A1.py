import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#constants
m = 9.1093837015e-31
e = 1.602176634e-19
eps = 8.8541878128e-12
h_c = 1.05457182e-34


def f_cln(n):
    return (m*(e**4))/(32*(np.pi**3)*(eps**2)*(h_c**3)*(n**3))

def f_qn(n):
    return ((m*(e**4))/(64*(np.pi**3)*(eps**2)*(h_c**3)))*((2*n-1)/(n**2*(n-1)**2))

p = 0
n_val=[]
logn_val = []
per_val = []
f_class = []
f_quant = []

tol = 1

while tol > 10e-6:
    p = p + 0.5
    n = 10**p

    log_n = np.log(n)
    n_val.append(n)
    logn_val.append(log_n)

    deltaF = abs(f_qn(n) - f_cln(n))
    rel_diff = deltaF/f_qn(n)
    rel_diff_per = rel_diff * 100

    per_val.append(rel_diff_per)

    f_class.append(f_cln(n))
    f_quant.append(f_qn(n))

    tol = rel_diff


data = {

  'n': n_val,
  'f_cln': f_class,
  'f_qn': f_quant,
  'Rel. Error %' : per_val 

}

df = pd.DataFrame(data)
print(df)

plt.plot(logn_val, per_val)
plt.scatter(logn_val, per_val)
plt.grid()
plt.title('% Relative Difference as a function of ln(n)')
plt.xlabel('ln(n)')
plt.ylabel('% Rel. Diff.')
plt.plot()
plt.show()

def energy(n):
    return -13.6/(n**2)


for i in range(1,11):
    x_vals = range(1,11)
    e_val = energy(i)*np.ones(10)
    plt.plot(x_vals, e_val,label = f'n = {i}')

plt.title('Energy Diagram')
plt.ylabel('Energy(ev)')

plt.grid()
plt.legend()
plt.show()
