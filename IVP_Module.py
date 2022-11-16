import numpy as np
import matplotlib.pyplot as plt


#Assignment 8 Part A,B,C

def euler(f, initial_cond , t):
    """
    Finds the solution of a Differential Equation using Euler Method.
    
    Parameters
    ---------
    f : function
        A Python function or method for which the solution is to be found.
    initial_cond : array
        An Array of the Initial Conditions.
    t : array
        The x-axis values.
    
    Returns
    ---------
    mat : matrix
        Returns a matrix with the solution of each Differential Equation the nth order Differential Equation was broken into. 
    """
    h = t[1] - t[0]

    mat = np.array([],[])
    mat = np.zeros([len(t), len(initial_cond)])

    mat[0,:] = initial_cond

    ele = np.array([])

    for i in range(0 , len(t)-1):
        ele = mat[i,:] + np.multiply(h, f(t[i], mat[i,:]))
        mat[i+1,:] = ele
    
    return mat


def RK_2(f, initial_cond ,t):
    """
    Finds the solution of a Differential Equation using RK-2 Method.
    
    Parameters
    ---------
    f : function
        A Python function or method for which the solution is to be found.
    initial_cond : array
        An Array of the Initial Conditions.
    t : array
        The x-axis values.
    
    Returns
    ---------
    mat : matrix
        Returns a matrix with the solution of each Differential Equation the nth order Differential Equation was broken into. 
    """
    h = t[1] - t[0]

    mat = np.array([],[])
    mat = np.zeros([len(t), len(initial_cond)])

    mat[0,:] = initial_cond

    k1 = np.array([])
    k2 = np.array([])

    for i in range(0 , len(t)-1):
        k1 = np.multiply(h, f(t[i], mat[i,:]))
        k2 = np.multiply(h, f(t[i]+h/2, mat[i,:]+ k1/2))
        sum = np.multiply((k1+k2),1/2)

        ele = mat[i,:] + sum
        mat[i+1,:] = ele

    return mat



def RK_4(f, initial_cond, t, e):
    """
    Finds the solution of a Differential Equation using RK-4 Method.
    
    Parameters
    ---------
    f : function
        A Python function or method for which the solution is to be found.
    initial_cond : array
        An Array of the Initial Conditions.
    t : array
        The x-axis values.
    
    Returns
    ---------
    mat : matrix
        Returns a matrix with the solution of each Differential Equation the nth order Differential Equation was broken into. 
    """
    h = t[1] - t[0]

    mat = np.array([],[])
    mat = np.zeros([len(t), len(initial_cond)])

    mat[0,:] = initial_cond

    k1 = np.array([])
    k2 = np.array([])
    k3 = np.array([])
    k4 = np.array([])
    ele = np.array([])

    for i in range(0 , len(t)-1):

        k1 = f(t[i], mat[i,:],e)
        k2 = f(t[i]+(h/2),(mat[i,:]+np.multiply(k1, (h/2))),e)
        k3 = f(t[i]+(h/2),(mat[i,:]+np.multiply(k2, (h/2))),e)
        k4 = f(t[i]+(h/1),(mat[i,:]+np.multiply(k3, (h/1))),e)
        sum = np.multiply((k1+np.multiply(k2,2)+np.multiply(k3,2)+k4), (1/6))

        ele = mat[i,:] + np.multiply((sum), h)
        mat[i+1,:] = ele

    return mat



