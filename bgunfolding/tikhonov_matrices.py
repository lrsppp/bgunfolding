import numpy as np

"""
Source: Hansen
Page 175

zero == False ggf. implementieren.. aber eigentlich immer zero BC
"""

__all__ = ['first_order_forward',
           'first_order_central',
           'second_order_central']

def first_order_forward(n, zero = True):
    """
    
    """
    m1 = -np.eye(n) + np.eye(n, k = 1)
    return np.vstack([np.ones(n), m1])
    
def first_order_central(n, zero = True):
    
    return np.eye(n, k = 1) - np.eye(n, k = -1)

def second_order_central(n, zero = True):
    """
    See: Lista L., Statistical Methods for Data Analaysis in Particle Physics (2017), p. 165
    """
    
    M = np.eye(n, k = 1) + np.eye(n, k = -1) - 2 * np.eye(n)
    if zero == True:
        M[0,0], M[-1,-1] = -1, -1
        return M
    
    else: return M
