import numpy as np

# activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def ReLu(x):
    return max(0,x)

def Sigmoid(x):
    return 1/(1-np.exp(-x))