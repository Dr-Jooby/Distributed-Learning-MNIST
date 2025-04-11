import numpy as np

# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

def accuracy(y_true, y_pred):
    y = np.argmax(y_pred)
    y = np.eye(10)[y]
    return int(np.sum(y == y_true) == y_pred.size)