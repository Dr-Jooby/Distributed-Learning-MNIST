import numpy as np
import pandas as pd

from Network import Network
from FCLayer import FCLayer
from ActivationLayer import ActivationLayer
from ActivationFunction import tanh, tanh_prime
from LossFunction import mse, mse_prime, accuracy

class Agent:
    def __init__(self):
        self.net = Network()
        self.net.add(FCLayer(28*28, 128, tanh, tanh_prime))
        self.net.add(FCLayer(128, 10, tanh, tanh_prime))
        self.net.use(mse, mse_prime, accuracy)

    def fit_train(self,x, y, epochs, learning_rate):
        self.net.new_fit(x, y, epochs, learning_rate)

    def predict(self, x, y):
        self.net.predict(x, y)