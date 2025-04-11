from Layer import Layer
import numpy as np

class FCLayer(Layer):
    def __init__(self, input_size, output_size, act, act_prime):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        self.act = act
        self.act_prime = act_prime

    def forward_prop(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        self.output = self.act(self.output)
        return self.output

    def back_prop(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate*weights_error
        self.bias -= learning_rate*output_error
        return input_error

    def return_bias(self):
        return self.bias

    def return_weight(self):
        return self.weights