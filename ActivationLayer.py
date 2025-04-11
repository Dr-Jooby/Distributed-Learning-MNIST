from Layer import Layer

class ActivationLayer(Layer):
    def __init__(self, act, act_prime):
        self.act = act
        self.act_prime = act_prime

    def forward_prop(self, input_data):
        self.input = input_data
        self.output = self.act(self.input)
        return self.output
    
    def back_prop(self, output_error, learning_rate):
        return self.act_prime(self.input) * output_error