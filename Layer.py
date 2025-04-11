class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_prop(self, input):
        raise NotImplementedError
    
    def back_prop(self, output_error, learning_rate):
        raise NotImplementedError
    
    def return_bias(self):
        raise NotImplementedError

    def return_weight(self):
        raise NotImplementedError