class Network:
    
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.train_acc = 0
        self.prediction = []
        self.test_acc = 0
        self.moreerrors = []
        self.acc = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime, acc):
        self.loss = loss
        self.loss_prime = loss_prime
        self.acc = acc

    def predict(self, input_data, y_test):
        samples = len(input_data)
        result = []
        self.test_acc = []
        prerr = 0

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_prop(output)

            prerr += self.acc(y_test[i],output)
            result.append(output)
            self.prediction.append(output)

        prerr /= samples/100
        self.test_acc = prerr
        return result
    
    def new_predict(self, input_data, y_test):
        samples = len(input_data)
        result = []
        accuracy = 0
        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_prop(output)
            accuracy += self.acc(y_test[i], output)
            result.append(output)
        return result

    def fit(self, x_train, y_train, epochs, learning_rate):

        samples = len(x_train)
        accuracy = 0
        for i in range(epochs):
            err = 0
            
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_prop(output)

                err += self.loss(y_train[j],output)
                accuracy += self.acc(y_train[j], output)
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.back_prop(error, learning_rate)
            self.moreerrors.append(error)
            err /= samples/100
            self.train_acc.append(100*accuracy/samples)
            # print('epoch %d/%d   error = %f %%' % (i+1, epochs, err))

    def new_fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)
        train_accuracy = 0
        # training loop
        for i in range(epochs):
            err = 0
            
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_prop(output)
                
                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                train_accuracy += self.acc(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.back_prop(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            self.train_acc = (100*train_accuracy/samples)
            # print('epoch %d/%d   error=%f' % (i+1, epochs, err))