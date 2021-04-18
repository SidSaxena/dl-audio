from abc import abstractstaticmethod
import numpy as np 

# save activations and derivatives
# implement backpropagation
# implement gradient descent
# implement train method
# train our network with dummy dataset
# make some prediction

class MLP:
    """A Multilayer Perceptron class"""

    def __init__(self, num_inputs=3, num_hidden=[3,3], num_outputs=2):
        
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        # initialise random weights
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append([w])
        self.weights = weights

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations
        
        
        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros(layers[i, layers[i+1]])
            derivatives.append(d)
        self.derivatives = derivatives


    def forward_propogate(self, inputs):
        
        activations = inputs
        self.activations[0] = inputs 

        for i, w in enumerate(self.weights):
            # calculate net inputs
            net_inputs = np.dot(activations, w)

            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations
        
        return activations
    
    def back_propagate(self, error):
        
        for i in reversed(range(len(self.derivatives))): 
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations)
            current_activations = self.activations[i]

            self.derivatives[i] = np.dot(current_activations, delta)

    
    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)





    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == '__main__':

    # create an MLP
    mlp = MLP()

    # create some inputs
    inputs = np.random.rand(mlp.num_inputs)

    # perform forward propogation
    outputs = mlp.forward_propogate(inputs)

    # print the result
    print("The network input is {}".format(inputs))
    print("The network output is {}".format(outputs))