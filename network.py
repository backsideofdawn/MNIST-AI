import numpy as np

class Network:
    def __init__(self, *layer_sizes):
        self.layer_sizes = layer_sizes
        self.layer_count = len(self.layer_sizes)

        self.weights = []
        self.weight_gradients = []

        self.biases = []
        self.bias_gradients = []

        # self.activations[index] are the inputs to the weights of weights[index]
        self.activations = [np.zeros(self.layer_sizes[0])] # The first index is the input to the network
        self.errors = []

        for i in range(self.layer_count - 1):
            # Normal distribution
            self.weights.append(np.random.rand(
                self.layer_sizes[i + 1],
                self.layer_sizes[i]
            ) * 2 - 1)
            self.weight_gradients.append(np.zeros(self.weights[i].shape))

            self.biases.append(np.zeros(self.layer_sizes[i + 1]))
            self.bias_gradients.append(np.zeros(self.biases[i].shape))

            self.activations.append(np.zeros(self.layer_sizes[i + 1]))
    
    def feed_forward(self, x):
        # If the input shape doesn't match the size of the first layer
        if len(x) != self.layer_sizes[0]:
            raise ValueError(f'Expected inputs of size {self.layer_sizes[0]}. Got {len(x)}')

        self.activations[0] = x
        # 1 is subtracted because the first 'layer' is just the input to the network and doesn't have any processing
        for i in range(self.layer_count - 1):
            z = (self.weights[i] @ self.activations[i]) + self.biases[i]
            # If we're on the output layer
            if i == self.layer_count - 2:
                activation = self.softmax(z)
            else:
                activation = self.relu(z)

            self.activations[i + 1] = activation

        return self.activations[-1]


    # This is mostly based off of chapter 2 in Neural Network and Deep Learning: http://neuralnetworksanddeeplearning.com/chap2.html
    def back_propagate(self, x, y, learning_rate, weights_grad, biases_grad):
        # Bias and weight grad are inputs that are passed by reference, not vaue
        
        self.feed_forward(x)

        errors = [np.zeros(layer_size) for layer_size in self.layer_sizes[1:]]
 
        # You don't need to calculate the softmax derivative, because when combined with the
        # cross entropy loss it the error simplifies
        errors[-1] = self.activations[-1] - y

        # Now for all the other layers
        for i in reversed(range(self.layer_count - 2)): # Minus two because the last layer is already figured out
            errors[i] = (self.weights[i + 1].T @ errors[i + 1]) * self.relu_grad(self.activations[i + 1])
        
        for i in reversed(range(self.layer_count - 1)):
            weights_grad[i] -= learning_rate * errors[i].reshape(-1, 1) @ self.activations[i].reshape(1, -1)
            biases_grad[i] -= learning_rate * errors[i]
        

    def batch_train(self, x_batch, y_batch, learning_rate=0.01):
        
        weights_grad = [np.zeros(layer.shape) for layer in self.weights]
        biases_grad = [np.zeros(layer.shape) for layer in self.biases]
        
        # Update the gradients for each example
        for x, y in zip(x_batch, y_batch):
            self.back_propagate(x, y, learning_rate, weights_grad, biases_grad)
        
        # Update the weights by the mean gradient
        for i in range(self.layer_count - 1):
            self.weights[i] += weights_grad[i] / len(x_batch)
            self.biases[i] += biases_grad[i] / len(x_batch)
            
    
    def batch_loss(self, x_batch, y_batch):
        loss = 0
        for x, y in zip(x_batch, y_batch):
            output = self.feed_forward(x)
            loss += self.loss(output, y)
        loss /= len(x_batch)
        return loss

    def loss(self, x, y, epsilon=1e-10):
        # Make sure the log of x isn't infinite
        x = np.clip(x, epsilon, 1.0 - epsilon)
        # Calculate cross entropy
        cross_entropy = -np.sum(y * np.log(x))
        return cross_entropy

    def softmax(self, z):
        exponents = np.exp(z - np.max(z))
        return exponents / np.sum(exponents)

    def softmax_grad(self, x):
        # Code from here: https://stackoverflow.com/questions/40575841/numpy-calculate-the-derivative-of-the-softmax-function/40576872#40576872
        SM = np.reshape(x, newshape=(-1, 1))
        return np.diagflat(x) - (SM @ SM.T)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_grad(self, x):
        return x > 0