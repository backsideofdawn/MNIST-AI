# Imports
import numpy as np # linear algebra
import struct
from array import array
from os.path  import join

# Following region is from https://www.kaggle.com/code/hojjatk/read-mnist-dataset?scriptVersionId=9466282&cellId=1
# region MnistDataLoader    
#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)        


# endregion

input_path = 'data'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')    

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

class Layer:
    def __init__(self, input_count: int, output_count: int, activation='linear', next_layer=None):
        # There are the inputs which are the outputs of the previous layer's neruons
        # and then you have this layer's outputs which are the result of the weight
        # multiplication and added bias
        self.input_count = input_count
        self.output_count = output_count

        self.weights = np.random.rand(output_count, input_count)
        self.weight_derivatives = np.zeros(output_count, input_count)

        self.bias = np.random.rand(output_count)
        self.bias_derivatives = np.zeros(output_count)

        self.output = None

        self.next_layer = next_layer
        accepted_activations = ['relu', 'tanh', 'softmax', 'linear']
        if not activation in accepted_activations:
            raise ValueError(f"Unknown activation type {self.activation}. Accepted values are {accepted_activations}")
        self.activation = activation
    
    def feed_forward(self, inputs):
        # If the number of inputs doesn't match the expected input count
        if len(inputs) != self.input_count:
            raise ValueError(f'Expected inputs of size {self.input_count}. Got {len(inputs)}')
        
        self.output = np.dot(inputs, self.weights) + self.bias

        if self.activation == 'relu':
            # Same as np.maximum(output, 0), but much faster
            self.output[self.output < 0] = 0
        elif self.activation == 'tanh':
            self.output = np.tanh(self.output)
        elif self.activation == 'softmax':
            exponents = np.exp(self.output)
            self.output = exponents / np.sum(exponents)
        elif self.activation == 'linear':
            pass
        return self.output
    
    def derivative_weighted_input_activation(self):
        if self.activation == 'relu':
            return np.array(self.output > 0, dtype=float)
        if self.activation == 'linear':
            return self.output
        # TODO: Add more activation derivatives
    
    def error_last_layer(self, example):
        # Loss derivative
        if self.loss_type == 'squared error':
            activation_loss_derivative = (2 * (self.output - example)) * (1 / len(self.output))
    
        return activation_loss_derivative * self.derivative_weighted_input_activation()
    
    def error_hidden_layer(self, next_layer_errors):
        weighted_input_error_derivative = np.dot(np.transpose(self.next_layer.weights), next_layer_errors)
        return weighted_input_error_derivative * self.derivative_weighted_input_activation()
    

    
class Network:
    def __init__(self, *neuron_counts, loss_type='squared error'):
        self.neuronCounts = neuron_counts
        accepted_loss_types = ['squared error']
        if not loss_type in accepted_loss_types:
            raise
        self.loss_type = loss_type
        self.layers = []
        # The output variable in this class is for the entire network
        # whereas in the Layer class it's only for a single layer
        self.output = None
        # The activation type for each layer is declared in the initalization of the object                   
        # The zip function makes a list of tuples given two lists, the shorter list being the length of the outputed list
        # j is just i but +1 index further in neuron counts
        for i, j in zip(neuron_counts, neuron_counts[1:]): 
            # If this is the last layer
            if j == neuron_counts[-1]:
                activation = 'softmax'
            else:
                activation = 'relu'
            
            new_layer = Layer(i, j, activation)
            self.layers.append(new_layer)
            self.layers[-2].next_layer = new_layer
    
    def feed_forward(self, inputs):
        # If the number of inputs doesn't match the size of the first layer
        if len(inputs) != self.layers[0].input_count:
            raise ValueError(f'Expected inputs of size {self.layers[0].input_count}. Got {len(inputs)}')
        
        self.output = inputs
        for i in range(len(self.neuronCounts)):
            # Feeds the last layers output into the next one's input, and stores the output in the output variable
            self.output = self.layers[i].feed_forward(self.output)
        return self.output
    
    def loss(self, output, example):
        if self.loss_type == 'squared error':
            return np.sum((output - example) ** 2) / len(output)
    
    def back_propagate(self, example, learning_rate):
        pass


        
        
# Some pseudocode
"""
x1 = ReLU((input * W1) + b1)
x2 = ReLU((x1 * W2) + b2)
x3 = ReLU((x2 * W3) + b3)
loss = (x3 - example)

d(x3) = (((x2 * W3) + b3) > 0) * W3 * d(x2)
d(x2) = (((x1 * W2) + b2) > 0) * W2 * d(x1)
"""
