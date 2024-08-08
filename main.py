# Imports
import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
from matplotlib import pyplot as plt

class Network:
    def __init__(self, *layer_sizes, loss='squared error', hidden_activation='relu', output_activation='softmax'):
        self.layer_sizes = layer_sizes
        self.layer_count = len(self.layer_sizes)

        accepted_loss_types = ['squared error']
        if not loss in accepted_loss_types:
            raise ValueError(f"{loss} is not an accepted loss type. The accepted values are {accepted_loss_types}")
        self.loss_type = loss

        accepted_activation_types = ['relu', 'softmax']

        if not hidden_activation in accepted_activation_types:
            raise ValueError(f"{hidden_activation} is not an accepted hidden activation function. The accepted values are {accepted_activation_types}")
        self.hidden_activation = hidden_activation

        if not output_activation in accepted_activation_types:
            raise ValueError(f"{output_activation} is not an accepted output activation function. The accepted values are {accepted_activation_types}")
        self.output_activation = output_activation
        
        self.weights = []
        self.weight_gradients = []

        self.biases = []
        self.bias_gradients = []

        self.activations = [np.zeros(self.layer_sizes[0])] # The first index is the input to the network
        self.errors = []
        
        for i in range(self.layer_count - 1):
            self.weights.append(np.random.rand(self.layer_sizes[i + 1], self.layer_sizes[i]) * 2 - 1)
            self.weight_gradients.append(np.zeros(self.weights[i].shape))

            self.biases.append(np.zeros(self.layer_sizes[i + 1]))
            self.bias_gradients.append(np.zeros(self.biases[i].shape))

            self.activations.append(np.zeros(self.layer_sizes[i + 1]))
            self.errors.append(np.zeros(self.layer_sizes[i + 1]))

    def feed_forward(self, x):
        # If the input shape doesn't match the size of the first layer
        if len(x) != self.layer_sizes[0]:
            raise ValueError(f'Expected inputs of size {self.layer_sizes[0]}. Got {len(x)}')
        
        self.activations[0] = x
        # 1 is subtracted because the first 'layer' is just the input to the network and doesn't have any processing
        for i in range(self.layer_count - 1):
            z = self.weights[i] @ self.activations[i] + self.biases[i]
            activation_type = self.output_activation if i == (self.layer_count - 2) else self.hidden_activation # If it's the last layer use the output activation type
            if activation_type == 'relu':
                activation = np.maximum(0, z)
            elif activation_type == 'softmax':
                exponents = np.exp(z - np.max(z))
                activation = exponents / np.sum(exponents)
            self.activations[i + 1] = activation
        
        return self.activations[-1]
    
    def back_propagate(self, x, learning_rate):
        self.feed_forward(x)

        errors = []
        # Error from loss to last layer activation
        if self.loss_type == 'squared error':
            loss_grad = 2 * self.activations[-1]
        
        # Error from the last layer activation to z
        if self.


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

x_train = [np.array(x).flatten() for x in x_train]
y_train = [np.array(y).flatten() for y in y_train]

x_test = [np.array(x).flatten() for x in x_test]
y_test = [np.array(y).flatten() for y in y_test]

the_network = Network(28 * 28, 16, 16, 10)
output = the_network.feed_forward(x_train[0])
plt.bar(range(len(output)), output)
output = the_network.feed_forward(x_train[1])
plt.bar(range(10), output)
plt.show()
print(the_network.feed_forward(x_train[0]))
# Some pseudocode
"""
x1 = ReLU((input * W1) + b1)
x2 = ReLU((x1 * W2) + b2)
x3 = ReLU((x2 * W3) + b3)
loss = (x3 - example)

d(x3) = (((x2 * W3) + b3) > 0) * W3 * d(x2)
d(x2) = (((x1 * W2) + b2) > 0) * W2 * d(x1)
"""
