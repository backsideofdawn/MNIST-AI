# Imports
import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
from matplotlib import pyplot as plt



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
            self.weights.append(np.random.rand(
                self.layer_sizes[i + 1],
                self.layer_sizes[i])
            - 0.5)
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
    def back_propagate(self, x, y, learning_rate):
        self.feed_forward(x)

        errors = [np.zeros(layer_size) for layer_size in self.layer_sizes[1:]]
 
        # You don't need to calculate the softmax derivative, because when combined with the
        # cross entropy loss it the error simplifies
        errors[-1] = self.activations[-1] - y
        self.weights[-1] -= learning_rate * (errors[-1].reshape(-1, 1) @ self.activations[-2].reshape(1, -1))
        self.biases[-1] -= learning_rate * errors[-1]

        # Now for all the other layers

        for i in reversed(range(self.layer_count - 2)): # Minus two because the last layer is already figured out
            errors[i] = (self.weights[i + 1].T @ errors[i + 1]) * self.relu_grad(self.activations[i + 1])
            self.weights[i] -= learning_rate * (errors[i].reshape(-1, 1) @ self.activations[i].reshape(1, -1))
            self.biases[i] -= learning_rate * errors[i].flatten()



    def squared_error_loss(self, x, y):
        return np.sum((y - x) ** 2) / len(x)

    def squared_error_grad(self, x):
        return (2 * x) / len(x)

    
    def cross_entropy_loss(self, x, y, epsilon=1e-10):
        x = np.clip(x, epsilon, 1.0 - epsilon)
        return -np.sum(y * np.log(x))

    def cross_entropy_grad(self, x, y):
        return x - y

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



# Following class is from https://www.kaggle.com/code/hojjatk/read-mnist-dataset?scriptVersionId=9466282&cellId=1
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

input_path = 'data'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

x_train = [np.array(x).flatten() for x in x_train]
y_train = np.eye(10)[y_train]

x_test = [np.array(x).flatten() for x in x_test]
y_test = np.eye(10)[y_test]

nn = Network(28 * 28, 16, 16, 10)
loss = nn.cross_entropy_loss(nn.feed_forward(x_train[0]), y_train[0])
print(f'Loss before: {loss}')
nn.back_propagate(x_train[0], y_train[0], 0.0001)
loss = nn.cross_entropy_loss(nn.feed_forward(x_train[0]), y_train[0])
print(f'Loss after: {loss:.10f}')