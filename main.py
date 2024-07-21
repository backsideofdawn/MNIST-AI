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
    def __init__(self, input_count, output_count):
        # There are the inputs which are the outputs of the previous layer's neruons
        # and then you have this layer's outputs which are the result of the weight
        # multiplication and added bias
        self.input_count = input_count
        self.output_count = output_count
        self.weights = np.random.rand(output_count, input_count)
        self.bias = np.random.rand(output_count)
    
    def feed_forward(self, inputs, activation):
        # If the number of inputs doesn't match the expected input count
        if len(inputs) != self.input_count:
            raise ValueError(f'Expected inputs of size {self.input_count}. Got {len(inputs)}')
        
        output = np.dot(inputs, self.weights) + self.bias

        if activation == 'relu':
            # Same as np.maximum(output, 0), but much faster
            output[output < 0] = 0
            return output
        elif activation == 'tanh':
            return np.tanh(output)
        elif activation == 'softmax':
            np.exp
        elif activation == 'linear':
            return output
        else:
            raise ValueError(f'Unknown activation type {activation}. Accepted values are "relu", "tanh", and "linear"')
class Network:
    def __init__(self, *neuronCounts):
        self.neuronCounts = neuronCounts
        self.layers = []
        for i in range(len(neuronCounts) - 1):
            newLayer = Layer(neuronCounts[i], neuronCounts[i + 1])
            self.layers.append(newLayer)
    
    def feed_forward(self, inputs):
        # If the number of inputs doesn't match the size of the first layer
        if len(inputs) != self.layers[0].input_count:
            raise ValueError(f'Expected inputs of size {self.layers[0].input_count}. Got {len(inputs)}')
        
