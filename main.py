# Imports
from numpy.core.defchararray import lower
from mnistdataloader import *
from network import *
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

# Loading data from files
input_path = 'data'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Formatting training data
x_train = np.array([np.array(x).flatten() for x in x_train])
y_train = np.eye(10)[y_train]

# Shuffling the data
shuffle_order = np.random.permutation(len(x_train))
x_train = x_train[shuffle_order]
y_train = y_train[shuffle_order]

# Splitting into batches
batch_size = 64
x_batches = [x_train[i:i+64] for i in range(int(np.ceil(len(x_train) / batch_size)))]
y_batches = [y_train[i:i+64] for i in range(int(np.ceil(len(y_train) / batch_size)))]

# Formatting test data
x_test = [np.array(x).flatten() for x in x_test]
y_test = np.eye(10)[y_test]

nn = Network(28 * 28, 32, 32, 32, 10)

while True:
    nn.batch_train(x_batches[0], y_batches[0])
exit = False

# print(f'Training on 20 batches of {batch_size}...')
# sleep(2)
# while True:
#     j = 1
#     for x, y, in zip(x_batches, y_batches):
#         for i in range(100):
#             nn.batch_train(x, y, learning_rate=0.01)
#         print(f'Training loss: {nn.batch_loss(x_train, y_train)}')
#         if j % 20 == 0:
#             print(f'\n\nTest loss is now {nn.batch_loss(x_test, y_test)}')
#             user_answer = input('Would you like to continue training for 20 more batches? y/n\n')
#             if lower(user_answer) in ('no', 'n'):
#                 exit = True
#                 break
#         j += 1
#     if exit: break

