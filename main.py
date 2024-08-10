# Imports
from mnistdataloader import *
from network import *
import numpy as np
from os.path  import join

# Loading data from files
input_path = 'data'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Formatting data
x_train = [np.array(x).flatten() for x in x_train]
y_train = np.eye(10)[y_train]

batch_size = 64
x_batches = [x_train[i:i+64] for i in range(int(np.ceil(len(x_train) / batch_size)))]
y_batches = [y_train[i:i+64] for i in range(int(np.ceil(len(y_train) / batch_size)))]

x_test = [np.array(x).flatten() for x in x_test]
y_test = np.eye(10)[y_test]

nn = Network(28 * 28, 16, 16, 10)
loss = nn.batch_loss(x_train, y_train)
print(f'Loss before: {loss}')
for x, y in zip(x_batches, y_batches):
    for i in range(100):
        nn.batch_train(x, y)
    print(f"{nn.batch_loss(x, y):.10f}")
loss = nn.batch_loss(x_train, y_train)
print(f'Loss after: {loss:.10f}')

