import random
import struct
import numpy as np

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        # Read the magic number and number of images
        magic, num_images = struct.unpack('>II', f.read(8))
        # Read the dimensions of the images
        rows, cols = struct.unpack('>II', f.read(8))
        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
        return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        # Read the magic number and number of labels
        magic, num_labels = struct.unpack('>II', f.read(8))
        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels
    
def load_data_non_IID():
    # Paths to the MNIST files (update these paths as necessary)
    train_images_path = 'D:/EEE Uni Stuff/EEE Y3/Individual Project Reinforcement Learning/Python Code Bits/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte'
    train_labels_path = 'D:/EEE Uni Stuff/EEE Y3/Individual Project Reinforcement Learning/Python Code Bits/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
    test_images_path = 'D:/EEE Uni Stuff/EEE Y3/Individual Project Reinforcement Learning/Python Code Bits/MNIST/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
    test_labels_path = 'D:/EEE Uni Stuff/EEE Y3/Individual Project Reinforcement Learning/Python Code Bits/MNIST/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

    # Load the data
    train_images = load_mnist_images(train_images_path)
    train_labels = load_mnist_labels(train_labels_path)
    test_images  = load_mnist_images(test_images_path)
    test_labels  = load_mnist_labels(test_labels_path)

    train_indices = np.arange(train_labels.shape[0])
    np.random.shuffle(train_indices)

    train_images = train_images[train_indices]
    train_labels  = train_labels[train_indices]

    test_indices = np.arange(test_labels.shape[0])
    np.random.shuffle(test_indices)

    test_images = test_images[test_indices]
    test_labels  = test_labels[test_indices]

    x_train = train_images.reshape(train_images.shape[0], 1, 28*28)
    x_train = x_train.astype('float32')
    x_train /= 255

    y_train = np.eye(10)[train_labels]

    x_test = test_images.reshape(test_images.shape[0], 1, 28*28)
    x_test = x_test.astype('float32')
    x_test /= 255

    y_test = np.eye(10)[test_labels]

    x = [[],[],[],[],[]]
    y = [[],[],[],[],[]]

    for i in range(len(train_labels)):
        if train_labels[i] == 0 or train_labels[i] == 1:
            y[0].append(train_labels[i])
            x[0].append(x_train[i])
        if train_labels[i] == 2 or train_labels[i] == 3:
            y[1].append(train_labels[i])
            x[1].append(x_train[i])
        if train_labels[i] == 4 or train_labels[i] == 5:
            y[2].append(train_labels[i])
            x[2].append(x_train[i])
        if train_labels[i] == 6 or train_labels[i] == 7:
            y[3].append(train_labels[i])
            x[3].append(x_train[i])
        if train_labels[i] == 8 or train_labels[i] == 9:
            y[4].append(train_labels[i])
            x[4].append(x_train[i])
    
    y_train_new = [[],[],[],[],[]]
    
    for i in range(5):
        y_train_new[i] = np.eye(10)[y[i]]

    return x_train, x_test, y_train, y_train_new, y_test, x

def load_data_IID():
    # Paths to the MNIST files (update these paths as necessary)
    train_images_path = 'D:/EEE Uni Stuff/EEE Y3/Individual Project Reinforcement Learning/Python Code Bits/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte'
    train_labels_path = 'D:/EEE Uni Stuff/EEE Y3/Individual Project Reinforcement Learning/Python Code Bits/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
    test_images_path = 'D:/EEE Uni Stuff/EEE Y3/Individual Project Reinforcement Learning/Python Code Bits/MNIST/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
    test_labels_path = 'D:/EEE Uni Stuff/EEE Y3/Individual Project Reinforcement Learning/Python Code Bits/MNIST/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

    # Load the data
    train_images = load_mnist_images(train_images_path)
    train_labels = load_mnist_labels(train_labels_path)
    test_images  = load_mnist_images(test_images_path)
    test_labels  = load_mnist_labels(test_labels_path)

    train_indices = np.arange(train_labels.shape[0])
    np.random.shuffle(train_indices)

    train_images = train_images[train_indices]
    train_labels  = train_labels[train_indices]

    test_indices = np.arange(test_labels.shape[0])
    np.random.shuffle(test_indices)

    test_images = test_images[test_indices]
    test_labels  = test_labels[test_indices]

    x_train = train_images.reshape(train_images.shape[0], 1, 28*28)
    x_train = x_train.astype('float32')
    x_train /= 255

    y_train = np.eye(10)[train_labels]

    x_test = test_images.reshape(test_images.shape[0], 1, 28*28)
    x_test = x_test.astype('float32')
    x_test /= 255

    y_test = np.eye(10)[test_labels]

    x = [[],[],[],[],[]]
    y = [[],[],[],[],[]]

    x = np.split(x_train, 5)
    y = np.split(train_labels, 5)

    y_train_new = [[],[],[],[],[]]
    
    for i in range(5):
        y_train_new[i] = np.eye(10)[y[i]]

    return x_train, x_test, y_train, y_train_new, y_test, x