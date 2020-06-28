import os
import gzip
import numpy as np
import pickle
import wget

from datasets.coil20.feed import feed

import scipy.io


def load_digits():
    from sklearn import datasets
    data = datasets.load_digits()
    
    return data.data, data.target


def load_fashion_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def unpickle_cifar100(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar100(path="data/cifar100/cifar-100-python"):
    
    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "test")
    
    if not os.path.exists(train_path):
        raise ValueError("Path %s not found. Please provide path to "
                         "untarred cifar100 dataset." % train_path)

    if not os.path.exists(test_path):
        raise ValueError("Path %s not found. Please provide path to "
                         "untarred cifar100 dataset." % test_path)

    train = unpickle_cifar100(train_path)
    test = unpickle_cifar100(test_path)
    
    return train, test


def load_shuttle(filepath="data/shuttle.mat"):
    if not os.path.exists(filepath):
        raise ValueError("File shuttle.mat not found. Please download "
                         "from 'https://www.dropbox.com/s/mk8ozgisimfn3dw/shuttle.mat'")

    mat = scipy.io.loadmat(filepath)
    
    X = mat["X"].astype(np.float32)
    y = mat["y"].astype(np.int32).ravel()
    return X, y


def load_coil20(path="data/coil20"):
    feed(feed_path=path, dataset_type='processed')

    from datasets import pa2np
    X, Y = pa2np(os.path.join(path, "X_processed.pa")), pa2np(os.path.join(path, "Y_processed.pa"))

    features = X.shape[2]*X.shape[3]
    new_X = np.zeros((X.shape[0], features))

    from skimage import color
    for i in range(X.shape[0]):
        img = X[i, :, :, :]
        shape = features
        gray = color.rgb2gray(np.moveaxis(img, 0, 2)).reshape(shape)
        new_X[i] = gray

    X = new_X.astype(np.float32)
    y = Y.astype(np.float32)
    
    return X, y
                 
def load_mnist(path="data/mnist/"):
    from datasets.mnist.feed import feed
    feed(feed_path=path)

    from datasets import pa2np
    X, Y = pa2np(os.path.join(path, "X.pa")), pa2np(os.path.join(path, "Y.pa"))

    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    y = Y
    
    return X, y


def load_word2vec(path):
    
    from gensim.models import KeyedVectors
    
    bin_file = os.path.join(path, "/GoogleNews-vectors-negative300.bin")
    
    if not os.path.exists(bin_file):
        raise ValueError("GoogleNews-vectors-negative300.bin was not found in " + path + 
                         ". You will need to download this file and place in 'path'")

    vecs = KeyedVectors.load_word2vec_format(bin_file, binary=True)

    return vecs.vectors

