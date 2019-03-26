from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases

install_aliases()

import numpy as np
import os
import gzip
import struct
import array

import matplotlib.pyplot as plt
import matplotlib.image
from urllib.request import urlretrieve


def download(url, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)


def mnist():
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)

    train_images = parse_images('data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
    test_images = parse_images('data/t10k-images-idx3-ubyte.gz')
    test_labels = parse_labels('data/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images, test_labels


def load_mnist():
    partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = partial_flatten(train_images) / 255.0
    test_images = partial_flatten(test_images) / 255.0
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    return N_data, train_images, train_labels, test_images, test_labels


def plot_images(images, ax, ims_per_row=5, padding=5, digit_dimensions=(28, 28),
                cmap=matplotlib.cm.binary, vmin=None, vmax=None):
    """Images should be a (N_images x pixels) matrix."""
    N_images = images.shape[0]
    N_rows = np.int32(np.ceil(float(N_images) / ims_per_row))
    pad_value = np.min(images.ravel())
    concat_images = np.full(((digit_dimensions[0] + padding) * N_rows + padding,
                             (digit_dimensions[1] + padding) * ims_per_row + padding), pad_value)
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0]) * row_ix
        col_start = padding + (padding + digit_dimensions[1]) * col_ix
        concat_images[row_start: row_start + digit_dimensions[0],
        col_start: col_start + digit_dimensions[1]] = cur_image
    cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    return cax


def save_images(images, filename, **kwargs):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_images(images, ax, **kwargs)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.savefig(filename)


# 1. a)
def binarize(data):
    init_matrix = np.zeros(data.shape)
    init_matrix[data >= 0.5] = 1
    return init_matrix


def get_target(data_label):  # get the row indexes for data corresponding to certain labels
    digits = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    return np.dot(digits, np.transpose(np.array(data_label)))


def get_target_rows(data_label, target_label):  # Boolean vectors indicating which ones are vectors
    logic_vector = get_target(data_label) == target_label
    rows = np.arange(logic_vector.shape[0])[logic_vector]
    return rows


# 1. b) calculate theta
def compute_theta(data, data_label, target_label):
    target_row_index = get_target_rows(data_label, target_label)
    target_data = data[target_row_index]
    theta_cd = np.sum(1 + target_data, axis=0) / (3 * target_data.shape[0])
    return theta_cd


# 1. c) plot theta matrix
def theta_matrix(data, data_label):  # turn \theta_1, ..., \theta_n to a matrix (to convert to an image)
    init_matrix = np.zeros((10, 784))
    for i in range(10):
        init_matrix[i,] = compute_theta(data, data_label, i)
    return init_matrix


# 1. d) average log likelihood & accuracy
# def log_likelihood_single_img(data_pt,data_pt_target,theta): #helper function
#    target=int(get_target(data_pt_target))
#    a1 = np.log(6*1/10)
#    theta_target = np.transpose(theta[target,])
#    a2 = np.dot(1+data_pt,np.log(theta_target))
#    a3 = np.dot(2-data_pt,np.log(1-theta_target))
#    return a1+a2+a3


def average_log_likelihood(data, data_label, theta_matrix):
    total = 0
    N = data.shape[0]
    for i in range(10):
        theta_i = np.transpose(theta_matrix[i,])
        row_num = get_target_rows(data_label, i)
        a2 = np.sum(np.dot(1 + data[row_num,], np.log(theta_i)))
        a3 = np.sum(np.dot(2 - data[row_num,], np.log(1 - theta_i)))
        total = total + a2 + a3
    return total / N


def predict(data, data_label, theta_matrix_):  # theta_matrix_ is 10 by 784
    theta_matrix_ = np.transpose(theta_matrix_)
    N = data.shape[0]  # number of data points
    prediction = np.ones((N, 1))  # create a prediction vector
    a2 = np.dot(1 + data, np.log(theta_matrix_))
    a3 = np.dot(2 - data, np.log(1 - theta_matrix_))
    predict_matrix = a2 + a3  # calculate log p(c|x)
    print("predict_matrix", predict_matrix.shape)
    probability_matrix = np.amax(predict_matrix, axis=1)  # take the biggest log probability
    print("probability_matrix", probability_matrix.shape)
    for i in range(N):
        prediction[i, 0] = int(np.where(predict_matrix[i,] == probability_matrix[i,])[0])
    return prediction


def accuracy(data, data_label, theta_matrix_):
    N = data.shape[0]
    print(N)
    predictions = predict(data, data_label, theta_matrix_).reshape((N, 1))
    print("data.shape", data.shape)
    labels = get_target(data_label).reshape((N, 1))
    print("labels.shape", labels.shape)
    diff = predictions - labels
    return float(sum(diff == 0) / N)


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)

    train_images = load_mnist()[1]
    train_labels = load_mnist()[2]
    test_images = load_mnist()[3]
    test_labels = load_mnist()[4]

    # building a model with samples
    #train_sample = binarize(train_images[1:10000, ])
    #train_labels = binarize(train_labels[1:10000, ])
    #test_sample = binarize(test_images[1:10000, ])
    #test_labels = binarize(test_labels[1:10000, ])

    train_sample = binarize(train_images)
    train_labels = binarize(train_labels)
    test_sample = binarize(test_images)
    test_labels = binarize(test_labels)


    theta_matrix_ = theta_matrix(train_sample, train_labels)

    #print(theta_matrix_)
    #save_images(theta_matrix_, "hi")
    print("Average Log-likelihood: ", average_log_likelihood(train_sample, train_labels, theta_matrix_))
    # print("Predictions: ", predict(train_sample, train_labels, theta_matrix_),predict(train_sample, train_labels, theta_matrix_).shape)
    # print("Labels: ", get_target(train_labels).reshape((9999, 1)), get_target(train_labels).shape)

    print("Training Accuracy", accuracy(train_sample, train_labels, theta_matrix_))  # accuracy on training set
    print("Testing Accuracy",accuracy(test_sample,test_labels,theta_matrix_)) #what happened here
