import numpy as np
import gzip
import os
import platform
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
class GetDataSet(object):
    def __init__(self):
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None
        self._index_in_train_epoch = 0



    def mnistDataSetConstruct(self):
        data_dir = r'D:\learn\dataset\mnist\MNIST\raw'
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = extract_images(train_images_path)
        train_labels = extract_labels(train_labels_path)
        test_images = extract_images(test_images_path)
        test_labels = extract_labels(test_labels_path)

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1
        # mlp
        # train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
        # test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])
        # lenet
        train_images = np.transpose(train_images, (0, 3, 1, 2))
        test_images = np.transpose(test_images, (0, 3, 1, 2))

        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)
        train_images = torch.tensor(train_images)
        test_images = torch.tensor(test_images)


        train_labels = torch.argmax(torch.tensor(train_labels), dim=1)
        test_labels = torch.argmax(torch.tensor(test_labels), dim=1)

        return train_images, train_labels, test_images, test_labels

    def cifarDataSetConstruct(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = torchvision.datasets.CIFAR10(root=r'D:\learn\dataset\cifar10', train=True, download=False,
                                                     transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=r'D:\learn\dataset\cifar10', train=False, download=False,
                                                    transform=transform)

        # 提取训练集和测试集的图像和标签
        train_images, train_labels = [], []
        test_images, test_labels = [], []

        for i in range(len(train_dataset)):
            img, label = train_dataset[i]
            train_images.append(img)
            train_labels.append(label)

        for i in range(len(test_dataset)):
            img, label = test_dataset[i]
            test_images.append(img)
            test_labels.append(label)

        train_images = torch.tensor(torch.stack(train_images))
        train_label = torch.tensor(train_labels)
        test_images = torch.tensor(torch.stack(test_images))
        test_labels = torch.tensor(test_labels)

        return train_images, train_label, test_images, test_labels

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return dense_to_one_hot(labels)







if __name__=="__main__":
    'test data set'
    mnistDataSet = GetDataSet() # test NON-IID
    if type(mnistDataSet.train_data) is np.ndarray and type(mnistDataSet.test_data) is np.ndarray and \
            type(mnistDataSet.train_label) is np.ndarray and type(mnistDataSet.test_label) is np.ndarray:
        print('the type of data is numpy ndarray')
    else:
        print('the type of data is not numpy ndarray')
    print('the shape of the train data set is {}'.format(mnistDataSet.train_data.shape))
    print('the shape of the test data set is {}'.format(mnistDataSet.test_data.shape))
    print(mnistDataSet.train_label[0:100], mnistDataSet.train_label[11000:11100])

