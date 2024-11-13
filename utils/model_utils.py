import json
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import trange
import random
from FLAlgorithms.additional.model import Net
from torch.utils.data import DataLoader

from utils.dataset import *

METRICS = ['glob_acc', 'per_acc', 'glob_loss', 'per_loss', 'user_train_time', 'server_agg_time']

def convert_data(X, y, dataset=''):
    '''
    to  tensor
    '''
    if not isinstance(X, torch.Tensor):
        if 'celeb' in dataset.lower():
            X=torch.Tensor(X).type(torch.float32).permute(0, 3, 1, 2)
            y=torch.Tensor(y).type(torch.int64)

        else:
            X=torch.Tensor(X).type(torch.float32)
            y=torch.Tensor(y).type(torch.int64)
    return X, y



def read_user_data_cl(index, data, dataset='', count_labels=False, task = 0):
    '''
    INPUT:
        data: data[train/test][user_id][task_id]
    
    OUTPUT:
    str: name of user[i]
    list of tuple: train data
    list of tuple: test data
    '''
    
    #data contains: clients, groups, train_data, test_data, proxy_data(optional)
    #print('attention: reversed!')
    #task = 4-task
    
    id = data['client_names'][index]
    train_data = data['train_data'][id]
    test_data = data['test_data'][id]
    
    X_train, y_train = train_data['x'][task], torch.Tensor(train_data['y'][task]).type(torch.long)
    X_test, y_test = test_data['x'][task], torch.Tensor(test_data['y'][task]).type(torch.long)

    if 'EMNIST' in dataset or dataset == 'MNIST-SVHN-FASHION':
        train_data = [(x, y) for x, y in zip(X_train, y_train)]  # a list of tuple
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
    elif dataset == 'CIFAR100':
        img_size = 32
        train_transform = transforms.Compose([transforms.RandomCrop((img_size, img_size), padding=4),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ColorJitter(brightness=0.24705882352941178),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        test_transform = transforms.Compose([transforms.Resize(img_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        train_data = Transform_dataset(X_train, y_train, train_transform)
        test_data = Transform_dataset(X_test, y_test, test_transform)

    if count_labels:
        label_info = {}
        unique_y, counts = torch.unique(y_train, return_counts=True)
        unique_y = unique_y.detach().numpy()
        counts = counts.detach().numpy()
        label_info['labels'] = unique_y
        label_info['counts'] = counts

        return id, train_data, test_data, label_info

    return id, train_data, test_data


def create_model(args):
    model = Net(args)
    return model

