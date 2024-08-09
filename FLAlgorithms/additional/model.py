from torch import nn
import torch
from torch import optim
import glog as logger
import numpy as np
import torch.nn.functional as F

from FLAlgorithms.additional.classify_net import S_ConvNet, Resnet_plus
from torch.nn import functional as F

eps = 1e-30

def MultiClassCrossEntropy(logits, labels, T):
    logits = torch.pow(logits+eps, 1/T)
    logits = logits/(torch.sum(logits, dim=1, keepdim=True)+eps)
    labels = torch.pow(labels+eps, 1/T)
    labels = labels/(torch.sum(labels, dim=1, keepdim=True)+eps)

    outputs = torch.log(logits+eps)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return outputs

class Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        beta1 = args.beta1
        beta2 = args.beta2
        weight_decay = args.weight_decay
        lr = args.lr
        c_channel_size = args.c_channel_size
        dataset = args.dataset

        self.algorithm = args.algorithm

        if 'EMNIST-Letters' in dataset:
            # self.xa_shape=[128, 4, 4]
            self.xa_shape=[512]
            self.num_classes = 26
            self.classifier = S_ConvNet(28, 1, c_channel_size, xa_dim=int(np.prod(self.xa_shape)), num_classes=self.num_classes)

        elif dataset=='CIFAR100':
            self.xa_shape=[512]
            self.num_classes = 100
            self.classifier = Resnet_plus(32, xa_dim=int(np.prod(self.xa_shape)), num_classes=self.num_classes)
            # self.classifier = S_ConvNet(32, 3, c_channel_size, xa_dim=int(np.prod(self.xa_shape)), num_classes=self.num_classes)


        elif dataset=='MNIST-SVHN-FASHION':
            self.xa_shape=[512]
            self.num_classes = 20
            self.classifier = S_ConvNet(32, 3, c_channel_size, xa_dim=int(np.prod(self.xa_shape)), num_classes=self.num_classes)

        # self.alpha = nn.Parameter(torch.full((args.num_users,), 1.0/args.num_users))
        self.alpha = torch.full((args.num_users,), 1.0/args.num_users)
        print(self.alpha)

        # for n,p in self.classifier.named_parameters():
        #     print('n')

        self.alpha_optimizer = optim.Adam(
            [self.alpha],
            lr=lr, weight_decay=weight_decay,betas=(beta1, beta2)
        )
        self.classifier_optimizer = optim.Adam(
            self.classifier.parameters(),
            lr=lr, weight_decay=weight_decay,betas=(beta1, beta2)
        )

        parameters_fb = [a[1] for a in filter(lambda x: 'fc2' in x[0], self.classifier.named_parameters())]
        self.classifier_fb_optimizer = optim.Adam(
            parameters_fb, lr=lr, weight_decay=weight_decay, 
            betas=(beta1, beta2),
        )


        class_params = sum(p.numel() for p in self.classifier.parameters())


    def to(self, device):
        self.classifier.to(device)
        return self

    def parameters(self):
        for param in  self.classifier.parameters():
            yield param

    def named_parameters(self):
        for name, param in self.classifier.named_parameters():
            yield 'classifier.'+name, param

