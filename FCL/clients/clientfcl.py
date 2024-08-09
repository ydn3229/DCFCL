import torch
import torch.nn.functional as F
import numpy as np
from FCL.clients.clientbase import Client
import copy
from torch.utils.tensorboard import SummaryWriter

# example of calculating the frechet inception distance
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm


class ClientFCL(Client):
    def __init__(self,
                 args,
                 id,
                 model,
                 train_data,
                 test_data,
                 label_info,
                 opti,
                 my_model_name=None,
                 unique_labels=None,
                 ):
        super().__init__(args, id, model, train_data, test_data, opti, my_model_name=my_model_name,
                         unique_labels=unique_labels)
        self.label_info = label_info
        self.args = args
        self.minloss = args.minloss

    # ==================================== FCL as clients ================================

    def train(
            self,
            glob_iter_,
            glob_iter,
            importance_of_new_task=.5,
    ):
        '''
        @ glob_iter: the overall iterations across all tasks

        '''

        # init loss:
        c_loss_all = 0
        g_loss_all = 0

        # preparation:
        self.clean_up_counts()
        self.model.train()


        # device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        # =============================================== FCL ===============================================
        if self.args.FCL == 1 or self.args.local == 1 or self.args.l2c == 1:
            for iteration in range(self.local_epochs):

                samples = self.get_next_train_batch(count_labels=True)
                x, y = samples['X'].to(device), samples['y'].to(device)


                # train the model with a batch
 
                # result = self.train_a_batch_all(
                #     x=x, y=y,
                #     device=device,
                #     available_labels=self.available_labels,
                #     glob_iter_=glob_iter_,
                #     importance_of_new_task=importance_of_new_task, classes_so_far=self.classes_so_far)
                result = self.train_a_batch_lwf(
                        current_task=self.current_task,
                        last_copy=self.last_copy,
                        if_last_copy=self.if_last_copy,
                        x=x, y=y,
                        device=device)

                c_loss_all += result['loss_all']
            c_loss_avg = c_loss_all / self.local_epochs
            print('-----------loss-all-------------','client', self.id, '---------',c_loss_avg)



        # =============================================== Fedavg ===============================================
        elif self.args.fedavg == 1:

            for iteration in range(self.local_epochs):
                samples = self.get_next_train_batch(count_labels=True)
                x, y = samples['X'].to(device), samples['y'].to(device)

                result = self.train_a_batch_all(
                        available_labels=self.available_labels,
                        glob_iter_=glob_iter_,
                        x=x, y=y,
                        device=device,
                        importance_of_new_task=importance_of_new_task, classes_so_far=self.classes_so_far)
                # print('-----------acc------------',result['acc_rate'])
                c_loss_all += result['loss']
            c_loss_avg = c_loss_all / self.local_epochs
            print('-----------loss-all-------------','client', self.id, '---------',c_loss_avg)

        # =============================================== FedLwF ===============================================
        elif self.args.fedlwf == 1:
            for iteration in range(self.local_epochs):
                samples = self.get_next_train_batch(count_labels=True)
                x, y = samples['X'].to(device), samples['y'].to(device)

                result = self.train_a_batch_lwf(
                        current_task=self.current_task,
                        last_copy=self.last_copy,
                        if_last_copy=self.if_last_copy,
                        x=x, y=y,
                        device=device)

                c_loss_all += result['loss_all']
            c_loss_avg = c_loss_all / self.local_epochs
        # =============================================== FedProx ===============================================
        elif self.args.fedprox == 1:
            net = copy.deepcopy(self.model)
            self.state = copy.deepcopy(self.model)
            for iteration in range(self.local_epochs):
                samples = self.get_next_train_batch(count_labels=True)
                x, y = samples['X'].to(device), samples['y'].to(device)
                result = self.train_a_batch_prox(
                        available_labels=self.available_labels,
                        glob_iter_=glob_iter_,
                        x=x, y=y,
                        device=device,
                        net=net,
                        mu=self.args.mu,
                )
                c_loss_all += result['loss']
                c_loss_avg = c_loss_all / (iteration+1)
                if c_loss_avg < self.minloss:
                    self.minloss = c_loss_avg
                    self.state = copy.deepcopy(self.model)
            self.model = self.state


    #             print('c_loss_avg: ', c_loss_avg)

    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr = max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {label: 0 for label in range(self.unique_labels)}


# tools
# calculate frechet inception distance
def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid