import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy
from utils.model_utils import get_dataset_name
from utils.model_config import RUNCONFIGS
from FLAlgorithms.optimizers.fedoptimizer import pFedIBOptimizer
from torch import optim
from FLAlgorithms.additional.model import Net

class Client:
    """
    Base class for users in federated learning.
    """

    def __init__(
            self, args, id, model, train_data, test_data, opti, my_model_name=None, unique_labels=None):

        print('*'*108)
        print(model)
        
        # self.model = copy.deepcopy(model[0])
        self.model = Net(args)
        
        self.id = id  # integer
        self.train_data = train_data
        self.test_data = test_data
        self.train_samples = len(self.train_data)
        self.test_samples = len(self.test_data)
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.local_epochs = args.local_epochs
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.trainloader = DataLoader(self.train_data, self.batch_size, drop_last=True, shuffle=True)
        self.testloader = DataLoader(self.test_data, self.batch_size, drop_last=True)

        self.testloaderfull = DataLoader(self.test_data, self.test_samples)
        self.trainloaderfull = DataLoader(self.train_data, self.train_samples, shuffle=True)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

        self.test_data_so_far = []
        self.test_data_so_far += self.test_data
        self.test_data_so_far_loader = DataLoader(self.test_data_so_far, len(self.test_data_so_far))

        self.test_data_per_task = []
        self.test_data_per_task.append(self.test_data)

        dataset_name = get_dataset_name(self.dataset)
        self.unique_labels = unique_labels

        # those parameters are for personalized federated learning.
        self.prior_decoder = None
        self.prior_params = None

        # continual federated learning
        self.classes_so_far = []  # all labels of a client so far
        self.available_labels_current = []  # labels from all clients on T (current)
        self.current_labels = []  # current labels for itself
        self.classes_past_task = []  # classes_so_far (current labels excluded)
        self.available_labels_past = []  # labels from all clients on T-1
        self.current_task = 0
        self.init_loss_fn()
        self.label_counts = {}
        self.available_labels = []  # l from all c from 0-T
        self.label_set = [i for i in range(10)]
        self.my_model_name = my_model_name
        self.last_copy = None
        self.if_last_copy = False
        self.args = args
        
        self.opti = opti
        self.param_vector = self.get_vector()
        self.l2_norm = None
        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")
        self.differ = None

    def next_task(self, train, test, label_info=None, if_label=True):

        # update last model:

        self.last_copy = copy.deepcopy(self.model).to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.if_last_copy = True


        self.train_data = train
        self.test_data = test

        self.train_samples = len(self.train_data)
        self.test_samples = len(self.test_data)

        self.trainloader = DataLoader(self.train_data, self.batch_size, drop_last=True, shuffle=True)
        self.testloader = DataLoader(self.test_data, self.batch_size, drop_last=True)

        self.testloaderfull = DataLoader(self.test_data, len(self.test_data))
        self.trainloaderfull = DataLoader(self.train_data, len(self.train_data), shuffle=True)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

        # update classes_past_task
        self.classes_past_task = copy.deepcopy(self.classes_so_far)

        # update classes_so_far
        if if_label:
            self.classes_so_far.extend(label_info['labels'])
            self.current_labels.clear()
            self.current_labels.extend(label_info['labels'])

        # update test data for CL: (classes so far)
        self.test_data_so_far += self.test_data
        self.test_data_so_far_loader = DataLoader(self.test_data_so_far, len(self.test_data_so_far))

        # update test data for CL: (test per task)
        self.test_data_per_task.append(self.test_data)

        # update class recorder:
        self.current_task += 1

        return

    def init_loss_fn(self):
        self.ce_loss = nn.CrossEntropyLoss()


    def set_parameters(self, model, beta=1):
        '''
        self.model: old client model
        model: new collaboration model
        '''
        for old_param, new_param in zip(self.model.classifier.parameters(), model.classifier.parameters()):
            if beta == 1:
                old_param.data = new_param.data.clone()
            else:
                old_param.data = beta * new_param.data.clone() + (1 - beta) * old_param.data.clone()


    def train_a_batch_all(self, x, y,device,
                          available_labels,
                          classes_so_far,
                          glob_iter_,
                          importance_of_new_task=.5
                          ):
        # print('进入')
        self.model.to(device)
            # print(p.requires_grad)
        # self.opti = torch.optim.SGD(self.model.classifier.parameters(), lr=0.0001, weight_decay=0.9)
        self.model.train()
        sum_samples = len(x)
        # print(x.shape)
        # x = torch.reshape(x,(x.shape[0],1,x.shape[1],x.shape[2]))
        output, xa, logits = self.model.classifier(x)
        # loss = self.ce_loss(output, y.long())
        loss = torch.nn.functional.cross_entropy(output, y.long())
        acc = (torch.sum(torch.argmax(output, dim=1) == y)).item() #sum total correct samples
        # updation
        para_1 = self.get_vector()
        self.model.classifier_optimizer.zero_grad()
        loss.backward()
        self.model.classifier_optimizer.step()    
        para_2 = self.get_vector()
        diff = np.array_equal(para_1, para_2)
        # print(diff)
        self.differ = para_2 - para_1


        acc_rate = acc / sum_samples

        return {'loss': loss.item(), 'correct_samples': acc, 'total_samples': sum_samples, 'acc_rate':acc_rate}

    
    def criterion(self, t, outputs, targets, outputs_old=None):
        """Returns the loss value"""
        loss = 0
        if t > 0:
            # Knowledge distillation loss for all previous tasks
            loss += self.lamb * self.cross_entropy(torch.cat(outputs[:t], dim=1),
                                                   torch.cat(outputs_old[:t], dim=1), exp=1.0 / self.T)
        # Current cross-entropy loss -- with exemplars use all heads
        if len(self.exemplars_dataset) > 0:
            return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets.long())
        # print(len(outputs))
        # print(targets.shape)
        # print('='*200)
        return loss + torch.nn.functional.cross_entropy(outputs[t], targets.long() - self.model.task_offset[t])

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce
    
    def train_a_batch_lwf(self,
                          current_task,
                          last_copy,
                          if_last_copy,
                          x, y,
                          device):


        T = 2

        # ===================
        # 1. prediction loss


        # ===================
        self.model.to(self.args.device)
        sum_samples = len(x)
        real_output, xa, real_logits = self.model.classifier(x)
        # print('-'*108)
        # print('real_logits',real_logits)
        # print('-'*108)
        # print('real_output',real_output)
        class_loss = self.ce_loss(real_output, y.long())


        # =================================
        # 2. KD loss : last copy -> client
        # =================================
        if current_task>0:
            last_copy.to(device)
            real_logit_logp = torch.log(F.softmax(real_logits / T, dim=1))
            last_logits, xa, last_output = last_copy.classifier(x)
            last_logit_p = F.softmax(last_logits / T, dim=1).clone().detach()

            # kd_loss_copy = self.ensemble_loss(real_logit_logp, last_logit_p)
            kd_loss_copy = self.cross_entropy(real_output,last_output, exp=1.0 / 2)
            # kd_loss_copy = -torch.sum(last_logit_p * real_logit_logp).item()

        else:
            kd_loss_copy = 0

        if current_task > 0:
            alpha = self.args.alpha
        else:
            alpha = 0
        # print('---------------------',kd_loss_copy)
        # print('---------------------',class_loss)

        loss_all = class_loss + alpha * kd_loss_copy

        # updation
        para_1 = self.get_vector()
        self.model.classifier_optimizer.zero_grad()
        loss_all.backward()
        self.model.classifier_optimizer.step()
        para_2 = self.get_vector()
        self.differ = para_2 - para_1
        return {'loss_all': loss_all.item()}

    def train_a_batch_prox(self,available_labels,
                           glob_iter_,
                           x, y,
                           device,
                           net,
                           mu
                          ):

        # print('--------------------------------------------',mu)
        net.to(device)
        self.model.to(device)
        sum_samples = len(x)
        # print(x.shape)
        output, xa, logits = self.model.classifier(x)
        proximal_term = 0.0
        for w, w_t in zip(self.model.classifier.parameters(), net.classifier.parameters()):
            proximal_term += (w - w_t).norm(2)
        loss = self.ce_loss(output, y.long()) + mu / 2 * proximal_term
        acc = (torch.sum(torch.argmax(output, dim=1) == y)).item() #sum total correct samples
        # updation
        # para_1 = self.get_vector()
        self.model.classifier_optimizer.zero_grad()
        loss.backward()
        self.model.classifier_optimizer.step()    
        # para_2 = self.get_vector()
        # diff = np.array_equal(para_1, para_2)
        # print(diff)
        acc_rate = acc / sum_samples

        return {'loss': loss.item(), 'correct_samples': acc, 'total_samples': sum_samples, 'acc_rate':acc_rate}



    def get_parameters(self):
        for param in self.model.classifier.parameters():
            param.detach()
        return self.model.classifier.parameters()

    def clone_model_paramenter(self, param, clone_param):
        with torch.no_grad():
            for param, clone_param in zip(param, clone_param):
                clone_param.data = param.data.clone()
        return clone_param

    def get_grads(self):
        grads = []
        for param in self.model.classifier.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def test(self, personal=True):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model.classifier.eval()
        test_acc = 0
        loss = 0
        # dataloader = torch.nn.functional.dataloader(dataloader,batch_size=20)
        # dataloader = self.testloader
        total_num = 0

        for x, y in self.testloaderfull:
            x = x.to(device)
            y = y.to(device)
            # print(x.shape)
            # print('-'*108)
            # if batch_idx != total_batches - 1:
            output, xa, logits = self.model.classifier(x)
            loss += self.ce_loss(output, y)
            total_num += y.shape[0]
            # print(y.shape[0])
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()  # counts: how many correct samples
        return loss

    def test_a_dataset(self, dataloader):
        '''
        test_acc: total correct samples
        loss: total loss (on a dataset)
        y_shape: total tested samples
        '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            self.model.classifier.eval()
            test_acc = 0
            loss = 0
            # dataloader = torch.nn.functional.dataloader(dataloader,batch_size=20)
            # dataloader = self.testloader
            total_num = 0
            total_batches = len(dataloader)

            for batch_idx, (x, y) in enumerate(dataloader):
                x = x.to(device)
                y = y.to(device)
                # print(x.shape)
                # print('-'*108)
                # if batch_idx != total_batches - 1:
                output, xa, logits = self.model.classifier(x)
                loss += self.ce_loss(output, y)
                total_num += y.shape[0]
                # print(y.shape[0])
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()  # counts: how many correct samples
        return test_acc, loss, total_num

    def test_per_task(self):

        self.model.classifier.eval()
        test_acc = []
        loss = []
        y_shape = []

        # evaluate per task:
        for test_data in self.test_data_per_task:
            # print(len(test_data))
            test_data_loader = DataLoader(test_data, 20)
            test_acc_, loss_, y_shape_ = self.test_a_dataset(test_data_loader)
            
            test_acc.append(test_acc_)
            loss.append(loss_)
            y_shape.append(y_shape_)

        return test_acc, loss, y_shape

    def test_all(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.classifier.eval()
        test_acc = 0
        loss = 0
        for x, y in self.test_data_so_far_loader:
            x = x.to(device)
            y = y.to(device)
            output, xa, logits = self.model.classifier(x)
            loss += self.ce_loss(output, y)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        return test_acc, loss, y.shape[0]


    def get_next_train_batch(self, count_labels=True):
        try:
            # Samples a new batch for personalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        result = {'X': X, 'y': y}
        if count_labels:
            unique_y, counts = torch.unique(y, return_counts=True)
            unique_y = unique_y.detach().numpy()
            counts = counts.detach().numpy()
            result['labels'] = unique_y
            result['counts'] = counts
        return result

    def get_next_test_batch(self):
        try:
            # Samples a new batch for personalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X, y)

    def save_model(self, task, epoch):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + str(self.id) + "task_" + str(task) + "epoch_" + str(epoch) + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))

    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))



    def compute_l2_norm(self):
        print(self.differ)
        self.l2_norm = np.linalg.norm(self.differ, ord=2)
        return self.l2_norm

    def get_vector(self):
        param_list = []
        for param in self.model.classifier.parameters():
            param_list.append(param.data.cpu().numpy().flatten())
        self.param_vector = np.concatenate(param_list)
        return self.param_vector
