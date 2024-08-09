from FCL.clients.clientfcl import ClientFCL
from FCL.clients.aggerate import aggeregator
from FCL.model import ENV
from utils.model_utils import read_data, read_user_data, aggregate_user_data, create_generative_model, read_user_data_cl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision.utils import save_image
import os
import copy
import time
from torch import optim
from utils.dataset import *
import pandas as pd
from utils.util import *


MIN_SAMPLES_PER_LABEL = 1
import random

# writer
from torch.utils.tensorboard import SummaryWriter

_logger = logging.getLogger('train')

class FCLInit(aggeregator):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        # Initialize data for all users
        self.data = get_dataset(args, args.dataset, args.datadir, args.data_split_file)
        data = self.data
        all_test = data['all_test']
        _logger.info(all_test)
        # data contains: clients, groups, train_data, test_data, proxy_data
        clients = data['client_names']
        total_users = len(clients)
        self.total_users = total_users
        self.total_test_samples = 0
        self.local = 'local' in self.algorithm.lower()
        self.use_adam = 'adam' in self.algorithm.lower()
        self.init_loss()
        self.current_task = 0
        self.init_configs()
        self.local_epochs = args.local_epochs
        self.opti = optim.SGD(self.model.parameters(), lr=0.01)
        self.unique_labels = data['unique_labels']

        # =========================
        # init users with task = 0
        # =========================
        self.users = []
        for i in range(total_users):
            id, train_data, test_data, label_info = read_user_data_cl(i, data, dataset=args.dataset, count_labels=True,
                                                                      task=0)

            # count total samples (accumulative)
            self.total_train_samples += len(train_data)
            self.total_test_samples += len(test_data)
            id, train, test = read_user_data_cl(i, data, dataset=args.dataset, task=0)
            id = int(id[-1])

            # ============ initialize Users with data =============

            # init.
            user = ClientFCL(
                args,
                id,
                model,
                train_data,
                test_data,
                label_info,
                self.opti,
                unique_labels=self.unique_labels
            )

            self.users.append(user)

            # update classes so far & current labels
            user.classes_so_far.extend(label_info['labels'])
            user.current_labels.extend(label_info['labels'])
        # 创建利润表
        self.create_table()
        # _logger.log("Number of Train/Test samples:", self.total_train_samples, self.total_test_samples)
        # print("Data from {} users in total.".format(total_users))
        # print("Finished creating FedAvg server.")

        # ==================
        # training devices:
        # ==================
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print('Using device: ', device)


    def train(self, args):

        for task in range(args.ntask):

            # ===================
            # The First Task
            # ===================
            if task == 0:

                # update labels info. for the first task
                available_labels = set()
                available_labels_current = set()
                for u in self.users:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))

                for u in self.users:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)

            # ===================
            # Initialize new Task
            # ===================
            else:
                self.current_task = task

                for i in range(self.total_users):

                    id, train_data, test_data, label_info = read_user_data_cl(i, self.data, dataset=args.dataset,
                                                                              count_labels=True, task=task)

                    # update dataset

                    self.users[i].next_task(train_data, test_data, label_info)

                    # update labels info.
                    available_labels = set()
                    available_labels_current = set()
                    for u in self.users:
                        available_labels = available_labels.union(set(u.classes_so_far))
                        available_labels_current = available_labels_current.union(set(u.current_labels))

                    for u in self.users:
                        u.available_labels = list(available_labels)
                        u.available_labels_current = list(available_labels_current)

            # ===================
            #    print info.
            # ===================

            for u in self.users:
                print("classes so far: ", u.classes_so_far)
                print("available labels for the Client: ", u.available_labels)
                print("available labels (current) for the Client: ", u.available_labels_current)

            # ===================
            #    visualization
            # ===================
            # 1. server side:
            # 2. user side:

            # ============ train ==============
            epoch_per_task = int(self.num_glob_iters / args.ntask)

            for glob_iter_ in range(epoch_per_task):

                # ===================
                #    visualization
                # ===================
                # if self.args.visual == 1:
                #     my_classes_so_far = None
                #     # 1. user side:
                #     for num, u in enumerate(self.users):
                #         my_classes_so_far = u.classes_so_far
                #         for label in my_classes_so_far:
                #             title = 'N'+ str(args.naive) +'-'+ str(self.current_task) + '-C' + str(num) +  '-' + str(label) + '-' + str(args.dataset)
                #             visualize(self.generator, 25, title, [label])
                #
                #         my_classes_so_far = u.available_labels
                #
                #     # 2. user side:
                #     for label in my_classes_so_far:
                #         title = 'N'+ str(args.naive) +'-'+ str(self.current_task) + '-S-' + str(label) + '-' + str(args.dataset)
                #         visualize(self.generator, 25, title, [label])

                glob_iter = glob_iter_ + epoch_per_task * task

                _logger.info("Round number:", glob_iter, " | Current task:", task)
                _logger.info('\n')

                # select users
                self.selected_users, self.user_idxs = self.select_users(glob_iter, self.num_users, return_idx=True)

                # if arg.algorithm contains "local". In most cases, it does not.
                # broadcast averaged prediction model to clients
                if not self.local:
                    assert self.mode == 'all'

                    # send parameteres: server -> client
                    if glob_iter == 0:
                        # init global model
                        self.init_send_parameters()

                chosen_verbose_user = np.random.randint(0, len(self.users))
                self.timestamp = time.time()  # log user-training start time

                # ---------------
                #   train user
                # ---------------
                '''
                1. regularization from global model: kd with its/global labels 
                2. regularization from itself
                '''

                for user_id, user in zip(self.user_idxs, self.selected_users):  # allow selected users to train
                    verbose = user_id == chosen_verbose_user

                    # perform regularization using generated samples after the first communication round
                    user.train(
                        glob_iter_,
                        glob_iter)
                    # 每一次训练完都要更新client的向量和l2
                    param_vector = user.get_vector()
                    user.compute_l2_norm()
                    # for j, task_all in enumerate(user.test_data_per_task):
                    #     _logger.info(user_id,':测试集:',j,'数量:',len(task_all))

                # log training time
                curr_timestamp = time.time()
                train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
                self.metrics['user_train_time'].append(train_time)

                self.timestamp = time.time()  # log server-agg start time

                # =================
                # 2. Server update
                # =================
                # =================== Local ==================
                if args.local == 1:
                    acc = self.evaluate_per_client_per_task()
                    # _logger.info('all-----------',acc)

                # =================== FedAVG / FedProx / Fedlwf ==================
                elif args.fedavg == 1 or args.fedlwf == 1 or args.fedprox == 1:
                    # send parameteres: client -> server
                    # 聚合前
                    _logger.info('聚合前')
                    acc = self.evaluate_per_client_per_task()
                    self.aggregate_parameters(partial=False)
                    self.init_send_parameters()
                    # 聚合后
                    _logger.info('聚合后')
                    acc = self.evaluate_per_client_per_task()

                # =================== FedCL ===================
                elif args.l2c == 1:
                    _logger.info('聚合前')
                    acc = self.evaluate_per_client_per_task()
                    self.aggregate_parameter_l2c()
                    # 聚合后
                    _logger.info('聚合后')
                    acc = self.evaluate_per_client_per_task()
                # =================== Form a coalition ===================
                elif args.FCL == 1:
                    if args.offline == 0:
                        _logger.info('聚合前')
                        acc = self.evaluate_per_client_per_task()
                        # form coalition models
                        # 聚合的时候要重新更新相似度矩阵
                        similarity_matrix = self.cal_cos()
                        # print(similarity_matrix)
                        # 每一轮都要更新表
                        self.moodify_table()
                        # # 第一个全局轮：对每个状态遍历
                        if glob_iter == 0:
                            self.coalition_form()
                        else:
                            self.coalition_form_new()
                        self.form_coalition_models()
                        # send parameters from coalitions to clients
                        self.send_parameters(task)
                        _logger.info('聚合后')
                        acc = self.evaluate_per_client_per_task()
                        
                        # if glob_iter_ == epoch_per_task - 1:
                        #     df = pd.DataFrame(accs)
                        #     name = 'acc-64-' + str(task) + '.xlsx'
                        #     path = os.path.join(args.result,name)
                        #     df.to_excel(path, index=False)
                    else:
                        print('offline mode: No C -> S aggregating')

                curr_timestamp = time.time()  # log  server-agg end time
                agg_time = curr_timestamp - self.timestamp
                self.metrics['server_agg_time'].append(agg_time)
        return acc
        self.save_models(task, glob_iter)



    def get_balanced_samples(self, x_all, y_all, args):
        '''
        output: x and y
        '''
        x_balanced = y_balanced = []
        labels = set()

        # obtain label set:
        for i in y_all:
            labels.add(int(i))
        labels = list(labels)

        # dict of indexes of labels
        d = {}
        for label in labels:
            d[label] = []

        for index, label in enumerate(y_all):
            label = int(label)
            d[label].append(index)

        y_balanced = np.random.choice(labels, args.gen_batch_size)

        # pick up x data:
        for label in y_balanced:
            x_balanced.append(random.choice(d[label]))  # d[label] is empty

        x_balanced = x_all[x_balanced]

        return x_balanced, torch.tensor(y_balanced)

    def get_label_weights(self):

        label_weights = []
        qualified_labels = []
        for label in range(self.unique_labels):

            weights = []
            for user in self.selected_users:
                weights.append(user.label_counts[label])

            # weights:
            if np.max(weights) > MIN_SAMPLES_PER_LABEL:  # 1
                qualified_labels.append(label)

            # uniform
            if np.sum(weights) == 0:  # this label is unavailable for all clients this round
                label_weights.append(np.zeros(len(weights)))
            else:
                label_weights.append(np.array(weights) / np.sum(weights))

        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))

        return label_weights, qualified_labels

    def get_label_so_far_global(self):
        l = set()
        for u in self.selected_users:
            c = u.classes_so_far()
            l = l.union(set(c))

        return list(l)

    def get_label_weights_all(self):
        # build labels so far:
        # w: [num of labels, num of Clients]

        qualified_labels = []
        weights = []
        one_hots = []

        for u in self.users:
            qualified_labels.extend(u.classes_so_far)
            weights.append(u.classes_so_far)

        # make it unique
        qualified_labels = list(set(qualified_labels))

        # weights:
        for w in weights:
            one_hot = np.eye(self.unique_labels)[w].sum(axis=0)
            one_hots.append(one_hot)

        one_hots = np.array(one_hots)
        one_hots = np.transpose(one_hots)

        # normalize weights according to each label
        one_hots_sum = one_hots.sum(axis=1)

        # replace 0 with -1
        one_hots_sum[one_hots_sum == 0] = -1

        one_hots_sum = one_hots_sum.reshape(len(one_hots_sum), -1)

        one_hots = one_hots / one_hots_sum

        return one_hots, qualified_labels
