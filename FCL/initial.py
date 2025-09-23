from FCL.clients.clientfcl import ClientFCL
from FCL.clients.aggerate import aggeregator
from utils.model_utils import read_user_data_cl
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


class FCLInit(aggeregator):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        # Initialize data for all users
        self.data = get_dataset(args, args.dataset, args.datadir, args.data_split_file)
        data = self.data
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
        self.server_control = None
        self.client_control = None
        self.proto_global = None


        # scaffold----------------------
        if args.scaffold == 1:
            self.server_control = self.init_control(model)
            self.set_control_cuda(self.server_control, True)
            # init users with task = 0
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
        # ----------------scaffold-------------
        if args.scaffold == 1:
            self.client_control = {
                user.id: self.init_control(model) for user in self.users
            }
        self.create_table()

        # ==================
        # training devices:
        # ==================
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print('Using device: ', device)


    def train(self, args):
        delta_models = {}
        delta_controls = {}
        all_acc = []
        all_accs = []
        all_forget = []
        mask = None
        for task in range(args.ntask):
            # ===================
            # The First Task
            # ===================

            if task == 0:
                # update acc info.

                # update labels info. for the first task
                available_labels = set()
                available_labels_current = set()
                for u in self.users:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))

                for u in self.users:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)

                self.all_union = []

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

                # 每一个通信轮
                proto_locals = dict()
                radius_locals = dict()
                client_losses = []
                w_locals = list()

                glob_iter = glob_iter_ + epoch_per_task * task

                print("Round number:", glob_iter, " | Current task:", task)

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
                    # -----scaffold
                    if args.scaffold == 1:
                        self.set_control_cuda(self.client_control[user.id], True)
                    # perform regularization using generated samples after the first communication round
                    if args.AFCL == 1:
                        if self.args.mask_model:
                            mask = sorted([a * 4 + i for a in user.class_label for i in range(4)])
                        if self.proto_global:
                            self._update_proto_radius_labels(user)
                        result = user.train(
                            glob_iter_,
                            glob_iter, self.server_control, self.client_control, mask, round=glob_iter, proto_queue=self.proto_queue)
                        client_loss = result['loss_terms']
                        num_sample_class = result['num_sample_class']
                        client_losses.append(client_loss)

                        radius, prototype, class_label = user.proto_save(user.current_labels)
                        proto_locals[user_id] = {'sample_num': user.get_sample_number(),
                                             'prototype': prototype,
                                             'num_samples_class': num_sample_class}
                        radius_locals[user_id] = {'sample_num': user.get_sample_number(),
                                              'radius': radius}

                        user.prototype["local"] = {**user.prototype["local"], **prototype}
                        user.radius["local"] = radius

                        self.proto_queue.insert(prototype, radius, num_sample_class)

                        user.num_sample_class = num_sample_class
                        w_locals.append((user.get_sample_number(), copy.deepcopy(user.model), user_id))

                    else:
                        user.train(
                            glob_iter_,
                            glob_iter,self.server_control, self.client_control, mask, round=None, proto_queue=None
                              )
                    if args.FCL == 1:
                        user.compute_l2_norm()
                    if args.scaffold == 1:
                        self.client_control[user.id] = user.client_control
                        delta_models[user.id] = user.delta_model
                        delta_controls[user.id] = user.delta_control
                        self.set_control_cuda(self.client_control[user.id], False)

                # log training time
                curr_timestamp = time.time()
                train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)

                self.timestamp = time.time()  # log server-agg start time

                # =================
                # 2. Server update
                # =================
                # =================== Local ==================
                if args.local == 1:
                    accs, acc, num_all = self.evaluate_per_client_per_task()
                elif args.scaffold == 1:
                    self.model.to(self.device)
                    self.update_global(
                        global_model=self.model,
                        delta_models=delta_models,
                    )

                    new_control = self.update_global_control(
                        control=self.server_control,
                        delta_controls=delta_controls,
                    )
                    self.server_control = copy.deepcopy(new_control)

                    accs, acc, num_all = self.evaluate_per_client_per_task()
                    self.init_send_parameters()

                    accs, acc, num_all = self.evaluate_per_client_per_task()
                    all_acc.append(acc)
                # =================== FedAVG / FedProx / Fedlwf ==================
                elif args.fedavg == 1 or args.fedlwf == 1 or args.fedprox == 1:
                    # send parameteres: client -> server
                    accs, acc, num_all = self.evaluate_per_client_per_task()
                    self.aggregate_parameters(partial=False)
                    self.init_send_parameters()
                    accs, acc, num_all = self.evaluate_per_client_per_task()
                    all_acc.append(acc)
                    if glob_iter_ == epoch_per_task - 1:
                        all_accs.append(accs)
                        if task!=0:
                            forget_rate = self.evaluate_forget(all_accs, num_all)
                            all_forget.append(forget_rate)
                elif args.peravg == 1 or args.pfedme == 1:
                    accs, acc, num_all = self.evaluate_per_client_per_task()
                    if args.pfedme == 1:
                        self.persionalized_aggregate()
                    elif args.peravg == 1:
                        self.peravg_aggregate()
                    else:
                        self.aggregate_parameters(partial=False)
                    self.init_send_parameters()
                    accs, acc, num_all = self.evaluate_per_client_per_task()
                    print('-'*108)
                    print(acc)
                    all_acc.append(acc)
                    if glob_iter_ == epoch_per_task - 1:
                        all_accs.append(accs)
                        if task!=0:
                            forget_rate = self.evaluate_forget(all_accs, num_all)
                            all_forget.append(forget_rate)

                # =================== FedCL ===================
                elif args.l2c == 1:
                    accs, acc, num_all = self.evaluate_per_client_per_task()
                    self.aggregate_parameter_l2c()
                    accs, acc, num_all = self.evaluate_per_client_per_task()
                    all_acc.append(acc)
                    if glob_iter_ == epoch_per_task - 1:
                        all_accs.append(accs)
                        if task!=0:
                            forget_rate = self.evaluate_forget(all_accs, num_all)
                            all_forget.append(forget_rate)
                # =================== Form a coalition ===================
                elif args.FCL == 1:
                    if args.offline == 0:
                        accs, acc, num_all = self.evaluate_per_client_per_task()
                        curr_timestamp = time.time()
                        similarity_matrix = self.cal_cos()
                        self.moodify_table()
                        if glob_iter == 0:
                            self.coalition_form()
                        else:
                            self.coalition_form_new()
                        self.form_coalition_models()
                        self.all_union.append(self.unions)
                        # send parameters from coalitions to clients
                        self.send_parameters(task)
                        accs, acc, num_all = self.evaluate_per_client_per_task()
                        all_acc.append(acc)
                        if glob_iter_ == epoch_per_task - 1:
                            all_accs.append(accs)
                            if task != 0:
                                forget_rate = self.evaluate_forget(all_accs, num_all)
                                all_forget.append(forget_rate)

                elif args.ClusterFL == 1:
                    accs, acc, num_all = self.evaluate_per_client_per_task()
                    # form coalition models
                    curr_timestamp = time.time()
                    similarity_matrix = self.cal_cos()
                    # print(similarity_matrix)
                    op_par = self.optimal_bipartition(similarity_matrix)
                    print(op_par)
                    self.form_coalition_models()
                    # send parameters from coalitions to clients
                    self.send_parameters(task)
                    accs, acc, num_all = self.evaluate_per_client_per_task()
                    all_acc.append(acc)
                    if glob_iter_ == epoch_per_task - 1:
                        all_accs.append(accs)
                        if task != 0:
                            forget_rate = self.evaluate_forget(all_accs, num_all)

                elif args.AFCL == 1:
                    print(f"Aggregating model weights...")

                    w_global = self._aggregate_w(w_locals)

                    self.afcl_aggregate(args.global_weight, self.model, w_global)
                    self.model=copy.deepcopy(w_global)


                    if self.args.aggregate_proto_by_class:
                        self.proto_global = self._aggregate_proto_by_class(proto_locals)
                    else:
                        self.proto_global = self._aggregate_proto(proto_locals)
                    self.radius_global = self._aggregate_radius(radius_locals)
                    self.init_send_parameters()

                    accs, acc, num_all = self.evaluate_per_client_per_task()
                    # form coalition models
                    curr_timestamp = time.time()
                    similarity_matrix = self.cal_cos()
                    # print(similarity_matrix)
                    op_par = self.optimal_bipartition(similarity_matrix)
                    print(op_par)
                    self.form_coalition_models()
                    # send parameters from coalitions to clients
                    self.send_parameters(task)
                    accs, acc, num_all = self.evaluate_per_client_per_task()
                    all_acc.append(acc)
                    if glob_iter_ == epoch_per_task - 1:
                        all_accs.append(accs)
                        if task != 0:
                            forget_rate = self.evaluate_forget(all_accs, num_all)

        print(all_acc)
        print(num_all)
        #
        # f = pd.DataFrame(all_acc)
        # name = 'acc' + '-' + args.dataset + '-' + 'FCL' + '-' + str(args.sw) + '.xlsx'
        # df.to_excel(name, index=False)
        # df = pd.DataFrame(all_forget)
        # name = 'forget' + '-' + args.dataset + '-' + 'FCL' + '-' + str(args.sw) + '.xlsx'
        # df.to_excel(name, index=False)
        # if args.FCL == 1:
        #     df = pd.DataFrame(self.all_union)
        #     name = 'union' + '-' + args.dataset + '-' + 'FCL' + '-' + str(args.sw) + '.xlsx'
        #     df.to_excel(name, index=False)

        return acc, forget_rate


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
