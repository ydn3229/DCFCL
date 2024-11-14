import torch
import os
import numpy as np
import h5py
import copy
import torch.nn.functional as F
import time
import torch.nn as nn

from torch import optim
from utils.util import *
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.parameter import Parameter

_logger = logging.getLogger('train')

class aggeregator:
    def __init__(self, args, model, seed):

        # Set up the main attributes
        self.dataset = args.dataset
        self.num_glob_iters = args.num_glob_iters
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.total_train_samples = 0
        self.K = args.K
        self.args = args

        self.model = copy.deepcopy(model)

        self.users = []
        self.selected_users = []
        self.num_users = args.num_users
        self.algorithm = args.algorithm
        self.personalized = 'pFed' in self.algorithm
        self.mode = 'partial' if 'partial' in self.algorithm.lower() else 'all'
        self.seed = seed
        self.deviations = {}
        self.timestamp = None
        self.save_path = args.result_path
        os.system("mkdir -p {}".format(self.save_path))
        self.aggregators = {}
        self.similarity_matrix = [[]]
        self.beta = args.beta

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)

        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1, 0.02)
            m.bias.data.fill_(0)

    def init_configs(self):
        self.n_teacher_iters = 5
        self.n_student_iters = 1
        self.unions = None

    def if_personalized(self):
        return 'pFed' in self.algorithm or 'PerAvg' in self.algorithm

    def if_ensemble(self):
        return 'FedE' in self.algorithm

    # ========================== dcFCL aggregate code ==============================
    # init aggregator with zero weight
    def zero_model_parameters(self, model):
        for param in model.parameters():
            param.data.fill_(0)

    def init_aggregators(self, unions):
        assert (self.selected_users is not None and len(self.selected_users) > 0)
        self.aggregators = {}
        us = len(unions)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for i in range(us):
            self.aggregators[i] = copy.deepcopy(self.model)
            self.aggregators[i].to(device)
            self.zero_model_parameters(self.aggregators[i])
            
    def aggregate_parameter_l2c(self):
        for user in self.selected_users:
            self.zero_model_parameters(user.model.classifier)
            user.model.alpha.data = torch.softmax(user.model.alpha, dim=0)
            # user.model.alpha.data = torch.softmax(user.model.alpha, dim=0)
            for i, user_o in enumerate(self.selected_users):
                self.add_parameters_l2c(user, user_o, user.model.alpha[i])
            self.update_alpha(user)
    
    def update_alpha(self, user):
        loss= user.test()
        _logger.info(user.model.alpha)
        user.model.alpha_optimizer.zero_grad()
        # _logger.info('-'*108)
        # _logger.info(loss)
        loss.backward()
        user.model.alpha_optimizer.step()
        _logger.info(user.model.alpha)

    def add_parameters_l2c(self, user, user_o, ratio):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for server_param, user_param in zip(user.model.classifier.parameters(), user_o.model.classifier.parameters()):
            self.model.to(device)

            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_union(self, union_id):
        total_train = 0
        for client_id in self.unions[union_id]:
            client = self.users[client_id]
            total_train += client.train_samples
        for client_id in self.unions[union_id]:
            client = self.users[client_id]
            ratio = client.train_samples / total_train
            print(ratio)
            for agg_param, user_param in zip(self.aggregators[union_id].parameters(), client.model.parameters()):

                # print(agg_param.data)
                # print(user_param.data)
                agg_param.data = agg_param.data + user_param.data.cuda() * ratio

    def form_coalition_models(self,partial=False):
        # self.unions = ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9),)
        print(self.unions)
        self.init_aggregators(self.unions)
        for union_id, union in enumerate(self.unions):
            print(union_id)
            print(union)
            all_zeros = True
            for param in self.aggregators[union_id].parameters():
                if not torch.all(param == 0):
                    all_zeros = False
                    break
            # print(all_zeros)
            self.aggregate_union(union_id)
            # print(self.aggregators[union_id].parameters())

    def send_parameters(self, task=None, beta=1, selected=False):
        for union_id in range(len(self.unions)):
            for client_id in self.unions[union_id]:
                client = self.users[client_id]
                client.set_parameters(self.aggregators[union_id], beta=beta)

    def init_send_parameters(self, beta=1):
        for client in self.users:
            client.set_parameters(self.model, beta=beta)

    # ================================ 更新相似度矩阵 ===============================
    def cal_cos(self):
        sw = self.args.sw
        n = self.num_users
        self.similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                m_1_differ= self.users[i].differ
                m_2_differ = self.users[j].differ
                similarity_differ = cosine_similarity(m_1_differ.reshape(1, -1), m_2_differ.reshape(1, -1))[0][0]
                m_1_vector= self.users[i].get_vector()
                m_2_vector = self.users[j].get_vector()
                similarity_vector = cosine_similarity(m_1_vector.reshape(1, -1), m_2_vector.reshape(1, -1))[0][0]
                if self.args.ClusterFL == 1:
                    similarity = similarity_differ
                else:
                    similarity = similarity_differ + sw * similarity_vector
                self.similarity_matrix[i][j] = similarity
                self.similarity_matrix[j][i] = similarity
        return self.similarity_matrix

    def draw_matrix(self, matrix):
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, cmap='viridis')
        plt.title('Heatmap of the Matrix')
        plt.show()



    # ========================== create table ================================
    def cal_partitions(self,set_):
        if not set_:
            yield []
            return
        first = set_.pop()
        for smaller in self.cal_partitions(set_):
            for n, subset in enumerate(smaller):
                yield smaller[:n] + [subset | {first}] + smaller[n + 1:]
            yield [{first}] + smaller

    def all_part(self):
        n = self.num_users
        clients = set(np.arange(0, n))
        self.all_partitions = list(self.cal_partitions(clients))


    def create_table(self):
        self.all_part()
        self.pay_table = {}
        for part in self.all_partitions:
            part = tuple(tuple(p) for p in part)
            self.pay_table[part] = [0] * self.num_users

    def cal_cos_n(self, union, c_id):
        weight = 1.0 / len(union)
        union = list(set(union) - {c_id})
        item_1 = 0
        # 计算 客户端互相关相似度
        for i in range(len(union)-1):
            for j in range(i+1, len(union)):
                client_i = union[i]
                client_j = union[j]
                item_1 += 2 * self.users[client_i].l2_norm * self.users[client_j].l2_norm * self.similarity_matrix[client_i][client_j]
        item_2 = 0
        item_3 = 0
        for i in union:
            item_2 += self.users[i].l2_norm * self.similarity_matrix[i][c_id]
            item_3 += self.users[i].l2_norm ** 2
        # reward = cos(w_avg, w_cid)
        reward = item_2 / (item_1 + item_3) ** 0.5

        return reward


    def moodify_table(self):
        for index, (part, list_client) in enumerate(self.pay_table.items()):
            for union in part:
                if len(union) >1:
                    for c_id in union:
                        list_client[c_id] = self.cal_cos_n(union, c_id)
                else:
                    list_client[union[0]] = 0

    # ================================= EPCF 过程 ===========================================
    def single_step_transfer(self, s_id, state, state_list_client):
        next_state = s_id
        for index, (part, list_client) in enumerate(self.pay_table.items()):
            if part!=state:
                # case1
                for i in range(self.num_users):
                    if (i,) in part:
                        if self.single_reward[i] > self.pay_table[state][i]:
                            next_state = index
                            return next_state
                # case2
                for union in part:
                    union_reward = [list_client[u] for u in union]
                    state_reward = [state_list_client[u] for u in union]
                    all_ge = all(x >= y for x, y in zip(union_reward, state_reward))
                    at_least_one_gt = any(x > y for x, y in zip(union_reward, state_reward))
                    if all_ge and at_least_one_gt:
                        next_state = index
                        return next_state
        return next_state

    def find_absorbing_states(self, trans_list):
        n = len(trans_list)
        absorbing_states = []

        for i in range(n):
            if trans_list[i][0] == trans_list[i][1]:
                absorbing_states.append(i)

        return absorbing_states

    def coalition_form(self):
        trans_list = []
        self.single_reward = [0]*self.num_users
        for index, (state, list_client) in enumerate(self.pay_table.items()):
            next_state = self.single_step_transfer(index, state, list_client)
            trans_list.append((index, next_state))
        absorbing_states = self.find_absorbing_states(trans_list)
        self.stable_state = None
        pay_list = list(self.pay_table.keys())
        if len(absorbing_states)>1:
            warefare = 0
            for s_id in absorbing_states:
                state = pay_list[s_id]
                list_client = self.pay_table[state]
                w = sum(list_client)
                if w > warefare:
                    warefare = w
                    self.stable_state = s_id
        else:
            self.stable_state = absorbing_states[0]
        self.unions = pay_list[self.stable_state]
        _logger.info(self.unions)
    def coalition_form_new(self):
        self.last_stable_state = self.stable_state
        before_state = self.last_stable_state
        pay_list = list(self.pay_table.items())
        state, state_list_client = pay_list[before_state]
        next_state = self.single_step_transfer(before_state, state, state_list_client)
        i = 0
        while next_state != before_state and i<100:
            i+=1
            before_state = next_state
            state, state_list_client = pay_list[before_state]
            next_state = self.single_step_transfer(before_state, state, state_list_client)
            # print(next_state)
        self.stable_state = next_state
        self.unions = pay_list[self.stable_state][0]
        _logger.info(self.unions)

    # ================================== Fedavg aggregation model ================================
    def add_parameters(self, user, ratio):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for server_param, user_param in zip(self.model.parameters(), user.model.parameters()):
            self.model.to(device)

            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self, partial=False):
        assert (self.selected_users is not None and len(self.selected_users) > 0)
        self.zero_model_parameters(self.model)
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples  # length of the train data for weighted importance

        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)


    # ===================================== commonly used ======================================

    def select_users(self, round, num_users, return_idx=False):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        Return:
            list of selected clients objects
        '''
        if (num_users == len(self.users)):
            print("All users are selected")
            return self.users, [i for i in range(len(self.users))]

        num_users = min(num_users, len(self.users))
        if return_idx:
            user_idxs = np.random.choice(range(len(self.users)), num_users, replace=False)  # , p=pk)
            return [self.users[i] for i in user_idxs], user_idxs
        else:
            return np.random.choice(self.users, num_users, replace=False)

    def init_loss(self):
        self.ce_loss = nn.CrossEntropyLoss()

    def test(self, selected=False):
        '''
        tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        users = self.selected_users if selected else self.users
        for c in users:
            ct, c_loss, ns = c.test()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(c_loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses

    def test_all(self, selected=False):
        '''
        tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        users = self.selected_users if selected else self.users
        for c in users:
            ct, c_loss, ns = c.test_all()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(c_loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses

    def test_per_task(self, selected=False):
        '''
        tests latest model on leanrt tasks
        '''
        accs = {}
        nss = {}
        users = self.selected_users if selected else self.users
        acc_num = 0
        all_num = 0
        for c in users:
            accs[c.id] = []

            ct, c_loss, ns = c.test_per_task()
            nss[c.id] = ns
            # per past task:
            for task in range(len(ct)):
                acc = ct[task] / ns[task]
                accs[c.id].append(acc)
                acc_num += ct[task]
                all_num += ns[task]
        acc = acc_num / all_num

        return accs, acc, nss


    def evaluate(self, save=True, selected=False):
        # override evaluate function to log vae-loss.
        test_ids, test_samples, test_accs, test_losses = self.test(selected=selected)

        glob_acc = np.sum(test_accs) * 1.0 / np.sum(test_samples)

        glob_loss = np.sum(
            [x * y.detach().to(torch.device('cpu')) for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(
            test_samples)
        print("Average Global Accuracy = {:.4f}, Loss = {:.2f}.".format(glob_acc, glob_loss))

    def evaluate_all(self, save=True, selected=False):
        # override evaluate function to log vae-loss.
        test_ids, test_samples, test_accs, test_losses = self.test_all(selected=selected)

        glob_acc = np.sum(test_accs) * 1.0 / np.sum(test_samples)
        glob_loss = np.sum(
            [x * y.detach().to(torch.device('cpu')) for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(
            test_samples)

        print("Average Global Accuracy (classes so far) = {:.4f}, Loss = {:.2f}.".format(glob_acc, glob_loss))

    def evaluate_per_client_per_task(self, save=True, selected=False):
        accs, acc, ns = self.test_per_task()

        for k, v in accs.items():
            _logger.info(k)
            _logger.info(v)
        
        return accs, acc, ns

    def write(self, accuracy, file=None, mode='a'):
        with open(file, mode) as f:
            line = str(accuracy) + '\n'
            f.writelines(line)

    def save_models(self, task, epoch):
        for client in self.users:
            client.save_model(task, epoch)

    def evaluate_forget(self, all_acc, num_all):
        sum_num = 0
        sum_fr = 0
        users = self.users
        for c in users:
            id = c.id
            # print('-'*108)
            # print(id)
            for task in range(len(num_all[id])-1):
                akti = []
                nki = num_all[id][task]
                # print(nki)
                akTi = all_acc[-1][id][task]
                # print(akTi)
                for t in range(task, len(num_all[id])-1):
                    akti.append(all_acc[t][id][task])
                    # print(akti)
                    max_akti = max(akti)
                    # print(max_akti)
                sum_fr += (max_akti - akTi) * nki
                sum_num += nki

        forget_rate = sum_fr / sum_num

        return forget_rate


    # ------------------------------------------scaffold-----------------------------------------------------------

    def set_control_cuda(self, control, cuda=True):
        for name in control.keys():
            if cuda is True:
                control[name] = control[name].cuda()
            else:
                control[name] = control[name].cpu()

    def init_control(self, model):
        """ a dict type: {name: params}
        """
        control = {
            name: torch.zeros_like(
                p.data
            ).cpu() for name, p in model.state_dict().items()
        }
        return control
    def get_delta_model(self, model0, model1):
        """ return a dict: {name: params}
        """
        state_dict = {}
        for name, param0 in model0.state_dict().items():
            param1 = model1.state_dict()[name]
            state_dict[name] = param0.detach() - param1.detach()
        return state_dict

    def update_global(self, global_model, delta_models):
        state_dict = {}

        for name, param in global_model.state_dict().items():
            vs = []
            for client in delta_models.keys():
                vs.append(delta_models[client][name])
            vs = torch.stack(vs, dim=0)

            try:
                mean_value = vs.mean(dim=0)
                vs = param - self.args.glo_lr * mean_value
            except Exception:
                # for BN's cnt
                mean_value = (1.0 * vs).mean(dim=0).long()
                vs = param - self.args.glo_lr * mean_value
                vs = vs.long()

            state_dict[name] = vs

        global_model.load_state_dict(state_dict, strict=True)

    def update_global_control(self, control, delta_controls):
        new_control = copy.deepcopy(control)
        for name, c in control.items():
            mean_ci = []
            for _, delta_control in delta_controls.items():
                mean_ci.append(delta_control[name])
            ci = torch.stack(mean_ci).mean(dim=0)
            new_control[name] = c - ci
        return new_control



    # ----------------------------------------------ClusterFL----------------------------------------
    def optimal_bipartition(self, similarity_matrix):

        M = similarity_matrix.shape[0]

        # Step 3: Get sorted indices in descending order based on the similarity matrix
        s = np.argsort(-similarity_matrix.flatten())

        # Step 4: Initialize each element as its own cluster
        C = [{i} for i in range(M)]

        # Step 5: Iterate through all sorted index pairs
        for idx in s:
            i1, i2 = divmod(idx, M)  # Convert linear index to matrix indices
            c_tmp = set()

            # Step 8: Check if i1 or i2 are in any existing clusters
            for c in C:
                if i1 in c or i2 in c:
                    c_tmp = c_tmp.union(c)  # Merge clusters
                    C.remove(c)  # Remove from existing clusters

            # Step 12: Add the merged cluster back to the list of clusters
            C.append(c_tmp)

            # Step 13-15: If we have exactly two clusters, return them
            if len(C) == 2:
                self.unions = C
                return C


    # --------------------------------------------- PFedMe ---------------------------------------------
    def persionalized_aggregate(self):
        # store previous parameters
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        previous_param = copy.deepcopy(self.model)
        previous_param.to(device)
        self.zero_model_parameters(self.model)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples

        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
            #self.add_parameters(user, 1 / len(self.selected_users))

        # aaggregate avergage model with previous model using parameter beta
        for pre_param, param in zip(previous_param.classifier.parameters(), self.model.classifier.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data


    # --------------------------------------------- PFedMe ---------------------------------------------
    def peravg_aggregate(self):

        assert (self.selected_users is not None and len(self.selected_users) > 0)
        self.zero_model_parameters(self.model)
        ratio = 1.0 / len(self.selected_users)
        for user in self.selected_users:
            self.add_parameters(user, ratio)
