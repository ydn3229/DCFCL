import torch
import torch.nn.functional as F
import numpy as np
from FCL.clients.clientbase import Client
import copy
from torch.utils.tensorboard import SummaryWriter
import math

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
        self.client_control = None
        self.delta_model = None
        self.delta_control = None
    # ==================================== FCL as clients ================================

    def train(
            self,
            glob_iter_,
            glob_iter,
            server_control, client_control,
            mask, round=None, proto_queue=None,
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
        if self.args.FCL == 1 or self.args.local == 1 or self.args.l2c == 1 or self.args.ClusterFL == 1:
            # para_before = self.get_vector()
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



        # =============================================== Fedavg ===============================================
        elif self.args.fedavg == 1:

            for iteration in range(self.local_epochs):
                samples = self.get_next_train_batch(count_labels=True)
                x, y = samples['X'].to(device), samples['y'].to(device)

                result = self.train_a_batch_all(
                        x=x, y=y,
                        device=device)
                # print('-----------acc------------',result['acc_rate'])
                c_loss_all += result['loss']
            c_loss_avg = c_loss_all / self.local_epochs
            # print('-----------loss-all-------------','client', self.id, '---------',c_loss_avg)

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
        elif self.args.scaffold == 1:
            glo_model = copy.deepcopy(self.model)
            para_before = self.get_vector()
            for iteration in range(self.local_epochs):
                samples = self.get_next_train_batch(count_labels=True)
                x, y = samples['X'].to(device), samples['y'].to(device)

                result = self.train_a_batch_scaf(x, y,device,
                          server_control, client_control,
                            current_task = self.current_task,
                            last_copy = self.last_copy
                )

                c_loss_all += result['loss']
            para_after = self.get_vector()
            self.differ = para_after - para_before
            print(self.differ)
            glo_model.to(device)
            delta_model = self.get_delta_model(glo_model, self.model)
            client_control, delta_control = self.update_local_control(
                    delta_model=delta_model,
                    server_control=server_control,
                    client_control=client_control[self.id],
                    steps=self.local_epochs,
                    lr=self.args.learning_rate,
                )
            self.client_control = copy.deepcopy(client_control)
            self.delta_model = copy.deepcopy(delta_model)
            self.delta_control = copy.deepcopy(delta_control)

        #     ------------------------------------ peravg ----------------------------------------------
        elif self.args.peravg == 1:

            for iteration in range(self.local_epochs):
                temp_model = copy.deepcopy(self.model)
                samples = self.get_next_train_batch(count_labels=True)
                x, y = samples['X'].to(device), samples['y'].to(device)

                result = self.train_a_batch_peravg_1(
                        x=x, y=y,
                        device=device,
                    current_task=self.current_task,
                    last_copy=self.last_copy
                )

                samples = self.get_next_train_batch(count_labels=True)
                x, y = samples['X'].to(device), samples['y'].to(device)

                result = self.train_a_batch_peravg_2(
                    x=x, y=y,
                    device=device, temp_model=temp_model,
                    current_task=self.current_task,
                    last_copy=self.last_copy)

                c_loss_all += result['loss']
            c_loss_avg = c_loss_all / self.local_epochs
            # print('-----------loss-all-------------','client', self.id, '---------',c_loss_avg)

    # =============================================== pFedMe ===============================================
        elif self.args.pfedme == 1:
            for iteration in range(self.local_epochs):
                samples = self.get_next_train_batch(count_labels=True)
                x, y = samples['X'].to(device), samples['y'].to(device)

                result = self.train_a_batch_pFedMe(
                    x=x, y=y,
                    device=device)
                # print('-----------acc------------',result['acc_rate'])
                c_loss_all += result['loss']
            c_loss_avg = c_loss_all / self.local_epochs
            self.set_parameters(self.local_model)
            # print('-----------loss-all-------------', 'client', self.id, '---------', c_loss_avg)


    # =============================================== AFCL ===============================================
        elif self.args.AFCL == 1:
            LOSS_KEYS = ["Proto_aug_loss"]
            epoch_loss = []
            loss_terms = epoch_loss_terms = {name: [] for name in LOSS_KEYS}
            num_sample_class = {k: 0 for k in range(100)}
            temp_model = copy.deepcopy(self.model)
            for iteration in range(self.local_epochs):

                samples = self.get_next_train_batch(count_labels=True)
                x, y = samples['X'].to(device), samples['y'].to(device)
                for lab in y.tolist():
                    num_sample_class[lab] += 1
                result = self.train_a_batch_afcl(
                    images=x, labels=y,
                    device=device,args = self.args, old_model = temp_model, mask = mask, epoch_loss=epoch_loss, round = round, proto_queue = proto_queue)
                # print('-----------acc------------',result['acc_rate'])
                loss = result['loss']
                epoch_loss = result['epoch_loss']
                for key in LOSS_KEYS:
                    epoch_loss_terms[key].append(loss[key].item())
            for key in LOSS_KEYS:
                loss_terms[key].append(np.mean(epoch_loss_terms[key]))
            return loss_terms, num_sample_class

            # print('-----------loss-all-------------', 'client', self.id, '---------', c_loss_avg)

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

    def update_local_control(
            self, delta_model, server_control,
            client_control, steps, lr):
        new_control = copy.deepcopy(client_control)
        delta_control = copy.deepcopy(client_control)

        for name in delta_model.keys():
            c = server_control[name]
            ci = client_control[name]
            delta = delta_model[name]

            new_ci = ci.data - c.data + delta.data / (steps * lr)
            new_control[name].data = new_ci
            delta_control[name].data = ci.data - new_ci
        return new_control, delta_control

    def update_global(self, r, global_model, delta_models):
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

    def update_global_control(self, r, control, delta_controls):
        new_control = copy.deepcopy(control)
        for name, c in control.items():
            mean_ci = []
            for _, delta_control in delta_controls.items():
                mean_ci.append(delta_control[name])
            ci = torch.stack(mean_ci).mean(dim=0)
            new_control[name] = c - ci
        return new_control

    def proto_save(self, current_task_classes, device):
        features = []
        labels = []
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            for iteration in range(self.local_epochs):
                samples = self.get_next_train_batch(count_labels=True)
                x, y = samples['X'].to(device), samples['y'].to(device)
                feature = self.model.feature(x.to(device))
                labels.append(y.numpy())
                features.append(feature.cpu().numpy())

        labels = np.concatenate([label_vector for label_vector in labels])
        features = np.concatenate([feature_vector for feature_vector in features], axis=0)
        feature_dim = features.shape[1]

        prototype = {}
        radius = {}
        class_label = []
        for item in current_task_classes:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]

            if np.size(feature_classwise) == 0:
                prototype[item] = np.zeros(self.feature_size,)
            else:
                prototype[item] = np.mean(feature_classwise, axis=0)

            if not self.prototype["local"] or (self.args.proto_queue and self.args.multi_radius):
                cov = np.cov(feature_classwise.T)
                if not math.isnan(np.trace(cov)):
                    radius[item] = np.trace(cov) / feature_dim
                else:
                    radius[item] = 0

        if self.radius["local"] and ((self.args.proto_queue and self.args.multi_radius) is False):
            radius = copy.deepcopy(self.radius["local"])
        else:
            radius = np.sqrt(np.mean(list(radius.values())))

        self.model.train()
        return radius, prototype, class_label

    def get_proto(self):
        return self.prototype

    def get_radius(self):
        return self.radius

    def get_class_label(self):
        return self.class_label

    def get_sample_number(self):
        return self.local_sample_number