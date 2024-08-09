from torch import nn
import torch
from torch import optim
import glog as logger
import numpy as np
import torch.nn.functional as F

from FLAlgorithms.PreciseFCLNet.classify_net import S_ConvNet, Resnet_plus
from nflows.flows.base import Flow
from nflows.transforms.permutations import RandomPermutation, ReversePermutation
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.nn.nets.myresnet import ResidualNet
from torch.nn import functional as F
from nflows.distributions.normal import StandardNormal
from utils.utils import myitem

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

class PreciseModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        beta1 = args.beta1
        beta2 = args.beta2
        weight_decay = args.weight_decay
        lr = args.lr
        flow_lr = args.flow_lr
        c_channel_size = args.c_channel_size
        dataset = args.dataset

        self.algorithm = args.algorithm

        self.k_loss_flow = args.k_loss_flow
        self.k_kd_global_cls = args.k_kd_global_cls
        self.k_kd_last_cls = args.k_kd_last_cls
        self.k_kd_feature = args.k_kd_feature
        self.k_kd_output = args.k_kd_output
        self.k_flow_lastflow = args.k_flow_lastflow

        self.flow_explore_theta = args.flow_explore_theta
        self.fedprox_k = args.fedprox_k

        self.classify_criterion = nn.NLLLoss()
        self.classify_criterion_noreduce = nn.NLLLoss(reduction='none')

        self.flow = None
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


        self.classifier_optimizer = optim.Adam(
            self.classifier.parameters(),
            lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
        )

        parameters_fb = [a[1] for a in filter(lambda x: 'fc2' in x[0], self.classifier.named_parameters())]
        self.classifier_fb_optimizer = optim.Adam(
            parameters_fb, lr=lr, weight_decay=weight_decay, 
            betas=(beta1, beta2),
        )


        class_params = sum(p.numel() for p in self.classifier.parameters())


    def to(self, device):
        self.classifier.to(device)
        if self.algorithm=='PreciseFCL':
            self.flow.to(device)
        return self

    def parameters(self):
        for param in  self.classifier.parameters():
            yield param
        if self.algorithm=='PreciseFCL':
            for param in  self.flow.parameters():
                yield param

    def named_parameters(self):
        for name, param in self.classifier.named_parameters():
            yield 'classifier.'+name, param
        if self.algorithm=='PreciseFCL':
            for name, param in self.flow.named_parameters():
                yield 'flow.'+name, param

    def get_1d_nflow_model(self,
                        feature_dim, 
                        hidden_feature, 
                        context_feature,
                        num_layers):
        transforms = []
        
        for l in range(num_layers):
            assert num_layers//2>1
            if l < num_layers//2:
                transforms.append(ReversePermutation(features=feature_dim))
            else:
                transforms.append(RandomPermutation(features=feature_dim))
            
            mask = (torch.arange(0, feature_dim)>=(feature_dim//2)).float()
            # net_func = lambda in_d, out_d: MLP(in_shape=[in_d], out_shape=[out_d],
            #                                     hidden_sizes=[hidden_feature]*3, activation=F.leaky_relu)
            net_func = lambda in_d, out_d: ResidualNet(in_features=in_d, out_features=out_d,
                                                hidden_features=hidden_feature, context_features=context_feature,
                                                num_blocks=2, activation=F.leaky_relu, dropout_probability=0)
            transforms.append(AffineCouplingTransform(mask=mask, transform_net_create_fn=net_func))
        
        transform = CompositeTransform(transforms)
        base_dist = StandardNormal(shape=[feature_dim])
        flow = Flow(transform, base_dist)
        return flow

    def train_a_batch(self,
                        x, y,
                        train_flow,
                        flow,
                        last_flow,
                        last_classifier,
                        global_classifier,
                        classes_so_far,
                        classes_past_task,
                        available_labels,
                        available_labels_past):
        
        # ===================
        # 1. prediction loss
        # ====================
        if not train_flow:
            return self.train_a_batch_classifier(x, y, flow, last_classifier, global_classifier, classes_past_task, available_labels)
        else:
            return self.train_a_batch_flow(x, y, last_flow, classes_so_far, available_labels_past)

    def sample_from_flow(self, flow, labels, batch_size):
        label = np.random.choice(labels, batch_size)
        class_onehot = np.zeros((batch_size, self.num_classes))
        class_onehot[np.arange(batch_size), label] = 1
        class_onehot = torch.Tensor(class_onehot).cuda()
        flow_xa = flow.sample(num_samples=1, context=class_onehot).squeeze(1)
        flow_xa = flow_xa.detach()
        return flow_xa, label, class_onehot

    def probability_in_localdata(self, xa_u, y, prob_mean, flow_xa, flow_label):
        flow_xa_label_set = set(flow_label)
        flow_xa_prob = torch.zeros([flow_xa.shape[0]], device=flow_xa.device)
        for flow_yi in flow_xa_label_set:
            if (y==flow_yi).sum()>0:
                xa_u_yi = xa_u[y==flow_yi]
                xa_u_yi_mean = torch.mean(xa_u_yi, dim=0, keepdim=True)
                xa_u_yi_var = torch.mean((xa_u_yi-xa_u_yi_mean)*(xa_u_yi-xa_u_yi_mean), dim=0, keepdim=True)

                flow_xa_yi = flow_xa[flow_label==flow_yi]
                prob_xa_yi_ = 1/np.sqrt(2*np.pi)*torch.pow(xa_u_yi_var+eps, -0.5)*torch.exp(-torch.pow(flow_xa_yi-xa_u_yi_mean, 2)*torch.pow(xa_u_yi_var+eps, -1)*0.5)
                prob_xa_yi = torch.mean(prob_xa_yi_, dim=1)
                flow_xa_prob[flow_label==flow_yi] = prob_xa_yi
            else:
                flow_xa_prob[flow_label==flow_yi] = prob_mean
        return flow_xa_prob
        
    def train_a_batch_classifier(self, x, y, flow, last_classifier, global_classifier, classes_past_task, available_labels):
        
        if self.algorithm=='PreciseFCL' and type(flow)!=type(None) and self.k_loss_flow>0:
            batch_size = x.shape[0]

            with torch.no_grad():
                _, xa, _ = self.classifier(x)
                xa = xa.reshape(xa.shape[0], -1)

                y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
                log_prob, xa_u = flow.log_prob_and_noise(xa, y_one_hot)
                log_prob = log_prob.detach()
                xa_u = xa_u.detach()
                prob_mean = torch.exp(log_prob/xa.shape[1]).mean()+eps

                flow_xa, label, _ = self.sample_from_flow(flow, available_labels, batch_size)
                flow_xa_prob = self.probability_in_localdata(xa_u, y, prob_mean, flow_xa, label)
                flow_xa_prob = flow_xa_prob.detach()
                flow_xa_prob_mean = flow_xa_prob.mean()

            flow_xa = flow_xa.reshape(flow_xa.shape[0], *self.xa_shape)
            softmax_output_flow, _ = self.classifier.forward_from_xa(flow_xa)
            c_loss_flow_generate = (self.classify_criterion_noreduce(torch.log(softmax_output_flow+eps), torch.Tensor(label).long().cuda())*flow_xa_prob).mean()
            # c_loss_flow_generate = self.classify_criterion(torch.log(softmax_output_flow+eps), torch.Tensor(label).long().cuda())
            k_loss_flow_explore_forget = (1-self.flow_explore_theta)*prob_mean+self.flow_explore_theta

            kd_loss_output_last_flow, kd_loss_output_global_flow = self.knowledge_distillation_on_output(flow_xa, softmax_output_flow, last_classifier, global_classifier)
            kd_loss_flow = (kd_loss_output_last_flow + kd_loss_output_global_flow)*self.k_kd_output

            c_loss_flow = (c_loss_flow_generate*k_loss_flow_explore_forget + kd_loss_flow)*self.k_loss_flow
            
            self.classifier_fb_optimizer.zero_grad()
            c_loss_flow.backward()
            self.classifier_fb_optimizer.step()
        else:
            prob_mean = 0.0
            c_loss_flow = 0.0
            kd_loss_flow = 0.0
            flow_xa_prob_mean = 0.0

        softmax_output, xa, logits = self.classifier(x)
        
        c_loss_cls = self.classify_criterion(torch.log(softmax_output+eps), y)

        if self.algorithm=='PreciseFCL':
            kd_loss_feature_last, kd_loss_output_last, kd_loss_feature_global, kd_loss_output_global = \
                                    self.knowledge_distillation_on_xa_output(x, xa, softmax_output, last_classifier, global_classifier)
            kd_loss_feature = (kd_loss_feature_last + kd_loss_feature_global)*self.k_kd_feature
            kd_loss_output = (kd_loss_output_last + kd_loss_output_global)*self.k_kd_output
            kd_loss = kd_loss_feature + kd_loss_output
        else:
            kd_loss_feature, kd_loss_output, kd_loss  = 0,0,0

        c_loss = c_loss_cls + kd_loss

        correct = (torch.sum(torch.argmax(softmax_output, dim=1) == y)).item()

        self.classifier_optimizer.zero_grad()
        c_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1, norm_type='inf')
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
        self.classifier_optimizer.step()

        prob_mean = myitem(prob_mean)
        c_loss_flow = myitem(c_loss_flow)
        kd_loss = myitem(kd_loss)
        kd_loss_flow = myitem(kd_loss_flow)
        kd_loss_feature = myitem(kd_loss_feature)
        kd_loss_output = myitem(kd_loss_output)

        return {'c_loss': c_loss.item(), 'kd_loss': kd_loss, 'correct': correct, 'flow_prob_mean': flow_xa_prob_mean,
                 'c_loss_flow': c_loss_flow, 'kd_loss_flow': kd_loss_flow, 'kd_loss_feature': kd_loss_feature, 'kd_loss_output': kd_loss_output}

    def knowledge_distillation_on_output(self, xa, softmax_output, last_classifier, global_classifier):
        if self.k_kd_last_cls>0 and type(last_classifier)!=type(None):
            softmax_output_last, _ = last_classifier.forward_from_xa(xa)
            softmax_output_last = softmax_output_last.detach()
            kd_loss_output_last = self.k_kd_last_cls*MultiClassCrossEntropy(softmax_output, softmax_output_last, T=2)
        else:
            kd_loss_output_last = 0

        if self.k_kd_global_cls>0:
            softmax_output_global, _ = global_classifier.forward_from_xa(xa)
            softmax_output_global = softmax_output_global.detach()
            kd_loss_output_global = self.k_kd_global_cls*MultiClassCrossEntropy(softmax_output, softmax_output_global, T=2)
        else:
            kd_loss_output_global = 0

        return kd_loss_output_last, kd_loss_output_global
    
    def knowledge_distillation_on_xa_output(self, x, xa, softmax_output, last_classifier, global_classifier):
        if self.k_kd_last_cls>0 and type(last_classifier)!=type(None):
            softmax_output_last, xa_last, _ = last_classifier(x)
            xa_last = xa_last.detach()
            softmax_output_last = softmax_output_last.detach()
            kd_loss_feature_last = self.k_kd_last_cls*torch.pow(xa_last-xa, 2).mean()
            kd_loss_output_last = self.k_kd_last_cls*MultiClassCrossEntropy(softmax_output, softmax_output_last, T=2)
        else:
            kd_loss_feature_last = 0
            kd_loss_output_last = 0
        
        if self.k_kd_global_cls>0:
            softmax_output_global, xa_global, _ = global_classifier(x)
            xa_global = xa_global.detach()
            softmax_output_global = softmax_output_global.detach()
            kd_loss_feature_global = self.k_kd_global_cls*torch.pow(xa_global-xa, 2).mean()
            kd_loss_output_global = self.k_kd_global_cls*MultiClassCrossEntropy(softmax_output, softmax_output_global, T=2)
        else:
            kd_loss_feature_global = 0
            kd_loss_output_global = 0
        
        return kd_loss_feature_last, kd_loss_output_last, kd_loss_feature_global, kd_loss_output_global

    def train_a_batch_flow(self, x, y, last_flow, classes_so_far, available_labels_past):            
        xa = self.classifier.forward_to_xa(x)
        xa = xa.reshape(xa.shape[0], -1)
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
        loss_data = -self.flow.log_prob(inputs=xa, context=y_one_hot).mean()

        if self.algorithm=='PreciseFCL' and type(last_flow)!=type(None):
            batch_size = x.shape[0]
            with torch.no_grad():
                flow_xa, label, label_one_hot = self.sample_from_flow(last_flow, available_labels_past, batch_size)
            loss_last_flow = -self.flow.log_prob(inputs=flow_xa, context=label_one_hot).mean()
        else:
            loss_last_flow = 0
        loss_last_flow = self.k_flow_lastflow*loss_last_flow

        loss = loss_data + loss_last_flow

        self.flow_optimizer.zero_grad()
        loss.backward()
        self.flow_optimizer.step()

        return {'flow_loss': loss_data.item(), 'flow_loss_last': myitem(loss_last_flow)}
