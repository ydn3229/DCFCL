from functools import reduce
import torch
from torch import nn, autograd
from torch.autograd import Variable
import os
import os.path
from FCL.model_sort import *
import numpy as np
from torch.nn import functional as F
import torchvision

EPSILON = 1e-16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Data2DataCrossEntropyLoss(torch.nn.Module):
    def __init__(self, num_classes, temperature, m_p, master_rank):
        super(Data2DataCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.m_p = m_p
        self.master_rank = master_rank
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def make_index_matrix(self, labels):
        labels = labels.detach().cpu().numpy()
        num_samples = labels.shape[0]
        mask_multi, target = np.ones([self.num_classes, num_samples]), 0.0

        for c in range(self.num_classes):
            c_indices = np.where(labels == c)
            mask_multi[c, c_indices] = target
        return torch.tensor(mask_multi).type(torch.long).to(self.master_rank)

    def remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.master_rank)
        return M[mask].view(h, -1)

    def forward(self, embed, proxy, label):
        # calculate similarities between sample embeddings
        sim_matrix = self.calculate_similarity_matrix(embed, embed) + self.m_p - 1
        # remove diagonal terms
        sim_matrix = self.remove_diag(sim_matrix / self.temperature)
        # for numerical stability
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = F.relu(sim_matrix) - sim_max.detach()

        # calculate similarities between sample embeddings and the corresponding proxies
        smp2proxy = self.cosine_similarity(embed, proxy)
        # make false negative removal
        removal_fn = self.remove_diag(self.make_index_matrix(label)[label])
        # apply the negative removal to the similarity matrix
        improved_sim_matrix = removal_fn * torch.exp(sim_matrix)

        # compute positive attraction term
        pos_attr = F.relu((self.m_p - smp2proxy) / self.temperature)
        # compute negative repulsion term
        neg_repul = torch.log(torch.exp(-pos_attr) + improved_sim_matrix.sum(dim=1))
        # compute data to data cross-entropy criterion
        criterion = pos_attr + neg_repul
        return criterion.mean()


class ENV(nn.Module):
    def __init__(self, z_size,
                 image_size,
                 image_channel_size,
                 dataset,
                 ):

        super().__init__()

        self.z_size = z_size
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.lossfuct = nn.CrossEntropyLoss()
        # Build backbone
        self.lamda = None

    def train_a_batch(self, x, y,
                      classes_so_far,
                      importance_of_new_task=.5
                      ):
        # =============== update D ==============
        # run the critic on the real data.
        c_loss_real = self.ce_loss(x, y, classes_so_far, return_g=True, return_aux=True)
        c_loss = c_loss_real
        # updation
        self.model.zero_grad()
        c_loss.backward()
        self.model.step()

        return {'c_loss': c_loss.item()}


    def set_lambda(self, l):
        self.lamda = l

    def ce_loss(self, x, y, classes_so_far, return_g=False, return_aux=False, return_feature=False):

        # info
        batch_size = x.size(0)

        # generate label:
        dis_label = torch.FloatTensor(batch_size)
        dis_label = dis_label.to(torch.device('cuda:0'))
        y = y.to(torch.device('cuda:0'))
        dis_label = Variable(dis_label)
        dis_label.data.fill_(1)
        preds = self.model(x, if_features=True)
        # data transform
        y_hot = torch.nn.functional.one_hot(y.to(torch.int64), self.num_classes).float()
        loss = self.lossfuct(preds,y_hot)

        return loss


    def visualize(self, sample_size=16, path='./images'):

        os.makedirs(os.path.dirname(path), exist_ok=True)
        data, label = self.sample(sample_size, 1)

        torchvision.utils.save_image(
            data,
            path + '.jpg',
            nrow=6,
        )

        print('image is saved!')


    def train_a_batch_all(self, x, y,
                          available_labels,
                          classes_so_far,
                          glob_iter_,
                          importance_of_new_task=.5
                          ):

        c_loss = self.ce_loss(x, y, classes_so_far, return_g=True, return_aux=True)


        # c_loss = c_loss

        # updation
        self.model.zero_grad()
        c_loss.backward()
        self.model.step()

        return {'c_loss': c_loss.item()}