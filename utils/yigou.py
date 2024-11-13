import random

import numpy as np

def dirictlet_zipf_disclient(train_y_list, n_clients):
    # 100个客户端，10-20-20-20-30分5组，每组均按照dirictlet分布
    train_labels = np.array(train_y_list)
    n_classes = train_labels.max()+1
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]
    # 每一类在lable中的位置
    tongji = {k: [] for k in range(n_classes)}
    real_sum = 0
    for i, idx in enumerate(class_idcs):
        tongji[i] = len(idx)
        real_sum += len(idx)
    sum_num = list(tongji.values())
    sum_num = np.array(sum_num)
    real_dis = sum_num / real_sum

    s = 0.0
    rank = np.arange(1, n_clients + 1)
    probabilities = 1 / (rank ** s)
    probabilities /= np.sum(probabilities)  # 归一化概率

    client_num = (probabilities * len(train_labels)).reshape(-1,1)

    alpha = np.full(n_classes, 0.1)  # 设置alpha为0.1

    # 生成Dirichlet分布样本
    samples = np.random.dirichlet(alpha, size=n_clients)

    fenbu = samples * client_num
    # a--原始分布，有超出的数
    a = np.sum(fenbu, axis=0)
    where = np.where(a >= sum_num)[0]
    # b--新的分布，最大的数是上限
    b = np.copy(a)
    for w in where:
        b[w] = sum_num[w]

    fenbu_new = fenbu / a[np.newaxis, :]
    fenbu_new = fenbu_new * b[np.newaxis, :]
    fenbu_new = np.floor(fenbu_new)
    c = np.sum(fenbu_new, axis=0)
    client_idcs = [[] for i in range(n_clients)]
    for k_idx, fenbu_idx, mask_ind in zip(class_idcs, fenbu_new.T, c):
        for i, idy in enumerate(np.split(k_idx[:int(mask_ind)],
                                      (np.cumsum(fenbu_idx)[:-1]).
                                              astype(int))):
            client_idcs[i] += [idy]

    # client对应数据的位置索引
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    inds = [list(cli) for cli in client_idcs]
    for ii in inds:
        random.shuffle(ii)

    train_inds = [i[:-2000] for i in inds]
    test_inds = [i[-2000:] for i in inds]


    return train_inds, test_inds, client_idcs
