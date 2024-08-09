# Assuming 'cdata' is already defined
# Example structure of cdata
from divide_dataset import *
import torch
import random

def creat_dataset(dataset, clientnum, task, per_task_class, per_task_num):
    users = {'user_id':[], 'user_data':{}, 'user_class':[[] for _ in range(clientnum)]}

    for i in range(clientnum):
        name = 'f_0000' + str(i)
        users['user_id'].append(name)


    data_dict = {}
    train_data = dataset[0]
    train_labels = dataset[1]
    # 遍历数据集
    for idx, (data, label) in enumerate(zip(train_data, train_labels)):
        if label.item() not in data_dict:
            data_dict[label.item()] = {'data': [], 'indices': []}
        data_dict[label.item()]['data'].append(data)
        data_dict[label.item()]['indices'].append(idx)

    user_data = {}
    for u in users['user_id']:
        user_data[u] = {}
        user_data[u]['x'] = []
        user_data[u]['y'] = []
    class_num = len(data_dict)
    class_id = np.arange(0, class_num)
    class_used = set(class_id)
    per_class_num = per_task_num / per_task_class
    for t in range(task):
        for u in users['user_id']:
            class_used = set(class_id)
            id = int(u[-1])
            used_class = set(users['user_class'][id])
            if used_class is not None:
                class_used = class_used - used_class
            class_new = random.sample(list(class_used), per_task_class)
            data_x = torch.empty((0,))
            data_y = torch.empty((0,))
            for c in class_new:
                data_idx = random.sample(range(len(data_dict[c]['data'])), int(per_class_num))
                data_x_ = [data_dict[c]['data'][i] for i in data_idx]
                data_x_ = torch.cat(data_x_)
                data_x = torch.cat((data_x, data_x_))
                data_y_ = torch.full((int(per_class_num),), int(c))
                data_y = torch.cat((data_y, data_y_))
                users['user_class'][id].append(c)
            user_data[u]['x'].append(data_x)
            user_data[u]['y'].append(data_y)

    return users['user_id'], user_data





if __name__=="__main__":

    dataset = GetDataSet()
    dataset_name = "MNIST"
    train_images, train_labels, test_images, test_labels = dataset.mnistDataSetConstruct()
    task = 5
    per_task_class = 2
    per_task_num = 2000
    id, data = creat_dataset((train_images, train_labels), 10, task, per_task_class, per_task_num)

    # ================= create dataset =========================
    cdata = {'users':id, 'user data':data}
    data_path = os.path.join("../data", dataset_name)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    torch.save(cdata,
               os.path.join(data_path, "train" + ".pt"))

