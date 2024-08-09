#!/usr/bin/env python
import argparse
from FLAlgorithms.servers.serverpFedCL import FedCL
from FCL.initial import FCLInit
from utils.model_utils import create_model
from utils.plot_utils import *
import torch
from multiprocessing import Pool
from utils.util import *
import random
from collections import OrderedDict
import yaml
import pickle

def create_server_n_user(args, i):

    model = create_model(args)

    if ('FCL' in args.algorithm):
        server = FCLInit(args, model, i)
    else:
        print("Algorithm {} has not been implemented.".format(args.algorithm))
        exit()
    return server


def run_job(args, i, seed):

    torch.manual_seed(seed)
    random.seed(seed)
    print('random seed is: ', seed)
    print("\n\n         [ Start training iteration {} ]           \n\n".format(i))
    # Generate model
    server = create_server_n_user(args, i)
    if args.train:
        acc = server.train(args)
        #server.test()

    return acc
cudnn_deterministic = True
def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic

def main(args):
    # seed = [0]
    
    # for i in range(args.times):
    seed_everything(args.seed)
    
    acc = run_job(args, 0, args.seed)
    print('-------------------',acc)
        # for name , metrcic in zip(['TAw Acc', 'TAg Acc', 'TAw Forg', 'TAg Forg'],metrics):

    result = OrderedDict([('TAwa',acc)])

    pickle_log['train'][4] = dict(result)

    with open(os.path.join(output_dir, "log.pkl"), "wb") as f:
        pickle.dump(pickle_log, f)

    print('Done!')
    os.makedirs(os.path.join(output_dir, 'done'), exist_ok=True)


    print("Finished training.")
    


if __name__ == "__main__":
    _logger = logging.getLogger('train')

    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--train", type=int, default=1, choices=[0,1])
    parser.add_argument("--algorithm", type=str, default="FCL")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Local learning rate")
    parser.add_argument("--lamda", type=int, default=1, help="Regularization term")
    parser.add_argument("--mix_lambda", type=float, default=0.1, help="Mix lambda for FedMXI baseline")
    parser.add_argument("--num_glob_iters", type=int, default=6)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--K", type=int, default=1, help="Computation steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu","cuda"], help="run device (cpu | cuda)")
    parser.add_argument("--result_path", type=str, default="results", help="directory path to save results")
    parser.add_argument("--fedlwf", type=int, default=0, help="fedlwf")
    parser.add_argument("--fedavg", type=int, default=0, help="fedavg")
    parser.add_argument("--fedprox", type=int, default=0, help="fedprox")
    parser.add_argument("--l2c", type=int, default=1, help="fedprox")
    parser.add_argument("--local", type=int, default=0, help="local")        
    parser.add_argument("--FCL", type=int, default=0, help="FCL")
    parser.add_argument("--offline", type=int, default=0, help="offline")
    parser.add_argument("--naive", type=int, default=0, help="naive")
    parser.add_argument("--mu", type=float, default=0.005, help='proximal term consstant')
    parser.add_argument("--minloss", type=float, default=5, help='minloss of fedprox ')
    parser.add_argument("--dataset", type=str, default="EMNIST-Letters-shuffle", choices=['EMNIST-Letters', 'EMNIST-Letters-malicious',
                                                                        'EMNIST-Letters-shuffle', 'CIFAR100', 'MNIST-SVHN-FASHION'])
    parser.add_argument("--datadir", type=str, default='/home/1')
    parser.add_argument("--data_split_file", type=str, default="split_files/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl")
    parser.add_argument("--ntask", type=int, default=6)
    parser.add_argument("--num_users", type=int, default=8, help="Number of Users per round")
    parser.add_argument('--result', type=str, default='/home/result')
    # model
    parser.add_argument('--c-channel-size', type=int, default=64)
    parser.add_argument("--model", type=str, default="cnn")
    # optimizer
    parser.add_argument('--lr', type=float, default=1e-04)  
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=0)
    
    # tools
    parser.add_argument('--seed', type=int, default=62)
    parser.add_argument('--visual', type=int, default=0 )
    parser.add_argument('--draw', type=int, default=0 )
    parser.add_argument('--sw', type=float, default=1e-01)
    parser.add_argument('--alpha', type=float, default=0.2)




    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Subset of users      : {}".format(args.num_users))
    print("Number of global rounds       : {}".format(args.num_glob_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("Device            : {}".format(args.device))
    print("SW            : {}".format(args.sw))
    print("=" * 80)
    main(args)
    