#!/usr/bin/env python
import argparse
from FCL.initial import FCLInit
from utils.model_utils import create_model
import torch
from multiprocessing import Pool
from utils.util import *
import random
from collections import OrderedDict
import yaml
import pickle

def create_server_n_user(args, i):
    
    # create base model, irreverent to FedXXX
    model = create_model(args)

    if ('FCL' in args.algorithm):
        server = FCLInit(args, model, i)
    else:
        print("Algorithm {} has not been implemented.".format(args.algorithm))
        exit()
    return server


def run_job(args, i, seed):
    

    server = create_server_n_user(args, i)
    if args.train:
        acc, forget_rate = server.train(args)
        #server.test()

    return acc, forget_rate


def main(args):

    acc, forget_rate = run_job(args, 0, args.seed)
    print('-------------------',acc)
    print('-------------------',forget_rate)

    print("Finished training.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--train", type=int, default=1, choices=[0,1])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_glob_iters", type=int, default=60)
    parser.add_argument("--local_epochs", type=int, default=100)
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu","cuda"], help="run device (cpu | cuda)")
    parser.add_argument("--result_path", type=str, default="results", help="directory path to save results")
    parser.add_argument("--algorithm", type=str, default="FCL")
    parser.add_argument("--fedlwf", type=int, default=0, help="fedlwf")
    parser.add_argument("--fedavg", type=int, default=0, help="fedavg")
    parser.add_argument("--fedprox", type=int, default=0, help="fedprox")
    parser.add_argument("--l2c", type=int, default=0, help="fedprox")
    parser.add_argument("--local", type=int, default=0, help="local")
    parser.add_argument("--scaffold", type=int, default=0, help="scaffold")
    parser.add_argument("--ClusterFL", type=int, default=0, help="ClusterFL")
    parser.add_argument("--peravg", type=int, default=0, help="PerAvg")
    parser.add_argument("--pfedme", type=int, default=0, help="PFedMe")
    parser.add_argument("--AFCL", type=int, default=1, help="AFCL")
    parser.add_argument("--FCL", type=int, default=0, help="FCL")
    parser.add_argument("--offline", type=int, default=0, help="offline")
    parser.add_argument("--naive", type=int, default=0, help="naive")
    parser.add_argument("--mu", type=float, default=0.005, help='proximal term consstant')
    parser.add_argument("--minloss", type=float, default=5, help='minloss of fedprox ')
    parser.add_argument("--dataset", type=str, default="EMNIST-Letters", choices=['EMNIST-Letters', 'EMNIST-Letters-malicious',
                                                                        'EMNIST-Letters-shuffle', 'CIFAR100', 'MNIST-SVHN-FASHION', 'TEST-noniid'])
    parser.add_argument("--datadir", type=str, default=r'E:\czk\1')
    parser.add_argument("--data_split_file", type=str, default="split_files/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl")
    parser.add_argument("--ntask", type=int, default=6)
    parser.add_argument("--num_users", type=int, default=8, help="Number of Users per round")
    parser.add_argument('--result', type=str, default='/home/czk/Danny/2/results/emnist')

    # scaffold
    parser.add_argument("--weight_decay", type=float, default=0.001, help='weight decay of scaffold')
    parser.add_argument('--glo_lr', type=float, default=1.0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Local learning rate")
    # peravg and pFedMe
    parser.add_argument("--beta", type=float, default=0.1, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--personal_learning_rate", type=float, default=0.09,
                        help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--lamda", type=int, default=5, help="Regularization term")
    parser.add_argument("--K", type=int, default=3, help="Computation steps")
    # afcl
    parser.add_argument('--total_nc', default=100, type=int, help='class number for the dataset')
    parser.add_argument('--temp', default=0.1, type=float, help='training time temperature')
    parser.add_argument('--lambda_proto_aug', default=0, type=float, help='protoAug loss weight')
    parser.add_argument('--lambda_repr_loss', default=0, type=float, help='representation loss weight')
    parser.add_argument('--repr_loss_temp', default=1., type=float, help='representation loss temp')

    parser.add_argument('--proto_queue_length', default=100, type=int, help='length of the proto queue')
    parser.add_argument('--proto_queue',  action='store_true', default=False, help="use proto_queue for proto_aug loss")
    parser.add_argument('--mean_proto_queue',  action='store_true', default=False,
                        help="compute global prototypes as the weighted mean of the proto_queue")
    parser.add_argument('--multi_radius',  action='store_true', default=False,
                        help="keep multiple radiuses, one for each entry of the queue")
    parser.add_argument('--mask_model', action='store_true', default=False,
                        help="mask the model output")
    parser.add_argument('--proto_loss_curr_classes', default=False,
                        help="use all classes on proto loss, even current ones")
    parser.add_argument('--location_proto_aug', default='local', type=str,
                        help="local or global, which prototypes to use for the proto aug loss")
    parser.add_argument('--ema_global', default=0.9, type=float, help='exponential moving average smoothing factor')



    # model
    parser.add_argument('--c-channel-size', type=int, default=64)
    parser.add_argument("--model", type=str, default="cnn")
    # optimizer
    parser.add_argument('--lr', type=float, default=1e-03)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    
    # tools
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--visual', type=int, default=0 )
    parser.add_argument('--draw', type=int, default=0 )
    parser.add_argument('--sw', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1.0)


    args = parser.parse_args()

    main(args)
    