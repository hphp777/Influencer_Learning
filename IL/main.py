'''
Main file to set up the FL system and train
Code design inspired by https://github.com/FedML-AI/FedML
'''
import torch
import torch.nn as nn
import numpy as np
import random
import data_preprocessing.data_loader as dl
import argparse
from models.resnet import resnet56, resnet18
from models.resnet_fedalign import resnet56 as resnet56_fedalign
from models.resnet_fedalign import resnet18 as resnet18_fedalign
from models.unet_model import *
import torchvision.models as models 
import importlib
importlib.reload(models)
from torch.multiprocessing import set_start_method, Queue
import logging
import os
from collections import defaultdict
from data_preprocessing.datasets import NIHQualificationDataset, NIHTestDataset
from data_preprocessing.datasets import BraTS2021QualificationLoader, BraTS2021TestLoader
import time

# methods
import node
import data_preprocessing.custom_multiprocess as cm
from data_preprocessing.data_loader import _data_transforms_NIH, load_dynamic_db, dynamic_partition_data

def add_args(parser):
    # Training settings
    parser.add_argument('--task', type=str, default="segmentation",
                        help='classification, segmentation')
    
    parser.add_argument('--data_dir', type=str, default="D:/Data/BraTS2021/2D",
                        help='data directory: data/cifar100, data/cifar10, "C:/Users/hb/Desktop/data/NIH", \
                        C:/Users/hb/Desktop/data/CheXpert-v1.0-small, "D:/Data/BraTS2021/2D"')

    parser.add_argument('--dataset', type=str, default="BraTS2021",
                        help='data directory: cifar100, cifar10, NIH, CheXpert, BraTS2021')
    
    parser.add_argument('--dynamic_db', type=bool, default=True,
                        help='whether use of dynamic database')

    parser.add_argument('--partition_method', type=str, default='homo', metavar='N',
                        help='how to partition the dataset on local clients')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='alpha value for Dirichlet distribution partitioning of data(default: 0.5)')

    parser.add_argument('--client_number', type=int, default=5, metavar='NN',
                        help='number of clients in the FL system')

    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--alpha', type=float, default= 0.99, metavar='a',
                        help='distillation weight : 10.0, 5.0, 2.0, 0.99, 0.95, 0.5, 0.1, 0.05')
    
    parser.add_argument('--temperature', type=float, default=1.5, metavar='T',
                        help='20.0, 10.0, 8.0, 6.0, 4.5, 3.0, 2.0, 1.5')
    
    parser.add_argument('--num_of_influencer', type=int, default=1, metavar='T',
                        help='number of influencer')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.0001)

    parser.add_argument('--epochs', type=int, default=10, metavar='EP',
                        help='how many epochs will be trained locally per round')
    
    parser.add_argument('--influencing_epochs', type=int, default=2, metavar='EP',
                        help='how many epochs will be trained in the distillation(influencing) step')

    parser.add_argument('--influencing_round', type=int, default=30,
                        help='how many rounds of communications are conducted')

    parser.add_argument('--pretrained', action='store_true', default=False,  
                        help='test pretrained model')

    parser.add_argument('--mu', type=float, default=1.0, metavar='MU',
                        help='mu value for various methods')

    parser.add_argument('--save_client', action='store_true', default=False,
                        help='Save client checkpoints each round')

    parser.add_argument('--thread_number', type=int, default=1, metavar='NN',
                        help='number of parallel training threads')

    parser.add_argument('--client_sample', type=float, default=1.0, metavar='MT',
                        help='Fraction of clients to sample')
    args = parser.parse_args()

    return args

# Setup Functions
def set_random_seed(seed=1996):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ## NOTE: If you want every run to be exactly the same each time
    ##       uncomment the following lines
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# Helper Functions
def init_process(q, Client):
    # q is the client info
    set_random_seed()
    global client # 새롭게 클라이언트를 전역으로 선언
    # c0 is a client_dict
    # c1 is the namespace
    ci = q.get() # Queued에서 맨 앞의 원소를 하나 가져오고 remove
    client = Client(ci[0], ci[1]) 

def run_clients(r):
    try:
        return client.run(r) # give threads' number of model weight
    except KeyboardInterrupt:
        logging.info('exiting')
        return None

def allocate_clients_to_threads(args):
    mapping_dict = defaultdict(list) # default 값이 list인 dictionary
    for round in range(args.influencing_round): # 각 communication round마다 thread를 할당 해 줌
        if args.client_sample<1.0: # 여러 클라이언트들 중에서 sampling하여 트레이닝을 진행하는 경우
            num_clients = int(args.client_number*args.client_sample)
            client_list = random.sample(range(args.client_number), num_clients)
        else: # 주어진 클라이언트를 모두 사용하는 경우
            num_clients = args.client_number
            client_list = list(range(num_clients))
        if num_clients % args.thread_number==0 and num_clients>0:
            clients_per_thread = int(num_clients/args.thread_number) # 클라이언트 1명당 스레드 보통 1개 할당
            for c, t in enumerate(range(0, num_clients, clients_per_thread)):
                idxs = [client_list[x] for x in range(t, t+clients_per_thread)] # 앞에서 부터 차례대로 할당
                mapping_dict[c].append(idxs)
        else:
            raise ValueError("Sampled client number not divisible by number of threads")
    return mapping_dict

if __name__ == "__main__":
    try:
     set_start_method('spawn')
    except RuntimeError:
        pass
    set_random_seed()
    # get arguments
    parser = argparse.ArgumentParser()
    args = add_args(parser)
 
    # time.sleep(150*(args.client_number/16)) #  Allow time for threads to start up
    
    ###################################### get data
    
    train_indices = dynamic_partition_data(args.data_dir, args.partition_method, n_nets= args.client_number, alpha= args.partition_alpha, n_round = args.influencing_round, dynamic=args.dynamic_db)
    train_data_local_dict = load_dynamic_db(args.data_dir, args.partition_method, args.partition_alpha, args.client_number, args.batch_size, args.influencing_round, train_indices)
    
    if args.dataset == 'NIH':
        test_data = torch.utils.data.DataLoader(NIHTestDataset(args.data_dir, transform = _data_transforms_NIH()), batch_size = 32, shuffle = not True)
        qualification_data = torch.utils.data.DataLoader(NIHQualificationDataset(args.data_dir, transform = _data_transforms_NIH()), batch_size = 32, shuffle = not True)
        class_num = 14
    elif args.dataset == 'BraTS2021':
        test_data = torch.utils.data.DataLoader(BraTS2021TestLoader(args.data_dir), batch_size = 32, shuffle = not True)
        qualification_data = torch.utils.data.DataLoader(BraTS2021QualificationLoader(args.data_dir), batch_size = 32, shuffle = not True)
        class_num = 5

    ######################################################
    

    mapping_dict = allocate_clients_to_threads(args) 
    print("Client allocation for the threads during influencing round : ", mapping_dict)

    Client = node.Node

    # model
    if args.task == 'classification':
        # Model = resnet56
        Model = models.efficientnet_b0(pretrained=True)
        num_ftrs = Model.classifier[1].in_features
        Model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=class_num)
    elif args.task == 'segmentation':
        Model = UNet(1, class_num, bilinear=False) 

    client_dict = [{'train_data':train_data_local_dict, 'qulification_data': qualification_data, 'test_data' : test_data,'device': i % torch.cuda.device_count(),
                        'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num, 'dir': args.data_dir} for i in range(args.thread_number)]

    client_info = Queue()
    for i in range(args.thread_number):
        client_info.put((client_dict[i], args))

    pool = cm.MyPool(args.thread_number, init_process, (client_info, Client)) 

    for r in range(args.influencing_round):
        
        logging.info('Influencing Round: {} ********************************************************************************'.format(r+1))
        round_start = time.time()
        
        pool.map(run_clients, [r]) 
        # pool(run_clients) 

        round_end = time.time()
        total_sec = round_end-round_start
        total_min = (total_sec) // 60
        logging.info('Round {} Time: {:.0f}m {:.0f}s'.format(r, total_min, total_sec % 60))

    pool.close()
    pool.join()
