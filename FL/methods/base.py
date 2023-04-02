import torch
import logging
import json
from torch.multiprocessing import current_process
import numpy as np
import os
from sklearn.metrics import roc_auc_score,  roc_curve


class Base_Client():
    def __init__(self, client_dict, args):
        self.train_data = client_dict['train_data'] # dataloader(with all clients)
        self.test_data = client_dict['test_data'] # dataloader(with all clients)
        self.device = 'cuda:{}'.format(client_dict['device'])
        self.model_type = client_dict['model_type'] # model type is the model itself
        self.num_classes = client_dict['num_classes']
        self.dir = client_dict['dir']
        self.args = args
        self.round = 0
        self.client_map = client_dict['client_map']
        self.train_dataloader = None
        self.test_dataloader = None
        self.client_index = None
    
    def load_client_state_dict(self, server_state_dict):
        # If you want to customize how to state dict is loaded you can do so here
        self.model.load_state_dict(server_state_dict)
    
    def run(self, received_info): # thread의 갯수 만큼 실행됨
        # recieved info : a server model weights(OrderedDict)
        # one globally merged model's parameter
        client_results = []
        for client_idx in self.client_map[self.round]: # round is the index of communication round
            # 한 tread에 할당된 client의 index가 매 round마다 들어있음.
            self.load_client_state_dict(received_info) 
            self.train_dataloader = self.train_data[client_idx] # among dataloader, pick one
            self.test_dataloader = self.test_data[client_idx]
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None and self.train_dataloader._iterator._shutdown:
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()
            self.client_index = client_idx
            num_samples = len(self.train_dataloader)*self.args.batch_size
            weights = self.train()
            acc = self.test()
            client_results.append({'weights':weights, 'num_samples':num_samples,'acc':acc, 'client_index':self.client_index})
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()

        self.round += 1
        return client_results # clients' number of weights 
        # 하나의 thread에 할당된 client의 갯수 만큼의 client_result가 반환됨
        
    def train(self):
        # train the local model
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                if 'NIH' in self.dir or 'CheXpert' in self.dir:
                    out = self.model(images)  
                    loss = self.criterion(out, labels.type(torch.FloatTensor).to(self.device))
                else:
                    log_probs = self.model(images)
                    loss = self.criterion(log_probs, labels.type(torch.LongTensor).to(self.device))
                
                
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                            epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        weights = self.model.cpu().state_dict()
        return weights

    def test(self):
        self.model.to(self.device)
        self.model.eval()
        sigmoid = torch.nn.Sigmoid()
        test_correct = 0.0
        test_sample_number = 0.0
        val_loader_examples_num = len(self.test_dataloader.dataset)
        probs = np.zeros((val_loader_examples_num, self.num_classes), dtype = np.float32)
        gt    = np.zeros((val_loader_examples_num, self.num_classes), dtype = np.float32)
        k=0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.test_dataloader):
                target = target.type(torch.LongTensor)
                x = x.to(self.device)
                target = target.to(self.device)
                out = self.model(x)
                
                if 'NIH' in self.dir or 'CheXpert' in self.dir:
                    probs[k: k + out.shape[0], :] = out.cpu()
                    gt[   k: k + out.shape[0], :] = target.cpu()
                    k += out.shape[0] 
                    preds = np.round(sigmoid(out).cpu().detach().numpy())
                    targets = target.cpu().detach().numpy()
                    test_sample_number += len(targets)*self.num_classes
                    test_correct += (preds == targets).sum()
                else:
                    _, predicted = torch.max(out, 1)
                    correct = predicted.eq(target).sum()
                    test_correct += correct.item()
                    # test_loss += loss.item() * target.size(0)
                    test_sample_number += target.size(0)
            
            acc = (test_correct / test_sample_number)*100
            if self.args.dataset == 'NIH' or self.args.dataset == 'CheXpert':
                try:
                    auc = roc_auc_score(gt, probs)
                except:
                    auc = 0
                logging.info("************* Client {} AUC = {:.2f},  Acc = {:.2f}**************".format(self.client_index, auc, acc))
                return auc
            else:
                logging.info("************* Client {} Acc = {:.2f} **************".format(self.client_index, acc))
                return acc
    
class Base_Server():
    def __init__(self,server_dict, args):
        self.train_data = server_dict['train_data']
        self.test_data = server_dict['test_data']
        self.device = 'cuda:{}'.format(torch.cuda.device_count()-1)
        self.model_type = server_dict['model_type']
        self.num_classes = server_dict['num_classes']
        self.dir = server_dict['dir']
        self.acc = 0.0
        self.round = 0
        self.args = args
        self.save_path = server_dict['save_path']

    def run(self, received_info):
        server_outputs = self.operations(received_info)
        acc = self.test()
        self.log_info(received_info, acc)
        self.round += 1
        if acc > self.acc:
            torch.save(self.model.state_dict(), '{}/{}.pt'.format(self.save_path, 'server'))
            self.acc = acc
        acc_path = '{}/logs/{}_{}_harmony_acc.txt'.format(os.getcwd(), self.args.dataset,self.args.method)
        f = open(acc_path, 'a')
        f.write(str(acc) + '\n')
        f.close()
        return server_outputs
    
    def start(self):
        with open('{}/config.txt'.format(self.save_path), 'a+') as config:
            config.write(json.dumps(vars(self.args)))
        return [self.model.cpu().state_dict() for x in range(self.args.thread_number)]

    def log_info(self, client_info, acc):
        client_acc = sum([c['acc'] for c in client_info])/len(client_info)
        out_str = 'Test/AccTop1: {}, Client_Train/AccTop1: {}, round: {}\n'.format(acc, client_acc, self.round)
        with open('{}/out.log'.format(self.save_path), 'a+') as out_file:
            out_file.write(out_str)

    def operations(self, client_info):
        client_info.sort(key=lambda tup: tup['client_index']) # 뒤죽박죽된 client_info를 client의 index 순으로 정렬 (1 ~)
        client_sd = [c['weights'] for c in client_info] # clients' number of weights
        ################################################################################################
        cw = [c['num_samples']/sum([x['num_samples'] for x in client_info]) for c in client_info]

        ssd = self.model.state_dict()
        for key in ssd:
            ssd[key] = sum([sd[key]*cw[i] for i, sd in enumerate(client_sd)])
        self.model.load_state_dict(ssd)
        if self.args.save_client:
            for client in client_info:
                torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        return [self.model.cpu().state_dict() for x in range(self.args.thread_number)] # thread의 갯수만큼 server의 모델을 복사해서 반환

    def test(self):
        self.model.to(self.device)
        self.model.eval()
        sigmoid = torch.nn.Sigmoid()
        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        val_loader_examples_num = len(self.test_data.dataset)
        probs = np.zeros((val_loader_examples_num, self.num_classes), dtype = np.float32)
        gt    = np.zeros((val_loader_examples_num, self.num_classes), dtype = np.float32)
        k=0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.test_data):
                target = target.type(torch.LongTensor)
                x = x.to(self.device)
                target = target.to(self.device)
                out = self.model(x)
                # loss = self.criterion(pred, target)
                if 'NIH' in self.dir or 'CheXpert' in self.dir:
                    probs[k: k + out.shape[0], :] = out.cpu()
                    gt[   k: k + out.shape[0], :] = target.cpu()
                    k += out.shape[0] 
                    preds = np.round(sigmoid(out).cpu().detach().numpy())
                    targets = target.cpu().detach().numpy()
                    test_sample_number += len(targets)*self.num_classes
                    test_correct += (preds == targets).sum()
                else:
                    _, predicted = torch.max(out, 1)
                    correct = predicted.eq(target).sum()
                    test_correct += correct.item()
                    test_sample_number += target.size(0)

                
                # test_loss += loss.item() * target.size(0)
                
            acc = (test_correct / test_sample_number)*100
            if self.args.dataset == 'NIH' or self.args.dataset == 'CheXpert':
                auc = roc_auc_score(gt, probs)
                logging.info("***** Server AUC = {:.4f} ,Acc = {:.4f} *********************************************************************".format(auc, acc))
                return auc * 100
            else:
                logging.info("***** Server Acc = {:.4f} *********************************************************************".format(acc))
                return acc