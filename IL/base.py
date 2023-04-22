import torch
import logging
import json
from torch.multiprocessing import current_process
import numpy as np
import os
from sklearn.metrics import roc_auc_score,  roc_curve
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from datetime import datetime
from utils.metrics import dice_coeff, multiclass_dice_coeff, SegmentationMetrics

class Participant():

    def __init__(self, client_dict, args):

        self.train_data = client_dict['train_data'] # dataloader(with all clients)
        self.qualification_data = client_dict['qulification_data']
        self.test_data = client_dict['test_data']
        self.device = 'cuda:{}'.format(client_dict['device'])
        self.model_type = client_dict['model_type'] # model type is the model itself
        self.num_classes = client_dict['num_classes']
        self.num_client = args.client_number
        self.dir = client_dict['dir']
        self.model_weights = [self.model_type.state_dict()] * args.client_number
        self.args = args
        self.round = 0
        self.client_map = client_dict['client_map']
        self.train_dataloader = None
        self.test_dataloader = None
        self.client_index = None
        self.distill_logits = [0] * self.num_client
        self.qulification_scores = [0] * self.num_client

        now = datetime.now()

        self.result_dir = "C:/Users/hb/Desktop/code/Influencer_learning/IL/Results/{}_{}H_{}M".format(now.date(), str(now.hour), str(now.minute))
        os.mkdir(self.result_dir)
        self.model_dir = "C:/Users/hb/Desktop/code/Influencer_learning/IL/Results/{}_{}H_{}M/models".format(now.date(), str(now.hour), str(now.minute))
        os.mkdir(self.model_dir)
        c = open(self.result_dir + "/config.txt", "w")
        c.write("Task: {}, learning method: IL, alpha: {}, temperature: {}, dynamic_db: {}, num_of_influencer: {}, inf_round: {}, local_epoch: {}".format(self.args.task,str(self.args.alpha), str(self.args.temperature), str(self.args.dynamic_db), str(self.args.num_of_influencer), str(self.args.influencing_round), str(self.args.epochs)))
        for i in range(self.num_client):
            open(self.result_dir + "/participants{}.txt".format(i+1), "w")
    
    def run(self,inf_round): # thread의 갯수 만큼 실행됨

        # Step1 : local training
        logging.info("Step 1. Local training **************************************************************")
        for client_idx in self.client_map[self.round]:

            # 한 tread에 할당된 client의 index가 매 round마다 들어있음.
            self.train_dataloader = self.train_data[client_idx] # among dataloader, pick one
            self.test_dataloader = self.test_data
            self.qualification_dataloader = self.qualification_data
            
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None and self.train_dataloader._iterator._shutdown:
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()
            
            self.client_index = client_idx
            self.train(client_idx, inf_round)
            self.qulification_scores[client_idx], self.distill_logits[client_idx] = self.qulification(client_idx)
            
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()

        # Step2 : Influencing
        logging.info("Step 2. Influencing *****************************************************************")
        self.max_idx = self.qulification_scores.index(max(self.qulification_scores))
        # self.qulification_scores[self.max_idx] = 0
        # self.second_max_idx = self.qulification_scores.index(max(self.qulification_scores))
        logging.info("Selected Influencer : paticipant {}".format(self.max_idx+1))
        # self.ensemble_influencing(self.max_idx, self.second_max_idx, self.args)
        self.influencing(self.max_idx,self.args)

        logging.info("Step 3. Evaluation ******************************************************************")
        for client_idx in range(self.num_client):
            self.test(client_idx)

        self.round += 1
        
    
    def train(self, client_idx, inf_round):

        self.model.load_state_dict(self.model_weights[client_idx])
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []

        logging.info("The number of data of participant {} : {}".format(client_idx+1, len(self.train_dataloader[inf_round]) * 32))
        
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader[inf_round]):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                # print(images.size())
                self.optimizer.zero_grad()

                if self.args.task == 'classification':
                    if 'NIH' in self.dir or 'CheXpert' in self.dir:
                        out = self.model(images)  
                        loss = self.criterion(out, labels.type(torch.FloatTensor).to(self.device))
                    else:
                        log_probs = self.model(images)
                        loss = self.criterion(log_probs, labels.type(torch.LongTensor).to(self.device))

                elif self.args.task == "segmentation":
                    masks_pred = self.model(images)
                    true_masks = labels.squeeze(1).type(torch.LongTensor)
                    loss = self.criterion(F.softmax(masks_pred.to(self.device), dim=1).float(), true_masks.to(self.device)) 
                
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('----participant {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index+1, epoch+1, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        
        self.model_weights[client_idx] = self.model.cpu().state_dict()


    def qulification(self, client_idx):

        self.model.load_state_dict(self.model_weights[client_idx])
        self.model.to(self.device)
        self.model.eval()
        sigmoid = torch.nn.Sigmoid()
        test_correct = 0.0
        test_sample_number = 0.0
        val_loader_examples_num = len(self.qualification_dataloader.dataset)

        if self.args.task == 'classification':
            probs = np.zeros((val_loader_examples_num, self.num_classes), dtype = np.float32)
            gt    = np.zeros((val_loader_examples_num, self.num_classes), dtype = np.float32)
            k=0
        elif self.args.task == 'segmentation':
            dice_score = 0
            probs = []

        with torch.no_grad():

            for batch_idx, (x, target) in enumerate(self.qualification_dataloader):

                target = target.type(torch.LongTensor)
                x = x.to(self.device)
                target = target.to(self.device)
                out = self.model(x)
                
                if self.args.task == 'classification':
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
                elif self.args.task == 'segmentation':
                    mask_pred = F.one_hot(out.argmax(dim=1), 5).permute(0, 3, 1, 2).float()
                    mask_true = F.one_hot(target, 5).float()
                    mask_true = mask_true.squeeze(1).permute(0, 3, 1, 2)
                    probs += out.cpu().numpy().tolist()
                    dice_score += multiclass_dice_coeff(mask_pred, mask_true, reduce_batch_first=False)


            if self.args.task == 'classification':
                if self.args.dataset == 'NIH' or self.args.dataset == 'CheXpert':
                    try:
                        auc = roc_auc_score(gt, probs)
                    except:
                        auc = 0
                    logging.info("* Qualification Score of participant {} : AUC = {:.2f}*".format(self.client_index+1, auc, acc))
                    # return qualification score, logits
                    return auc, probs
                else:
                    logging.info("*************  Qualification Score (Client {}) : Acc = {:.2f} **************".format(self.client_index, acc))
                    return acc
            elif self.args.task == 'segmentation':
                dice_score /= len(self.qualification_dataloader)
                logging.info("* Qualification Score of participant {} : Dice Score = {:.2f}*".format(self.client_index+1, dice_score))
                return dice_score, probs

    def influencing (self, max_idx, args):

        Loss = torch.nn.KLDivLoss(reduction='batchmean')        
        logits_influencer = Variable(torch.Tensor(self.distill_logits[max_idx]).to(self.device), requires_grad=True)
        # logits_influencer = Variable(self.distill_logits[max_idx].to(self.device), requires_grad=True)
        alpha = args.alpha
        T = args.temperature
        sigmoid = nn.Sigmoid()
        logSigmoid = nn.LogSigmoid()
        eps=1e-8

        for e in range(args.influencing_epochs):
            for client_idx in range(self.num_client):

                if client_idx == max_idx:
                    continue
                if client_idx == self.max_idx:
                    continue

                self.model.load_state_dict(self.model_weights[client_idx])
                self.model.to(self.device)
                self.model.train()

                logits_follower = Variable(torch.Tensor(self.distill_logits[client_idx]).to(self.device), requires_grad=True)
                batch_loss = []

                # distillation
                for i in range(len(logits_influencer)):

                    if self.args.task == "classification":
                        follower = torch.clamp(sigmoid(logits_follower[i]), min=eps, max=1-eps)
                        influencer = torch.clamp(sigmoid(logits_influencer[i]), min=eps, max=1-eps)
                        # print("Follower logits: ", follower)
                        # print("Influencer logits: ", influencer)
                        KD_loss1 = Loss(torch.log(follower),influencer) * alpha
                        # print("Loss1: ", KD_loss1)
                        KD_loss2 = Loss(torch.log(1 - follower), 1 - influencer) * alpha
                        # print("Loss2: ", KD_loss2)
                        KD_loss = KD_loss1 + KD_loss2
                        # print("Loss: ", KD_loss)
                        KD_loss = KD_loss.sum()
                        # print("Sum: ", KD_loss)
                    elif self.args.task == "segmentation":
                        KD_loss = nn.KLDivLoss()(logSigmoid(logits_follower[i]/T),
                             sigmoid(logits_influencer[i]/T)) * (10 * T * T)
                    
                    KD_loss.backward()
                    self.optimizer.step()
                    batch_loss.append(KD_loss.item())

                if e == (args.influencing_epochs - 1):
                    if len(batch_loss) > 0:

                        avg_KD_loss = sum(batch_loss) / len(batch_loss)
                        logging.info('Follower {}. distillation Loss: {:.6f}  Thread {}  Map {}'.format(client_idx+1, avg_KD_loss, current_process()._identity[0], self.client_map[self.round]))
                
                        m = self.model.cpu().state_dict()
                        self.model_weights[client_idx] = m

    def ensemble_influencing (self, max_idx, second_max_idx, args):

        Loss = torch.nn.KLDivLoss(reduction="none")
        logits_influencer1 = torch.Tensor(self.distill_logits[max_idx]).to(self.device)
        logits_influencer2 = torch.Tensor(self.distill_logits[second_max_idx]).to(self.device)
        alpha = args.alpha
        T = args.temperature
        sigmoid = nn.Sigmoid()
        logSigmoid = nn.LogSigmoid()
        eps=1e-8

        logits_influencer = Variable((logits_influencer1 + logits_influencer2) / 2 , requires_grad=True)

        for e in range(args.influencing_epochs):
            for client_idx in range(self.num_client):

                if client_idx == max_idx:
                    continue
                if client_idx == self.max_idx:
                    continue

                self.model.load_state_dict(self.model_weights[client_idx])
                self.model.to(self.device)
                self.model.train()

                logits_follower = Variable(torch.Tensor(self.distill_logits[client_idx]).to(self.device), requires_grad=True)
                batch_loss = []

                # distillation
                for i in range(len(logits_influencer)):

                    follower = torch.clamp(sigmoid(logits_follower[i]), min=eps, max=1-eps)
                    influencer = torch.clamp(sigmoid(logits_influencer[i]), min=eps, max=1-eps)
                    # print("Follower logits: ", follower)
                    # print("Influencer logits: ", influencer)
                    KD_loss1 = Loss(torch.log(follower),influencer) * alpha
                    # print("Loss1: ", KD_loss1)
                    KD_loss2 = Loss(torch.log(1 - follower), 1 - influencer) * alpha
                    # print("Loss2: ", KD_loss2)
                    KD_loss = KD_loss1 + KD_loss2
                    # print("Loss: ", KD_loss)
                    KD_loss = KD_loss.sum()
                    # print("Sum: ", KD_loss)

                    # KD_loss = nn.KLDivLoss()(F.log_softmax(logits_follower[i]/T, dim=1),
                    #          F.softmax(logits_influencer[i]/T, dim=1)) * (alpha * T * T)
                    
                    KD_loss.backward()
                    self.optimizer.step()
                    batch_loss.append(KD_loss.item())

                if e == (args.influencing_epochs - 1):
                    if len(batch_loss) > 0:

                        avg_KD_loss = sum(batch_loss) / len(batch_loss)
                        logging.info('Follower {}. distillation Loss: {:.6f}  Thread {}  Map {}'.format(client_idx+1, avg_KD_loss, current_process()._identity[0], self.client_map[self.round]))
                
                        m = self.model.cpu().state_dict()
                        self.model_weights[client_idx] = m

    def test(self, client_idx):

        self.model.load_state_dict(self.model_weights[client_idx])
        self.model.to(self.device)
        self.model.eval()

        sigmoid = torch.nn.Sigmoid()
        test_correct = 0.0
        test_sample_number = 0.0
        val_loader_examples_num = len(self.test_dataloader.dataset)
        metric = SegmentationMetrics()
        

        if self.args.task == 'classification':
            probs = np.zeros((val_loader_examples_num, self.num_classes), dtype = np.float32)
            gt    = np.zeros((val_loader_examples_num, self.num_classes), dtype = np.float32)
            k=0
        elif self.args.task == 'segmentation':
            probs = []
            dice_score = 0
            recall = 0
            precision = 0

        with torch.no_grad():

            for batch_idx, (x, target) in enumerate(self.test_dataloader):

                target = target.type(torch.LongTensor)
                x = x.to(self.device)
                target = target.to(self.device)
                out = self.model(x)
                
                if self.args.task == "classification":
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

                elif self.args.task == "segmentation":
                    mask_pred = F.one_hot(out.argmax(dim=1), 5).permute(0, 3, 1, 2).float() # 알아서 2번째 dimension을 one hot 해줌
                    mask_true = F.one_hot(target, 5).float()
                    mask_true = mask_true.squeeze(1).permute(0, 3, 1, 2)
                    dice_score += multiclass_dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                    temp_precision, temp_recall = metric.calculate_multi_metrics(mask_true, mask_pred,5)
                    recall += temp_recall
                    precision += temp_precision

            
            torch.save(self.model_weights[client_idx], self.model_dir + "/participant{}.pth".format(client_idx))

            if self.args.task == "classification":
                if self.args.dataset == 'NIH' or self.args.dataset == 'CheXpert':
                    try:
                        auc = roc_auc_score(gt, probs)
                    except:
                        auc = 0

                    logging.info("Participant {} test result: AUC = {:.2f}**************".format(client_idx+1, auc))
                    f = open(self.result_dir + "/participants{}.txt".format(client_idx+1), "a")
                    f.write(str(auc) + "\n")
                    f.close()

                    return auc
                else:
                    logging.info("************* Client {} Acc = {:.2f} **************".format(client_idx, acc))
                    return acc
            elif self.args.task == 'segmentation':
                probs = np.array(probs)
                dice_score /= len(self.test_dataloader)
                recall /= len(self.test_dataloader)
                precision /= len(self.test_dataloader)
                logging.info("Participant {} test result: Dice Score = {:.2f}, precision = {:.2f}, recall = {:.2f}*".format(self.client_index+1, dice_score, precision, recall))
                f = open(self.result_dir + "/participants{}.txt".format(client_idx+1), "a")
                f.write(str(dice_score.item()) + "\n")
                f.close()
                return dice_score
    