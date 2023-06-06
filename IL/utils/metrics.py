from torch import Tensor
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = True, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]

class SegmentationMetrics(object):

    def __init__(self, eps=1e-5, average=True, ignore_background=True, activation='0-1'):
        self.eps = eps
        self.average = average
        self.ignore = ignore_background
        self.activation = activation

    def _get_class_data(self, gt_onehot, pred, class_num):
        # perform calculation on a batch
        # for precise result in a single image, plz set batch size to 1
        matrix = np.zeros((4, class_num))
        # gt_onehot = gt_onehot.cpu().numpy()
        # pred = pred.cpu.numpy()

        # print("size: ", gt_onehot.size(), pred.size())

        # calculate tp, fp, fn per class
        for i in range(class_num):
            # pred shape: (N, H, W)
            class_pred = pred[:, i, :, :]
            # gt shape: (N, H, W), binary array where 0 denotes negative and 1 denotes positive
            class_gt = gt_onehot[:, i, :, :]

            # contiguous() : way of saving elements in memory (row) 
            # .view(-1, ) : flatten the tensor
            pred_flat = class_pred.reshape(-1)  # shape: (N * H * W, )
            # print(pred_flat.cpu().numpy().tolist())
            gt_flat = class_gt.reshape(-1)  # shape: (N * H * W, )
            # print(gt_flat.cpu().numpy().tolist())
            # print("size: ", gt_flat.size(), pred_flat.size())

            tp = torch.dot(gt_flat, pred_flat)
            fp = torch.sum(pred_flat) - tp # real : 0, output : 1
            fn = torch.sum(gt_flat) - tp  # real : 1, output : 0
            tn = len(pred_flat) - (tp + fp + fn)

            matrix[:, i] = tp.item(), fp.item(), fn.item(), tn.item() # add by column
            # print(tp, tn, fp, fn)

        return matrix

    def calculate_multi_metrics(self, gt, pred, class_num):
        # calculate metrics in multi-class segmentation
        matrix = self._get_class_data(gt, pred, class_num)
        if self.ignore:
            matrix = matrix[:, 1:]

        tp = np.sum(matrix[0, :])
        fp = np.sum(matrix[1, :])
        fn = np.sum(matrix[2, :])
        tn = np.sum(matrix[3, :])

        pixel_acc = (tp + tn + self.eps) / (np.sum(matrix) + self.eps)
        dice = (2 * tp + self.eps) / (tp + fp + fn + self.eps)
        precision = (tp + self.eps) / (tp + fp + self.eps)
        recall = (tp + self.eps) / (tp + fn + self.eps)

        # if self.average:
            # dice = np.average(dice)
            # precision = np.average(precision)
            # recall = np.average(recall)

        return pixel_acc, dice, precision, recall, np.array(matrix)

class CriterionPixelWise(nn.Module):
    def __init__(self, ignore_index=255, use_weight=True, reduce=True):
        super(CriterionPixelWise, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds_S, preds_T):
        preds_T.detach()
        assert preds_S.shape == preds_T.shape,'the output dim of teacher and student differ'
        C,W,H = preds_S.shape
        softmax_pred_T = F.softmax(preds_T.permute(1,2,0).contiguous().view(-1,C), dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        loss = (torch.sum( - softmax_pred_T * logsoftmax(preds_S.permute(1,2,0).contiguous().view(-1,C))))/W/H
        return loss

class CriterionPairWiseforWholeFeatAfterPool(nn.Module):
    def __init__(self, scale, feat_ind):
        '''inter pair-wise loss from inter feature maps'''
        super(CriterionPairWiseforWholeFeatAfterPool, self).__init__()
        self.criterion = sim_dis_compute
        self.feat_ind = feat_ind
        self.scale = scale

    def forward(self, preds_S, preds_T):
        feat_S = preds_S[self.feat_ind]
        feat_T = preds_T[self.feat_ind]
        feat_T.detach()

        total_w, total_h = feat_T.shape[2], feat_T.shape[3]
        patch_w, patch_h = int(total_w*self.scale), int(total_h*self.scale)
        maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
        loss = self.criterion(maxpool(feat_S), maxpool(feat_T))
        return loss