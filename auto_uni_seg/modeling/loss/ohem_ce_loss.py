#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F




class OhemCELoss(nn.Module):

    def __init__(self, thresh, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)) #.cuda()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        if logits.is_cuda:
            self.thresh.cuda()

        n_min = labels[labels != self.ignore_lb].numel() // 16
   
        loss = self.criteria(logits, labels).view(-1)
        
        # print(torch.max(loss))
        loss_hard = loss[loss > self.thresh.to(logits.dtype)]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        
        return torch.mean(loss_hard)
    
class MdsOhemCELoss(nn.Module):

    def __init__(self, n_datasets, thresh, ignore_lb=255):
        super(MdsOhemCELoss, self).__init__()
        self.n_datasets = n_datasets
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)) #.cuda()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')
        # self.criteria = FocalLoss(gamma=2, ignore_index=ignore_lb, reduction='none')
        # self.criterias = [RecallCrossEntropy(n_classes=self.configer.get(f'dataset{i+1}', 'n_cats'), ignore_index=ignore_lb) for i in range(self.n_datasets)]

    def forward(self, logits, labels, dataset_ids):
        if labels.is_cuda:
            self.thresh.cuda()

        n_min = labels[labels != self.ignore_lb].numel() // 16
        losses = []
        loss_hardes = []
        
        cur_index = 0
        for i in range(self.n_datasets):
            if not (dataset_ids == i).any():
                continue
            # loss = self.criterias[i](logits[cur_index], labels[dataset_ids==i]).view(-1)
            this_logit = logits[cur_index]
            this_label = labels[dataset_ids==i]
            if this_logit.shape[2] != this_label.shape[2]:
                this_logit = F.interpolate(this_logit, size=(this_label.shape[2], this_label.shape[3]), mode="bilinear", align_corners=True)
            loss = self.criteria(this_logit, this_label).view(-1)
            loss_hard = loss[loss > self.thresh]

            cur_index+=1
            # print(loss.shape)
            
            losses.append(loss.clone())
            # loss_hardes.append(loss_hard.clone())
        
        losses = torch.cat(losses, dim=0)
        # loss_hard = torch.cat(loss_hardes, dim=0)
        
        # print("losses shape: ", losses.shape)
        loss_hard = losses[losses > self.thresh]
        # # print("loss.shape: ", losses.shape)
        # # print("loss_hard.shape: ", loss_hard.shape)
            
        if loss_hard.numel() < n_min:
            loss_hard, _ = losses.topk(n_min)
        
        # print("loss_hard.mean: ", torch.mean(loss_hard))
        # ret = None
        # for i in range(len(loss_hard)):
        #     if ret is None:
        #         ret = torch.mean(loss_hard[i])
        #     else:
        #         ret += torch.mean(loss_hard[i])
            # print("loss_hard[{}].mean: ".format(i), torch.mean(loss_hard[i]))
        
        return {'loss_ce': torch.mean(loss_hard)}

    
if __name__ == '__main__':
    logits = torch.randn((4, 19, 512, 1024)).cuda()
    labels = torch.ones((4, 512, 1024)).long().cuda()
    lossfunc = OhemCELoss(0.7)
    print(lossfunc(logits, labels))
