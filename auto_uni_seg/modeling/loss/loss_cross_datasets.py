from cmath import inf
from distutils.command.config import config
from traceback import print_tb
from .ohem_ce_loss import OhemCELoss, MdsOhemCELoss

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat, einsum
import logging

def LabelToOneHot(LabelVector, nClass, ignore_index=-1):
    
    OutOneHot = torch.zeros(len(LabelVector), nClass, dtype=torch.bool)
    if LabelVector.is_cuda:
        OutOneHot = OutOneHot.cuda()
        
    OutOneHot[LabelVector!=ignore_index, LabelVector[LabelVector!=ignore_index]]=1
    return OutOneHot

class UnifyPrototypeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, dataset_ids, M):
        ctx.save_for_backward(x, weight, dataset_ids, M)

        output = torch.einsum('bchw, nc -> bnhw', x, weight)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input and weight
        x, weight, dataset_ids, M = ctx.saved_tensors
        # print(grad_output.shape)
        # print(M.shape)
        # print(dataset_ids)
        # sys.stdout.flush()
        
        for i in range(0, int(torch.max(dataset_ids))+1):
            if not (dataset_ids == i).any():
                continue
            grad_output[dataset_ids == i] = torch.einsum('bchw, c -> bchw', grad_output[dataset_ids == i], M[i])

        grad_input = torch.einsum('bchw, cn -> bnhw', grad_output, weight)
        grad_weight = torch.einsum('bchw, bnhw -> cn', grad_output, x)

        # Reshape the gradients back to the original shape
        # grad_input = grad_input_reshaped.view(x.size())
        # print('ret')
        return grad_input, grad_weight, None, None

    
class CrossDatasetsCELoss_AdvGNN(nn.Module):
    def __init__(self, configer=None):
        super(CrossDatasetsCELoss_AdvGNN, self).__init__()
        self.configer = configer
        self.n_datasets = self.configer.get('n_datasets')
        self.num_prototype = self.configer.get('contrast', 'num_prototype')
        self.temperature = self.configer.get('contrast', 'temperature')
        self.with_mulbn = self.configer.get('contrast', 'with_mulbn')
        self.reweight = self.configer.get('loss', 'reweight')
        self.ignore_index = self.configer.get('loss', 'ignore_index')
        self.with_unify_label = self.configer.get('loss', 'with_unify_label')
        self.with_spa = self.configer.get('loss', 'with_spa')
        self.spa_loss_weight = self.configer.get('loss', 'spa_loss_weight')
        self.with_max_enc = self.configer.get('loss', 'with_max_enc')
        self.max_enc_weight = self.configer.get('loss', 'max_enc_weight')
        self.with_datasets_aux = self.configer.get('loss', 'with_datasets_aux')
        self.with_softmax_and_max = self.configer.get('GNN', 'output_softmax_and_max_adj')
        self.with_orth = self.configer.get('GNN', 'with_orth')
        self.with_max_adj = self.configer.get('GNN', 'output_max_adj')
        self.mse_or_adv = self.configer.get('GNN', 'mse_or_adv')
        self.max_iter = self.configer.get('lr', 'max_iter')
        # self.seg_gnn_alter_iters = self.configer.get('train', 'seg_gnn_alter_iters')
        self.gnn_iters = self.configer.get('train', 'gnn_iters')
        self.seg_iters = self.configer.get('train', 'seg_iters')
        self.total_cats = 0
        self.n_cats = []
        for i in range(1, self.n_datasets+1):
            this_cat = self.configer.get('dataset'+str(i), 'n_cats')
            self.n_cats.append(this_cat)
            self.total_cats += this_cat
        self.max_num_unify_class = int(self.configer.get('GNN', 'unify_ratio') * self.total_cats)

        # self.CELoss = nn.CrossEntropyLoss(ignore_index=255)
        self.OhemCELoss = OhemCELoss(0.7, ignore_lb=255)
        self.mdsOhemCELoss = MdsOhemCELoss(self.configer, 0.4, ignore_lb=255)
    
        self.advloss = nn.BCELoss()
        self.adv_loss_weight = self.configer.get('loss', 'adv_loss_weight')
        
        self.MSE_loss = torch.nn.MSELoss()
        self.MSE_sum_loss = torch.nn.MSELoss(reduction='sum')

        self.orth_weight = self.configer.get('GNN', 'orth_weight')
        if self.with_datasets_aux:
            self.aux_weight = self.configer.get('loss', 'aux_weight')
        
        self.adj_loss_weight = self.configer.get('loss', 'adj_loss_weight')
        
        self.GridSpilt = self.configer.get('loss', 'GridSplit')
        if self.GridSpilt:
            cur_cat = 0
            self.M = torch.zeros(self.n_datasets, self.max_num_unify_class)
            for i in range(self.n_datasets):
                this_n = int(0.5*self.max_num_unify_class*self.n_cats[i]/float(self.total_cats))
                self.M[i, cur_cat:cur_cat+this_n] = 1
                cur_cat += this_n
            
            self.M[:, cur_cat:] = 1
        
    
    def similarity_dsb(self, proto_vecs, reduce='mean'):
        """
        Compute EM loss with the probability-based distribution of each feature
        :param feat_domain: source, target or both
        :param temperature: softmax temperature
        """


        # dot similarity between features and centroids
        z = torch.mm(proto_vecs, proto_vecs.t())  # size N x C_seen

        # entropy loss to push each feature to be similar to only one class prototype (no supervision)
        if reduce == 'mean':
            loss = -1 * torch.mean(F.softmax(z / self.temperature, dim=1) * F.log_softmax(z / self.temperature, dim=1))
        elif reduce == 'sum':
            loss = -1 * torch.sum(F.softmax(z / self.temperature, dim=1) * F.log_softmax(z / self.temperature, dim=1))
            

        return loss


    def forward(self, preds, target, dataset_ids, is_adv=True, init_gnn_stage=False):
        logger = logging.getLogger()
        # if not is_adv:
        CELoss = self.mdsOhemCELoss
        # else:
        #     CELoss = self.CELoss

        logits = preds['seg']
        if 'aux' in preds:
            aux_logits = preds['aux']
        
        isSecondStage = False
        if 'gnn_stage' in preds:
            isSecondStage = preds['gnn_stage']
        
        unify_prototype = preds['unify_prototype']
        bi_graphs = preds['bi_graphs']
        adj_matrix = None
        if 'adj' in preds:
            adj_matrix = preds['adj']
            pretrain_bipart_graph = preds['pretrain_bipart_graph']
        
        adj_feat = None
        if 'adj_feat' in preds:
            adj_feat = preds['adj_feat']
        
        target_bi_graph = None
        if 'target_bi_graph' in preds:
            target_bi_graph = preds['target_bi_graph']
        
        if is_adv and self.mse_or_adv != "None":
            adv_out = preds['adv_out']
            
            label_real = torch.zeros(adv_out['ADV1'][0].shape[0], 1)
            label_fake = torch.ones(adv_out['ADV1'][0].shape[0], 1)
            
            if adv_out['ADV1'][0].is_cuda:
                label_real = label_real.cuda()
                label_fake = label_fake.cuda()
        
        loss = None
        adv_loss = None
        orth_loss = None
        graph_loss = None
        mse_loss = None
        aux_loss = None
        adj_loss = None
        if unify_prototype is not None:# and not init_gnn_stage:
            if self.with_datasets_aux:
                cur_cat = 0
                aux_logits = []
                for i in range(0, self.n_datasets):
                    if not (dataset_ids == i).any():
                        aux_logits.append([])
                        cur_cat += self.n_cats[i]
                        continue 
                    this_aux_logit = torch.einsum('bchw, nc -> bnhw', logits[dataset_ids == i], unify_prototype[cur_cat:cur_cat+self.n_cats[i]])
                    aux_logits.append(F.interpolate(this_aux_logit, size=(target.size(1), target.size(2)), mode="bilinear", align_corners=True))
                    cur_cat += self.n_cats[i]
                
                if self.GridSpilt:
                    self.M.grad = None
                    if logits.is_cuda:
                        self.M = self.M.cuda()
                    logits = UnifyPrototypeFunction.apply(logits, unify_prototype[self.total_cats:], dataset_ids, self.M)
                else:
                    logits = torch.einsum('bchw, nc -> bnhw', logits, unify_prototype[self.total_cats:])
                

            else:
                if self.GridSpilt:
                    self.M.grad = None
                    if logits.is_cuda:
                        self.M = self.M.cuda()
                    logits = UnifyPrototypeFunction.apply(logits, unify_prototype, dataset_ids, self.M)
                else:
                    logits = torch.einsum('bchw, nc -> bnhw', logits, unify_prototype)
                
        # print('logits', logits.shape)
        # print("logits_max : {}, logits_min : {}".format(torch.max(logits), torch.min(logits)))
        # logger.info('logit min: {}, logits max:{}'.format(torch.min(logits), torch.max(logits)))
        if is_adv and self.with_orth:# and adj_feat is None:
            # logger.info(adj_feat)
            if self.with_datasets_aux:
                orth_loss = self.orth_weight * self.similarity_dsb(unify_prototype[self.total_cats:])
            else:
                orth_loss = self.orth_weight * self.similarity_dsb(unify_prototype)
            # orth_loss = self.orth_weight * self.similarity_dsb(adj_feat)
            # loss = orth_loss
       
        
        remap_logits = []
        max_remap_logits = []
        incr_cat = 0
        
        for i in range(0, self.n_datasets):
            incr_cat += self.n_cats[i]
            # print("logits shape:", )
            if not (dataset_ids == i).any():
                continue
            
            
            if not init_gnn_stage:
                if is_adv and self.with_softmax_and_max and self.with_max_adj and not isSecondStage and len(bi_graphs) == 2*self.n_datasets:
                    max_remap_logit = torch.einsum('bchw, nc -> bnhw', logits[dataset_ids==i], bi_graphs[2*i])
                    max_remap_logit = F.interpolate(max_remap_logit, size=(target.size(1), target.size(2)), mode="bilinear", align_corners=True) 

                    softmax_remap_logits = torch.einsum('bchw, nc -> bnhw', logits[dataset_ids==i], bi_graphs[2*i + 1])
                    softmax_remap_logits = F.interpolate(softmax_remap_logits, size=(target.size(1), target.size(2)), mode="bilinear", align_corners=True) 
                    
                    max_remap_logits.append(max_remap_logit)
                    remap_logits.append(softmax_remap_logits)
                else:
                    remap_logit = torch.einsum('bchw, nc -> bnhw', logits[dataset_ids==i], bi_graphs[i])
                    remap_logit = F.interpolate(remap_logit, size=(target.size(1), target.size(2)), mode="bilinear", align_corners=True)

                    remap_logits.append(remap_logit)                

            if is_adv and self.with_spa and not isSecondStage and len(bi_graphs)==2*self.n_datasets:
                if len(bi_graphs)==2*self.n_datasets:
                    spa_loss = self.spa_loss_weight * torch.pow(torch.norm(bi_graphs[2*i+1], p='fro'), 2)
                else:
                    spa_loss = self.spa_loss_weight * torch.pow(torch.norm(bi_graphs[i], p='fro'), 2)
                if loss is None:
                    loss = spa_loss
                else:
                    loss = loss + spa_loss
            
            if is_adv and self.with_max_enc:
                max_enc_loss = self.max_enc_weight * self.MSE_loss(torch.max(bi_graphs[i], dim=1)[0], torch.ones(bi_graphs[i].size(0)).cuda())
                if loss is None:
                    loss = max_enc_loss
                else:
                    loss = loss + max_enc_loss

            if is_adv and target_bi_graph is not None and not isSecondStage:
                total_num = bi_graphs[i].shape[1]
                base_weight = 1 / total_num
                # bi_graphs[2*i + 1]
                if adj_loss is None:
                    if len(bi_graphs) == 2*self.n_datasets:
                        adj_loss = base_weight * self.MSE_sum_loss(bi_graphs[2*i+1][target_bi_graph[i] != 255], target_bi_graph[i][target_bi_graph[i] != 255])
                    else:
                        adj_loss = base_weight * self.MSE_sum_loss(bi_graphs[i][target_bi_graph[i] != 255], target_bi_graph[i][target_bi_graph[i] != 255])
                else:
                    if len(bi_graphs) == 2*self.n_datasets:
                        adj_loss += base_weight * self.MSE_sum_loss(bi_graphs[2*i+1][target_bi_graph[i] != 255], target_bi_graph[i][target_bi_graph[i] != 255])
                    else:
                        adj_loss += base_weight * self.MSE_sum_loss(bi_graphs[i][target_bi_graph[i] != 255], target_bi_graph[i][target_bi_graph[i] != 255])
                    
            
            if self.with_datasets_aux:
                if is_adv:
                    if aux_loss is None:
                        aux_loss = self.OhemCELoss(aux_logits[i], target[dataset_ids==i])
                    else:
                        aux_loss += self.OhemCELoss(aux_logits[i], target[dataset_ids==i])
                # else:
                #     this_aux_logit = F.interpolate(aux_logits[i][dataset_ids==i], size=(target.size(1), target.size(2)), mode="bilinear", align_corners=True)
                #     if aux_loss is None:
                        
                #         aux_loss = self.OhemCELoss(this_aux_logit, target[dataset_ids==i])
                #     else:
                #         aux_loss += self.OhemCELoss(this_aux_logit, target[dataset_ids==i])
        
        # remap_logits = torch.cat(remap_logits, dim=0)
        # if max_remap_logits:
        #     max_remap_logits = torch.cat(max_remap_logits, dim=0)
            # print(torch.sum(bi_graphs[i]))
        if not init_gnn_stage:
            if is_adv and self.with_softmax_and_max and self.with_max_adj and len(bi_graphs)==2*self.n_datasets:
                cur_iter = self.configer.get('iter')
                cur_iter = cur_iter % (self.gnn_iters+self.seg_iters) % self.gnn_iters
                    
                max_rate = float(cur_iter) / self.gnn_iters
                if loss is None:
                    loss = (max_rate * CELoss(max_remap_logits, target, dataset_ids) + (1 - max_rate) * CELoss(remap_logits, target, dataset_ids))
                else:
                    loss = loss + ( max_rate * CELoss(max_remap_logits, target, dataset_ids) + (1 - max_rate) * CELoss(remap_logits, target, dataset_ids))
                
            else:
                celoss = CELoss(remap_logits, target, dataset_ids)

                    
                if loss is None or torch.isnan(loss):
                    loss = celoss
                else:
                    loss = loss + celoss


                # needed_adj = adj_matrix[:self.total_cats, :self.total_cats]
                # needed_adj = needed_adj.contiguous().view(-1)
                # graph_loss = 100 * self.MSE_loss(needed_adj, pretrain_bipart_graph.view(-1))
                # if loss is None:
                #     loss = graph_loss
                # else:
                #     loss += graph_loss

        if init_gnn_stage and adj_matrix is not None:
            cur_cat = 0
            for j in range(0, self.n_datasets):

                cur_cat += self.n_cats[j]
                # print(adj_matrix.keys())
                # print(adj_matrix)
                needed_adj = adj_matrix[cur_cat-self.n_cats[j]:cur_cat, self.total_cats:]
                # needed_adj = needed_adj.contiguous().view(-1)
                if graph_loss is None:
                    graph_loss = 10 * self.MSE_loss(needed_adj, pretrain_bipart_graph[j])
                else:
                    graph_loss += 10 * self.MSE_loss(needed_adj, pretrain_bipart_graph[j])
                    
            if loss is None:
                loss = graph_loss
            else:
                loss += graph_loss          
    
        if init_gnn_stage:
            mse_loss = self.n_datasets * 10 * self.MSE_loss(unify_prototype, logits)
            if loss is None:
                loss = mse_loss
            else:
                loss = loss + mse_loss
        
        if is_adv:  
            if self.mse_or_adv == 'adv':
                real_out = self.advloss(adv_out['ADV1'][0], label_real) + self.advloss(adv_out['ADV2'][0], label_real) + self.advloss(adv_out['ADV3'][0], label_real)
                fake_out = self.advloss(adv_out['ADV1'][1], label_fake) + self.advloss(adv_out['ADV2'][1], label_fake) + self.advloss(adv_out['ADV3'][1], label_fake)
                adv_loss = real_out + fake_out

                G_fake_out = self.advloss(adv_out['ADV1'][2], label_real) + self.advloss(adv_out['ADV2'][2], label_real) + self.advloss(adv_out['ADV3'][2], label_real)
                loss = loss + self.adv_loss_weight * G_fake_out
            elif self.mse_or_adv == 'mse':
                adv_loss = self.MSE_loss(adv_out['ADV1'][1], adv_out['ADV1'][0]) + self.MSE_loss(adv_out['ADV2'][1], adv_out['ADV2'][0]) + self.MSE_loss(adv_out['ADV3'][1], adv_out['ADV3'][0])
                loss = loss + self.adv_loss_weight * adv_loss
                
        if aux_loss:
            loss = loss + self.aux_weight * aux_loss
        
        if orth_loss:
            loss += orth_loss
        
        if adj_loss:
            loss += self.adj_loss_weight * adj_loss
        
        return loss, orth_loss, aux_loss, adj_loss #, graph_loss, mse_loss
    
class CrossDatasetsCELoss_AdvGNN_ce(nn.Module):
    def __init__(self, configer=None):
        super(CrossDatasetsCELoss_AdvGNN_ce, self).__init__()
        self.configer = configer
        self.n_datasets = self.configer.get('n_datasets')
        self.num_prototype = self.configer.get('contrast', 'num_prototype')
        self.temperature = self.configer.get('contrast', 'temperature')
        self.with_mulbn = self.configer.get('contrast', 'with_mulbn')
        self.reweight = self.configer.get('loss', 'reweight')
        self.ignore_index = self.configer.get('loss', 'ignore_index')
        self.with_unify_label = self.configer.get('loss', 'with_unify_label')
        self.with_spa = self.configer.get('loss', 'with_spa')
        self.spa_loss_weight = self.configer.get('loss', 'spa_loss_weight')
        self.with_max_enc = self.configer.get('loss', 'with_max_enc')
        self.max_enc_weight = self.configer.get('loss', 'max_enc_weight')
        self.with_datasets_aux = self.configer.get('loss', 'with_datasets_aux')
        self.with_softmax_and_max = self.configer.get('GNN', 'output_softmax_and_max_adj')
        self.with_orth = self.configer.get('GNN', 'with_orth')
        self.with_max_adj = self.configer.get('GNN', 'output_max_adj')
        self.mse_or_adv = self.configer.get('GNN', 'mse_or_adv')
        self.max_iter = self.configer.get('lr', 'max_iter')
        # self.seg_gnn_alter_iters = self.configer.get('train', 'seg_gnn_alter_iters')
        self.gnn_iters = self.configer.get('train', 'gnn_iters')
        self.seg_iters = self.configer.get('train', 'seg_iters')
        self.total_cats = 0
        self.n_cats = []
        for i in range(1, self.n_datasets+1):
            this_cat = self.configer.get('dataset'+str(i), 'n_cats')
            self.n_cats.append(this_cat)
            self.total_cats += this_cat
        self.max_num_unify_class = int(self.configer.get('GNN', 'unify_ratio') * self.total_cats)

        self.CELoss = nn.CrossEntropyLoss(ignore_index=255)
        self.OhemCELoss = OhemCELoss(0.7, ignore_lb=255)
        self.mdsOhemCELoss = MdsOhemCELoss(self.configer, 0.4, ignore_lb=255)
    
        self.advloss = nn.BCELoss()
        self.adv_loss_weight = self.configer.get('loss', 'adv_loss_weight')
        
        self.MSE_loss = torch.nn.MSELoss()
        self.MSE_sum_loss = torch.nn.MSELoss(reduction='sum')

        self.orth_weight = self.configer.get('GNN', 'orth_weight')
        if self.with_datasets_aux:
            self.aux_weight = self.configer.get('loss', 'aux_weight')
        
        self.adj_loss_weight = self.configer.get('loss', 'adj_loss_weight')
        
        self.GridSpilt = self.configer.get('loss', 'GridSplit')
        if self.GridSpilt:
            cur_cat = 0
            self.M = torch.zeros(self.n_datasets, self.max_num_unify_class)
            for i in range(self.n_datasets):
                this_n = int(0.5*self.max_num_unify_class*self.n_cats[i]/float(self.total_cats))
                self.M[i, cur_cat:cur_cat+this_n] = 1
                cur_cat += this_n
            
            self.M[:, cur_cat:] = 1
        
    
    def similarity_dsb(self, proto_vecs, reduce='mean'):
        """
        Compute EM loss with the probability-based distribution of each feature
        :param feat_domain: source, target or both
        :param temperature: softmax temperature
        """


        # dot similarity between features and centroids
        z = torch.mm(proto_vecs, proto_vecs.t())  # size N x C_seen

        # entropy loss to push each feature to be similar to only one class prototype (no supervision)
        if reduce == 'mean':
            loss = -1 * torch.mean(F.softmax(z / self.temperature, dim=1) * F.log_softmax(z / self.temperature, dim=1))
        elif reduce == 'sum':
            loss = -1 * torch.sum(F.softmax(z / self.temperature, dim=1) * F.log_softmax(z / self.temperature, dim=1))
            

        return loss


    def forward(self, preds, target, dataset_ids, is_adv=True, init_gnn_stage=False):
        logger = logging.getLogger()
        # if not is_adv:
        # CELoss = self.mdsOhemCELoss
        # else:
        CELoss = self.OhemCELoss

        logits = preds['seg']
        if 'aux' in preds:
            aux_logits = preds['aux']
        
        isSecondStage = False
        if 'gnn_stage' in preds:
            isSecondStage = preds['gnn_stage']
        
        unify_prototype = preds['unify_prototype']
        bi_graphs = preds['bi_graphs']
        adj_matrix = None
        if 'adj' in preds:
            adj_matrix = preds['adj']
            pretrain_bipart_graph = preds['pretrain_bipart_graph']
        
        adj_feat = None
        if 'adj_feat' in preds:
            adj_feat = preds['adj_feat']
        
        target_bi_graph = None
        if 'target_bi_graph' in preds:
            target_bi_graph = preds['target_bi_graph']
        
        if is_adv and self.mse_or_adv != "None":
            adv_out = preds['adv_out']
            
            label_real = torch.zeros(adv_out['ADV1'][0].shape[0], 1)
            label_fake = torch.ones(adv_out['ADV1'][0].shape[0], 1)
            
            if adv_out['ADV1'][0].is_cuda:
                label_real = label_real.cuda()
                label_fake = label_fake.cuda()
        
        loss = None
        adv_loss = None
        orth_loss = None
        graph_loss = None
        mse_loss = None
        aux_loss = None
        adj_loss = None
        if unify_prototype is not None: # and not init_gnn_stage:
            if self.with_datasets_aux:
                cur_cat = 0
                aux_logits = []
                for i in range(0, self.n_datasets):
                    if not (dataset_ids == i).any():
                        aux_logits.append([])
                        cur_cat += self.n_cats[i]
                        continue 
                    this_aux_logit = torch.einsum('bchw, nc -> bnhw', logits[dataset_ids == i], unify_prototype[cur_cat:cur_cat+self.n_cats[i]])
                    aux_logits.append(F.interpolate(this_aux_logit, size=(target.size(1), target.size(2)), mode="bilinear", align_corners=True))
                    cur_cat += self.n_cats[i]
                
                if self.GridSpilt:
                    self.M.grad = None
                    if logits.is_cuda:
                        self.M = self.M.cuda()
                    logits = UnifyPrototypeFunction.apply(logits, unify_prototype[self.total_cats:], dataset_ids, self.M)
                else:
                    logits = torch.einsum('bchw, nc -> bnhw', logits, unify_prototype[self.total_cats:])
                

            else:
                if self.GridSpilt:
                    self.M.grad = None
                    if logits.is_cuda:
                        self.M = self.M.cuda()
                    logits = UnifyPrototypeFunction.apply(logits, unify_prototype, dataset_ids, self.M)
                else:
                    logits = torch.einsum('bchw, nc -> bnhw', logits, unify_prototype)
                
        # print('logits', logits.shape)
        # print("logits_max : {}, logits_min : {}".format(torch.max(logits), torch.min(logits)))
        # logger.info('logit min: {}, logits max:{}'.format(torch.min(logits), torch.max(logits)))
        if is_adv and self.with_orth:# and adj_feat is None:
            # logger.info(adj_feat)
            if self.with_datasets_aux:
                orth_loss = self.orth_weight * self.similarity_dsb(unify_prototype[self.total_cats:])
            else:
                orth_loss = self.orth_weight * self.similarity_dsb(unify_prototype)
            # orth_loss = self.orth_weight * self.similarity_dsb(adj_feat)
            # loss = orth_loss
       
        
        remap_logits = []
        max_remap_logits = []
        incr_cat = 0
        
        for i in range(0, self.n_datasets):
            incr_cat += self.n_cats[i]
            # print("logits shape:", )
            if not (dataset_ids == i).any():
                continue
            
            
            # if not init_gnn_stage:
            if is_adv and self.with_softmax_and_max and self.with_max_adj and not isSecondStage and len(bi_graphs) == 2*self.n_datasets:
                max_remap_logit = torch.einsum('bchw, nc -> bnhw', logits[dataset_ids==i], bi_graphs[2*i])
                max_remap_logit = F.interpolate(max_remap_logit, size=(target.size(1), target.size(2)), mode="bilinear", align_corners=True) 

                softmax_remap_logits = torch.einsum('bchw, nc -> bnhw', logits[dataset_ids==i], bi_graphs[2*i + 1])
                softmax_remap_logits = F.interpolate(softmax_remap_logits, size=(target.size(1), target.size(2)), mode="bilinear", align_corners=True) 
                
                max_remap_logits.append(max_remap_logit)
                remap_logits.append(softmax_remap_logits)
            else:
                remap_logit = torch.einsum('bchw, nc -> bnhw', logits[dataset_ids==i], bi_graphs[i])
                remap_logit = F.interpolate(remap_logit, size=(target.size(1), target.size(2)), mode="bilinear", align_corners=True)

                remap_logits.append(remap_logit)                

            if is_adv and self.with_spa and not isSecondStage and len(bi_graphs)==2*self.n_datasets and not init_gnn_stage:
                if len(bi_graphs)==2*self.n_datasets:
                    spa_loss = self.spa_loss_weight * torch.pow(torch.norm(bi_graphs[2*i+1], p='fro'), 2)
                else:
                    spa_loss = self.spa_loss_weight * torch.pow(torch.norm(bi_graphs[i], p='fro'), 2)
                if loss is None:
                    loss = spa_loss
                else:
                    loss = loss + spa_loss
            
            if is_adv and self.with_max_enc:
                max_enc_loss = self.max_enc_weight * self.MSE_loss(torch.max(bi_graphs[i], dim=1)[0], torch.ones(bi_graphs[i].size(0)).cuda())
                if loss is None:
                    loss = max_enc_loss
                else:
                    loss = loss + max_enc_loss

            if is_adv and target_bi_graph is not None and not isSecondStage:
                total_num = bi_graphs[i].shape[1]
                base_weight = 1 / total_num
                # bi_graphs[2*i + 1]
                if adj_loss is None:
                    if len(bi_graphs) == 2*self.n_datasets:
                        adj_loss = base_weight * self.MSE_sum_loss(bi_graphs[2*i+1][target_bi_graph[i] != 255], target_bi_graph[i][target_bi_graph[i] != 255])
                    else:
                        adj_loss = base_weight * self.MSE_sum_loss(bi_graphs[i][target_bi_graph[i] != 255], target_bi_graph[i][target_bi_graph[i] != 255])
                else:
                    if len(bi_graphs) == 2*self.n_datasets:
                        adj_loss += base_weight * self.MSE_sum_loss(bi_graphs[2*i+1][target_bi_graph[i] != 255], target_bi_graph[i][target_bi_graph[i] != 255])
                    else:
                        adj_loss += base_weight * self.MSE_sum_loss(bi_graphs[i][target_bi_graph[i] != 255], target_bi_graph[i][target_bi_graph[i] != 255])
                    
            
            if self.with_datasets_aux:
                if is_adv:
                    if aux_loss is None:
                        aux_loss = CELoss(aux_logits[i], target[dataset_ids==i])
                    else:
                        aux_loss += CELoss(aux_logits[i], target[dataset_ids==i])
                # else:
                #     this_aux_logit = F.interpolate(aux_logits[i][dataset_ids==i], size=(target.size(1), target.size(2)), mode="bilinear", align_corners=True)
                #     if aux_loss is None:
                        
                #         aux_loss = CELoss(this_aux_logit, target[dataset_ids==i])
                #     else:
                #         aux_loss += CELoss(this_aux_logit, target[dataset_ids==i])
        
        # remap_logits = torch.cat(remap_logits, dim=0)
        # if max_remap_logits:
        #     max_remap_logits = torch.cat(max_remap_logits, dim=0)
            # print(torch.sum(bi_graphs[i]))
            # if not init_gnn_stage:
            if is_adv and self.with_softmax_and_max and self.with_max_adj and len(bi_graphs)==2*self.n_datasets:
                cur_iter = self.configer.get('iter')
                cur_iter = cur_iter % (self.gnn_iters+self.seg_iters) % self.gnn_iters
                    
                max_rate = float(cur_iter) / self.gnn_iters
                if loss is None:
                    loss = (max_rate * CELoss(max_remap_logit, target[dataset_ids==i]) + (1 - max_rate) * CELoss(softmax_remap_logits, target[dataset_ids==i]))
                else:
                    loss = loss + (max_rate * CELoss(max_remap_logit, target[dataset_ids==i]) + (1 - max_rate) * CELoss(softmax_remap_logits, target[dataset_ids==i]))
                
            else:
                celoss = CELoss(remap_logit, target[dataset_ids==i])
                    
                if loss is None or torch.isnan(loss):
                    loss = celoss
                else:
                    loss = loss + celoss


                # needed_adj = adj_matrix[:self.total_cats, :self.total_cats]
                # needed_adj = needed_adj.contiguous().view(-1)
                # graph_loss = 100 * self.MSE_loss(needed_adj, pretrain_bipart_graph.view(-1))
                # if loss is None:
                #     loss = graph_loss
                # else:
                #     loss += graph_loss

        # if init_gnn_stage and adj_matrix is not None:
        #     cur_cat = 0
        #     for j in range(0, self.n_datasets):

        #         cur_cat += self.n_cats[j]
        #         # print(adj_matrix.keys())
        #         # print(adj_matrix)
        #         needed_adj = adj_matrix[cur_cat-self.n_cats[j]:cur_cat, self.total_cats:]
        #         # needed_adj = needed_adj.contiguous().view(-1)
        #         if graph_loss is None:
        #             graph_loss = 10 * self.MSE_loss(needed_adj, pretrain_bipart_graph[j])
        #         else:
        #             graph_loss += 10 * self.MSE_loss(needed_adj, pretrain_bipart_graph[j])
                    
        #     if loss is None:
        #         loss = graph_loss
        #     else:
        #         loss += graph_loss          
    
        # if init_gnn_stage:
        #     mse_loss = self.n_datasets * 10 * self.MSE_loss(unify_prototype, logits)
        #     if loss is None:
        #         loss = mse_loss
        #     else:
        #         loss = loss + mse_loss
        
        if is_adv:  
            if self.mse_or_adv == 'adv':
                real_out = self.advloss(adv_out['ADV1'][0], label_real) + self.advloss(adv_out['ADV2'][0], label_real) + self.advloss(adv_out['ADV3'][0], label_real)
                fake_out = self.advloss(adv_out['ADV1'][1], label_fake) + self.advloss(adv_out['ADV2'][1], label_fake) + self.advloss(adv_out['ADV3'][1], label_fake)
                adv_loss = real_out + fake_out

                G_fake_out = self.advloss(adv_out['ADV1'][2], label_real) + self.advloss(adv_out['ADV2'][2], label_real) + self.advloss(adv_out['ADV3'][2], label_real)
                loss = loss + self.adv_loss_weight * G_fake_out
            elif self.mse_or_adv == 'mse':
                adv_loss = self.MSE_loss(adv_out['ADV1'][1], adv_out['ADV1'][0]) + self.MSE_loss(adv_out['ADV2'][1], adv_out['ADV2'][0]) + self.MSE_loss(adv_out['ADV3'][1], adv_out['ADV3'][0])
                loss = loss + self.adv_loss_weight * adv_loss
                
        if aux_loss:
            loss = loss + self.aux_weight * aux_loss
        
        if orth_loss:
            loss += orth_loss
        
        if adj_loss:
            loss += self.adj_loss_weight * adj_loss
        
        return loss, orth_loss, aux_loss, adj_loss #, graph_loss, mse_loss
    


    

if __name__ == "__main__":
    test_CrossDatasetsCELoss()
    # test_LabelToOneHot()
    # loss_fuc = PixelPrototypeDistanceLoss()
    # a = torch.randn(2,4,3,2)
    # print(a)
    # lb = torch.tensor([[[0,1],[2,0],[255,0]],[[2,1],[1,255],[255,255]]])
    # seq = torch.randn(3,4)
    # print(seq)
    # print(loss_fuc(a,lb,seq))
       

