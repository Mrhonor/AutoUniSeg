import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.modeling.postprocessing import sem_seg_postprocess


import logging
from detectron2.utils.events import get_event_storage, EventStorage
import numpy as np
import pickle
import importlib.util
import sys

from .modeling.GNN.gen_graph_node_feature import gen_graph_node_feature
from .modeling.GNN.ltbgnn_llama import build_GNN_module
from .modeling.backbone.hrnet_backbone import HighResolutionNet
from .modeling.loss.ohem_ce_loss import OhemCELoss, MdsOhemCELoss
from .modeling.loss.relation_loss import relation_loss
from .modeling.loss.ow_loss import MdsOWLoss
from timm.models.layers import trunc_normal_
import clip


logger = logging.getLogger(__name__)

@META_ARCH_REGISTRY.register()
class AFFormer_ARCH(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """
    @configurable
    def __init__(self, 
                 seg_model, 
                 gnn_model,
                 pixel_mean, 
                 pixel_std, 
                 datasets_cats, 
                 size_divisibility, 
                 num_unify_classes,
                 graph_node_features,
                 init_gnn_iters,
                 Pretraining,
                 gnn_iters,
                 seg_iters,
                 max_iters,
                 first_stage_gnn_iters,
                 with_spa_loss,
                 with_orth_loss,
                 with_adj_loss,
                 with_relation_loss,
                 with_gaussian_loss,
                 relation_gt_graph,
                 loss_weight_dict,
                 ohem_thresh,
                 ignore_lb,
                 ):
        super(AFFormer_ARCH, self).__init__()
        self.seg_model = seg_model
        self.gnn_model = gnn_model
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.datasets_cats = datasets_cats
        self.n_datasets = len(datasets_cats)
        self.size_divisibility = size_divisibility
        self.num_unify_classes = num_unify_classes
        self.with_spa_loss = with_spa_loss
        self.with_orth_loss = with_orth_loss
        self.with_adj_loss = with_adj_loss
        self.with_relation_loss = with_relation_loss
        self.with_gaussian_loss = with_gaussian_loss
        
        self.relation_gt_graph = relation_gt_graph

        self.loss_weight_dict = loss_weight_dict
        self.init_gnn_iters = init_gnn_iters
        self.Pretraining = Pretraining
        
        self.gnn_iters = gnn_iters
        self.seg_iters = seg_iters
        self.max_iters = max_iters
        self.first_stage_gnn_iters = first_stage_gnn_iters
        self.sec_stage_gnn_iters = gnn_iters - first_stage_gnn_iters
        
        self.criterion = OhemCELoss(ohem_thresh, ignore_lb)
        self.MdsOhemLoss = MdsOhemCELoss(self.n_datasets, ohem_thresh, ignore_lb)
        self.celoss = nn.CrossEntropyLoss(ignore_index=ignore_lb)

    @classmethod
    def from_config(cls, cfg):
        
        # aff_cfg = cfg.MODEL.AFFORMER_CONFIG
        # module = import_module_from_path(aff_cfg)
        backbone = build_backbone(cfg)
        graph_node_features = gen_graph_node_feature(cfg)
        init_gnn_iters = cfg.MODEL.GNN.init_stage_iters
        Pretraining = cfg.MODEL.PRETRAINING
        # if Pretraining:
        #     gnn_model = None
        # else:
        gnn_model = build_GNN_module(cfg)
            
        gnn_iters = cfg.MODEL.GNN.GNN_ITERS
        seg_iters = cfg.MODEL.GNN.SEG_ITERS
        max_iters = cfg.SOLVER.MAX_ITER
        first_stage_gnn_iters = cfg.MODEL.GNN.FIRST_STAGE_GNN_ITERS
        with_spa_loss = cfg.LOSS.WITH_SPA_LOSS
        with_orth_loss = cfg.LOSS.WITH_ORTH_LOSS  
        with_adj_loss = cfg.LOSS.WITH_ADJ_LOSS 
        with_relation_loss = cfg.LOSS.WITH_RELATION_LOSS 
        with_gaussian_loss = cfg.LOSS.WITH_GAUSSIAN_LOSS
        relation_gt_graph =None
        if with_relation_loss:
            assert cfg.DATASETS.RELATION_GRAPH is not None, "relation graph is None"
            with open(cfg.DATASETS.RELATION_GRAPH, "rb") as file:
                relation_gt_graph = pickle.load(file)

        loss_weight_dict = {"loss_ce0": 1, "loss_ce1": 1, "loss_ce2": 1, "loss_ce3": 1, "loss_ce4": 1, "loss_ce5": 1, "loss_ce6": 1, "loss_aux0": 1, "loss_aux1": 3, "loss_aux2": 1, "loss_aux3": 1, "loss_aux4": 1, "loss_aux5": 3, "loss_aux6": 1, "loss_spa": 0.001, "loss_adj":1, "loss_orth":10, "loss_relation": 1}
        ignore_lb = cfg.DATASETS.IGNORE_LB
        ohem_thresh = cfg.LOSS.OHEM_THRESH
 
        return {
            'seg_model': backbone,
            'gnn_model': gnn_model,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            'datasets_cats': cfg.DATASETS.DATASETS_CATS,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "num_unify_classes": cfg.DATASETS.NUM_UNIFY_CLASS,
            "graph_node_features": graph_node_features,
            "init_gnn_iters": init_gnn_iters,
            "Pretraining": Pretraining,
            "gnn_iters": gnn_iters,
            "seg_iters": seg_iters,
            "max_iters": max_iters,
            "first_stage_gnn_iters": first_stage_gnn_iters,
            "with_spa_loss": with_spa_loss,
            "with_orth_loss": with_orth_loss,
            "with_adj_loss": with_adj_loss,
            "with_relation_loss": with_relation_loss,
            "with_gaussian_loss": with_gaussian_loss,
            "relation_gt_graph": relation_gt_graph,
            "loss_weight_dict": loss_weight_dict,     
            'ignore_lb': ignore_lb,
            'ohem_thresh': ohem_thresh,       
        }

    def forward(self, batched_inputs):
        
        images = [x["image"].cuda() for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # if self.training:
        # images = ImageList.from_tensors(images, 4)#self.size_divisibility)
        # else:
        images = ImageList.from_tensors(images, -1)

        if self.training:
            dataset_lbs = [x["dataset_id"] for x in batched_inputs]
            dataset_lbs = torch.tensor(dataset_lbs).long().cuda()
            targets = [x["sem_seg"].cuda() for x in batched_inputs]
            targets = self.prepare_targets(targets, images)
            targets = torch.cat(targets, dim=0)
        else:
            # if "dataset_id" not in batched_inputs[0]:
            #     dataset_lbs = 0
            # else:
            try:
                dataset_lbs = int(batched_inputs[0]["dataset_id"])
            except:
                dataset_lbs = 0

        if self.Pretraining:

            if self.training:
                outputs = self.seg_model(images.tensor, None, None, dataset_lbs, False)
                losses = self.clac_pretrain_loss(batched_inputs, images, targets, dataset_lbs, outputs)
                # losses = self.MdsOhemLoss(outputs['logits'], targets, dataset_lbs)
                        
                for k in list(losses.keys()):
                    if k in self.loss_weight_dict:
                        losses[k] *= self.loss_weight_dict[k]
                    # else:
                    #     # remove this loss if not specified in `weight_dict`
                    #     losses.pop(k)
                return losses
        
            else:
                outputs = self.seg_model(images.tensor, None, None, dataset_lbs, False)
                
                processed_results = []
                for logit, input_per_image, image_size in zip(outputs['logits'], batched_inputs, images.image_sizes):
                # for logit, input_per_image, image_size in zip(outputs['logits'], batched_inputs, images.image_sizes):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    # logit = F.interpolate(logit, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                                        
                    logit = retry_if_cuda_oom(sem_seg_postprocess)(logit, image_size, height, width)
                    # uni_logit = F.interpolate(uni_logit, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)

                    processed_results.append({"sem_seg": logit})
                return processed_results
        else:
            if self.training:
                self.env_init(self.iters)   
                outputs = self.seg_model(images.tensor, None, None, False)
                                
                unify_prototype, bi_graphs, _, _ = self.gnn_model(self.graph_node_features)
                if self.train_seg_or_gnn == self.SEG:
                    processed_results = []
                    for logit, input_per_image, image_size in zip(outputs['logits'], batched_inputs, images.image_sizes):
                        height = input_per_image.get("height", image_size[0])
                        width = input_per_image.get("width", image_size[1])
                        logit = F.interpolate(logit, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                        logit = retry_if_cuda_oom(sem_seg_postprocess)(logit, image_size, height, width)
                        # logger.info(f"logit shape:{logit.shape}")
                        processed_results.append({"sem_seg": logit})
                else:
                    # logger.info(f"{len(bi_graphs)}")
                    if self.with_datasets_aux:
                        ori_logits = torch.einsum('bchw, nc -> bnhw', outputs['emb'], unify_prototype[self.total_cats:])
                    else:
                        ori_logits = torch.einsum('bchw, nc -> bnhw', outputs['emb'], unify_prototype)
                    if len(bi_graphs) == 2*self.n_datasets:
                        logits = torch.einsum('bchw, nc -> bnhw', ori_logits, bi_graphs[2*dataset_lbs+1])
                    else:
                        logits = torch.einsum('bchw, nc -> bnhw', ori_logits, bi_graphs[dataset_lbs])
                    processed_results = []
                    for input_per_image, image_size in zip(batched_inputs, images.image_sizes):
                        height = input_per_image.get("height", image_size[0])
                        width = input_per_image.get("width", image_size[1])
                        logits = F.interpolate(logits, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                        
                        logits = retry_if_cuda_oom(sem_seg_postprocess)(logits, image_size, height, width)
                        # logger.info(f"logit shape:{logit.shape}")
                        processed_results.append({"sem_seg": logits, "uni_logits": ori_logits})
                    
                return processed_results
            


    def clac_pretrain_loss(self, batched_inputs, images, targets, dataset_lbs, outputs):
        losses = {}
        for idx, logit in enumerate(outputs['logits']):
            
            logits = F.interpolate(logit, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
            # logger.info(f"logits:{logits.shape}, target:{targets[dataset_lbs==idx].shape}")
            loss = self.criterion(logits, targets[dataset_lbs==idx])
                    
            # if torch.isnan(loss):
            #     logger.info(f"file_name:{batched_inputs[2*idx]['file_name']}, {torch.min(targets[dataset_lbs==idx])}")
                        
            #     continue
            losses[f'loss_ce{idx}'] = loss
        return losses

    def calc_loss(self, images, targets, dataset_lbs, outputs, unify_prototype, bi_graphs, batched_inputs, adj_matrix):
        losses = {}
        if self.train_seg_or_gnn == self.GNN:
            if self.with_datasets_aux:
                logits = torch.einsum('bchw, nc -> bnhw', outputs['emb'], unify_prototype[self.total_cats:])
            else:
                logits = torch.einsum('bchw, nc -> bnhw', outputs['emb'], unify_prototype)
            
            if self.with_gaussian_loss:
                losses['loss_gaussian'] = self.gaussian_loss(outputs['emb'], logits, targets, True, dataset_lbs, False, adj_matrix)
                
                                    
        else:
            remap_logits = outputs['logits']
            if self.with_datasets_aux:
                aux_logits_out = outputs['aux_logits']
            
            if self.with_gaussian_loss:
                losses['loss_gaussian'] = self.gaussian_loss(outputs['emb'], outputs['uni_logits'][0], targets, True, dataset_lbs, True, adj_matrix)
                    
                # remap_logits = []
        uot_rate = np.min([int(self.alter_iters) / self.first_stage_gnn_iters, 1])
        adj_rate = 1 - uot_rate
        cur_cat = 0
        for i in range(self.n_datasets):
            cur_cat += self.datasets_cats[i]
                    
            if not (dataset_lbs == i).any():
                continue
                    
            if self.train_seg_or_gnn == self.GNN:
                if len(bi_graphs) == 2*self.n_datasets:
                    remap_logits_1 = torch.einsum('bchw, nc -> bnhw', logits[dataset_lbs==i], bi_graphs[2*i])
                    remap_logits_2 = torch.einsum('bchw, nc -> bnhw', logits[dataset_lbs==i], bi_graphs[2*i+1])
                        
                    remap_logits_1 = F.interpolate(remap_logits_1, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                    loss_1 = self.criterion(remap_logits_1, targets[dataset_lbs==i])
                            
                    remap_logits_2 = F.interpolate(remap_logits_2, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                    loss_2 = self.criterion(remap_logits_2, targets[dataset_lbs==i])
                    loss = uot_rate*loss_1 + adj_rate*loss_2
                    if torch.isnan(loss):
                        logger.info(f"file_name:{batched_inputs[2*i]['file_name']}, {torch.min(targets[dataset_lbs==i])}")
                    else:
                        losses[f'loss_ce{i}'] = loss
                else:
                    remap_logits = torch.einsum('bchw, nc -> bnhw', logits[dataset_lbs==i], bi_graphs[i])
                        
                    remap_logits = F.interpolate(remap_logits, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                    loss = self.criterion(remap_logits, targets[dataset_lbs==i])
                    if torch.isnan(loss):
                        logger.info(f"file_name:{batched_inputs[2*i]['file_name']}, {torch.min(targets[dataset_lbs==i])}")
                    else:
                        losses[f'loss_ce{i}'] = loss
            else:
                remap_logits[i] = F.interpolate(remap_logits[i], size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                loss = self.criterion(remap_logits[i], targets[dataset_lbs==i])
                if torch.isnan(loss):
                    logger.info(f"file_name:{batched_inputs[2*i]['file_name']}, {torch.min(targets[dataset_lbs==i])}")
                else:
                    losses[f'loss_ce{i}'] = loss                   
            

            if self.with_datasets_aux:
                if self.train_seg_or_gnn == self.GNN:
                    aux_logits = torch.einsum('bchw, nc -> bnhw', outputs['emb'][dataset_lbs==i], unify_prototype[cur_cat-self.datasets_cats[i]:cur_cat])
                else:
                    aux_logits = aux_logits_out[i]
                        
                aux_logits = F.interpolate(aux_logits, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                aux_loss = self.criterion(aux_logits, targets[dataset_lbs==i])
                if torch.isnan(aux_loss):
                    logger.info(f"file_name:{batched_inputs[2*i]['file_name']}, {torch.min(targets[dataset_lbs==i])}")
                else:
                    losses[f'loss_aux{i}'] = aux_loss
                    

                
            if self.with_spa_loss and self.train_seg_or_gnn == self.GNN and self.inFirstGNNStage and self.iters > self.init_gnn_iters:
                if len(bi_graphs)==2*self.n_datasets:
                    spa_loss = torch.pow(torch.norm(bi_graphs[2*i+1], p='fro'), 2)
                else:
                    spa_loss =  torch.pow(torch.norm(bi_graphs[i], p='fro'), 2)
                        
                losses['loss_spa'] = spa_loss
                        
            if self.with_adj_loss and self.train_seg_or_gnn == self.GNN and self.inFirstGNNStage and self.iters > self.init_gnn_iters and self.target_bipart is not None:
                if len(bi_graphs) == 2*self.n_datasets:
                    total_num = bi_graphs[2*i+1].shape[0] * bi_graphs[2*i+1].shape[1]
                    base_weight = 1 / total_num
                            
                    if 'loss_adj' not in losses:
                        losses['loss_adj'] = base_weight * self.MSE_sum_loss(bi_graphs[2*i + 1][self.target_bipart[i] != 255], self.target_bipart[i][self.target_bipart[i] != 255])
                    else:
                        losses['loss_adj'] += base_weight * self.MSE_sum_loss(bi_graphs[2*i + 1][self.target_bipart[i] != 255], self.target_bipart[i][self.target_bipart[i] != 255])
                else:
                    total_num = bi_graphs[i].shape[0] * bi_graphs[i].shape[1]
                    base_weight = 1 / total_num
                            
                    if losses['loss_adj'] is None:
                        losses['loss_adj'] = base_weight * self.MSE_sum_loss(bi_graphs[2*i + 1][self.target_bipart[i] != 255], self.target_bipart[i][self.target_bipart[i] != 255])
                    else:
                        losses['loss_adj'] += base_weight * self.MSE_sum_loss(bi_graphs[2*i + 1][self.target_bipart[i] != 255], self.target_bipart[i][self.target_bipart[i] != 255])
                
        if self.with_relation_loss and self.train_seg_or_gnn == self.GNN:
            decay_weight = 1 - self.iters / self.max_iters
            relation_base_weight = decay_weight / (self.n_datasets * (self.n_datasets - 1) / 2)
            losses['loss_relation'] = relation_base_weight * relation_loss(adj_matrix, self.datasets_cats, self.relation_gt_graph)
               

        if self.with_orth_loss and self.train_seg_or_gnn == self.GNN:
            if self.with_datasets_aux:
                losses['loss_orth'] = self.similarity_dsb(unify_prototype[self.total_cats:])
            else:
                losses['loss_orth'] = self.similarity_dsb(unify_prototype)               
        return losses
    
    def get_unify_prototype(self):
        return self.proj_head.unify_prototype

    def set_dataset_adapter(self, dataset_adapter):
        self.dataset_adapter = dataset_adapter

    def set_dataset_adapter(self, dataset_adapter):
        self.dataset_adapter = dataset_adapter

    def env_init(self, iters):
        if self.initial == False:
            # self.alter_iters = torch.zeros(1)
            logger.info(f"initial: train_seg_or_gnn: {self.train_seg_or_gnn}, alter_iter:{self.alter_iters}")
            if self.train_seg_or_gnn == self.GNN:
                self.backbone.req_grad(False)
                self.proj_head.req_grad(False)
                self.gnn_model.req_grad(True)
                self.backbone.eval()
                self.proj_head.eval()
                self.gnn_model.train()
                if iters <= self.init_gnn_iters:
                    logger.info(f"init gnn stage")
                    self.gnn_model.frozenAdj(True)
                    self.gnn_model.set_init_stage(True)
                    self.init_gnn_stage = True
                else:
                    logger.info(f"gnn stage")
                    self.init_gnn_stage = False
                    self.gnn_model.set_init_stage(False)
                    self.gnn_model.frozenAdj(False)
                    if iters <= self.first_stage_gnn_iters:
                        self.inFirstGNNStage = True
                
            else:
                self.backbone.req_grad(True)
                self.proj_head.req_grad(True)
                self.gnn_model.req_grad(False)
                self.backbone.train()
                self.proj_head.train()
                self.gnn_model.eval()                    
                self.gnn_model.set_init_stage(False)
                self.init_gnn_stage = False
                # unify_prototype, bi_graphs = self.gnn_model.get_optimal_matching(self.graph_node_features, True)
                # self.proj_head.set_bipartite_graphs(bi_graphs)
                # self.proj_head.set_unify_prototype(unify_prototype, grad=False)
            self.initial = True

        if self.train_seg_or_gnn == self.GNN:
            if self.init_gnn_stage and iters > self.init_gnn_iters:
                logger.info(f"finish init gnn stage")
                self.change_to_seg()
                self.init_gnn_stage = False
                unify_prototype, bi_graphs, _, _ = self.gnn_model(self.graph_node_features)
                self.proj_head.set_bipartite_graphs(bi_graphs)
                self.proj_head.set_unify_prototype(unify_prototype, grad=False)
                self.gnn_model.set_init_stage(False)
            elif self.inFirstGNNStage and int(self.alter_iters) > self.first_stage_gnn_iters:
                logger.info(f"change to second_gnn_stage")
                self.gnn_model.frozenAdj(True)
                self.inFirstGNNStage = False
            if int(self.alter_iters) > self.gnn_iters:
                self.gnn_model.set_init_stage(False)
                self.change_to_seg()
                # unify_prototype, bi_graphs, _, _ = self.gnn_model(self.graph_node_features)
                unify_prototype, bi_graphs = self.gnn_model.get_optimal_matching(self.graph_node_features, True)
                self.proj_head.set_bipartite_graphs(bi_graphs)
                self.proj_head.set_unify_prototype(unify_prototype, grad=False)
        else:
            if int(self.alter_iters) > self.seg_iters:
                self.change_to_gnn()
                    
                self.inFirstGNNStage = True


    def change_to_seg(self):
        logger.info(f"change to seg_stage")
        self.train_seg_or_gnn = self.SEG
        self.backbone.req_grad(True)
        self.proj_head.req_grad(True)
        self.gnn_model.req_grad(False)
        self.backbone.train()
        self.proj_head.train()
        self.gnn_model.eval() 
        self.gnn_model.set_init_stage(False)
        self.gnn_model.frozenAdj(False)
        self.alter_iters = torch.zeros(1)
        if self.with_gaussian_loss:
            var = self.gaussian_loss.update()

    def change_to_gnn(self):
        logger.info(f"change to gnn_stage")
        self.train_seg_or_gnn = self.GNN
        self.backbone.req_grad(False)
        self.proj_head.req_grad(False)
        self.gnn_model.req_grad(True)
        self.backbone.eval()
        self.proj_head.eval()
        self.gnn_model.train()
        self.alter_iters = torch.zeros(1)
        if self.with_gaussian_loss:
            var = self.gaussian_loss.update()                
                    

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            # logger.info(f"image shape : {images.tensor.shape}, target shape : {targets_per_image.shape}")
            gt_masks = targets_per_image
            padded_masks = 255*torch.ones((1, h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[0], : gt_masks.shape[1]] = gt_masks
            new_targets.append(
                padded_masks
            )
        return new_targets

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            # elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            #     if hasattr(module, 'last_bn') and module.last_bn:
            #         nn.init.zeros_(module.weight)
            #     else:
            #         nn.init.ones_(module.weight)
            #     nn.init.zeros_(module.bias)
        for name, param in self.named_parameters():
            if name.find('affine_weight') != -1:
                if hasattr(param, 'last_bn') and param.last_bn:
                    nn.init.zeros_(param)
                else:
                    nn.init.ones_(param)
            elif name.find('affine_bias') != -1:
                nn.init.zeros_(param)
                        
        # self.load_pretrain()

        
    def load_pretrain(self):
        state = torch.load(backbone_url)
        self.backbone.load_state_dict(state, strict=False)

    def get_params(self):
        def add_param_to_list(param, wd_params, nowd_params):
            # for param in mod.parameters():
            if param.requires_grad == False:
                return
                # continue
            
            if param.dim() == 1:
                nowd_params.append(param)
            elif param.dim() == 4 or param.dim() == 2:
                wd_params.append(param)
            else:
                nowd_params.append(param)
                print(param.dim())
                # print(param)
                print(name)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        # for name, child in self.named_children():
        for name, param in self.named_parameters():
            
            if 'head' in name or 'aux' in name:
                add_param_to_list(param, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(param, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params
    
    def set_bipartite_graphs(self, bi_graphs):
        
        if len(bi_graphs) == 2 * self.n_datasets:
            for i in range(0, self.n_datasets):
                self.bipartite_graphs[i] = nn.Parameter(
                    bi_graphs[2*i], requires_grad=False
                    )
        else:
            # print("bi_graphs len:", len(bi_graphs))
            for i in range(0, self.n_datasets):
                # print("i: ", i)
                self.bipartite_graphs[i] = nn.Parameter(
                    bi_graphs[i], requires_grad=False
                    )
            
        
    def set_unify_prototype(self, unify_prototype, grad=False):
        if self.with_datasets_aux and unify_prototype.shape[0] != self.unify_prototype.shape[0]:
            self.unify_prototype.data = unify_prototype[self.total_cats:]
            self.unify_prototype.requires_grad=grad
            cur_cat = 0
            for i in range(self.n_datasets):
                self.aux_prototype[i].data = unify_prototype[cur_cat:cur_cat+self.datasets_cats[i]]
                cur_cat += self.datasets_cats[i]
                self.aux_prototype[i].requires_grad=grad
        else:
            self.unify_prototype.data = unify_prototype
            self.unify_prototype.requires_grad=grad

    def get_bipart_graph(self):
        if self.Pretraining:
            bi_graphs = self.proj_head.bipartite_graphs
        else:
            _, ori_bi_graphs, _, _ = self.gnn_model(self.graph_node_features)
            bi_graphs = []
            if len(ori_bi_graphs) == 2*self.n_datasets:
                for j in range(0, len(ori_bi_graphs), 2):
                    bi_graphs.append(ori_bi_graphs[j+1].detach())
            else:
                bi_graphs = [bigh.detach() for bigh in ori_bi_graphs]

        return bi_graphs
        
    def get_encode_lb_vec(self):
        text_feature_vecs = []
        with torch.no_grad():
            clip_model, _ = clip.load("ViT-B/32", device="cuda")
            for i in range(0, self.n_datasets):
                lb_name = self.configer.get("dataset"+str(i+1), "label_names")
                lb_name = [f'a photo of {name} from dataset {i+1}.' for name in lb_name]
                text = clip.tokenize(lb_name).cuda()
                text_features = clip_model.encode_text(text).type(torch.float32)
                text_feature_vecs.append(text_features)
                
        text_feature_vecs = torch.cat(text_feature_vecs, dim=0)
        self.unify_prototype.data = text_feature_vecs
        self.unify_prototype.requires_grad=False
                
    def set_target_bipart(self, target_bipart):
        self.target_bipart = target_bipart
        # self.target_bipart.requires_grad=False
        
    def similarity_dsb(self, proto_vecs, reduce='mean'):
        """
        Compute EM loss with the probability-based distribution of each feature
        :param feat_domain: source, target or both
        :param temperature: softmax temperature
        """


        # dot similarity between features and centroids
        z = torch.mm(proto_vecs, proto_vecs.t())  # size N x C_seen
        z = z / self.temperature

        if reduce == 'mean':
            loss = -1 * torch.mean(F.softmax(z, dim=1) * F.log_softmax(z, dim=1))
        elif reduce == 'sum':
            loss = -1 * torch.sum(F.softmax(z, dim=1) * F.log_softmax(z, dim=1))
        if torch.isnan(loss).any():
            logger.info(f"got nan in loss")
            raise Exception(f"loss_orth nan, z:{torch.max(z)}, zmin:{torch.min(z)} unify_prototype:{torch.isnan(z).any()}, softmax:{torch.isnan(F.softmax(z, dim=1)).any()}, log_softmax:{torch.isnan(F.log_softmax(z, dim=1)).any()}")

        return loss
    
    def calc_match_loss(self, images, targets, dataset_lbs, outputs, unify_prototype, bi_graphs, batched_inputs):
        losses = {}
        if self.train_seg_or_gnn == self.GNN:
            if self.with_datasets_aux:
                logits = torch.einsum('bchw, nc -> bnhw', outputs['emb'], unify_prototype[self.total_cats:])
            else:
                logits = torch.einsum('bchw, nc -> bnhw', outputs['emb'], unify_prototype)
        else:
            remap_logits = outputs['logits']
            if self.with_datasets_aux:
                aux_logits_out = outputs['aux_logits']
                    
        
        # if self.with_adj_loss and self.train_seg_or_gnn == self.GNN and self.inFirstGNNStage and self.iters > self.init_gnn_iters:
        #     self.start_multi_thread_mcmf(logits, targets, bi_graphs, dataset_lbs)
            
                # remap_logits = []
        uot_rate = np.min([int(self.alter_iters) / self.first_stage_gnn_iters, 1])
        adj_rate = 1 - uot_rate
        cur_cat = 0
        for i in range(self.n_datasets):
            cur_cat += self.datasets_cats[i]
                    
            if not (dataset_lbs == i).any():
                continue
                    
            if self.train_seg_or_gnn == self.GNN:
                if len(bi_graphs) == 2*self.n_datasets:
                    remap_logits_1 = torch.einsum('bchw, nc -> bnhw', logits[dataset_lbs==i], bi_graphs[2*i])
                    remap_logits_2 = torch.einsum('bchw, nc -> bnhw', logits[dataset_lbs==i], bi_graphs[2*i+1])
                        
                    remap_logits_1 = F.interpolate(remap_logits_1, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                    loss_1 = self.criterion(remap_logits_1, targets[dataset_lbs==i])
                            
                    remap_logits_2 = F.interpolate(remap_logits_2, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                    loss_2 = self.criterion(remap_logits_2, targets[dataset_lbs==i])
                    loss = uot_rate*loss_1 + adj_rate*loss_2
                    if torch.isnan(loss):
                        logger.info(f"file_name:{batched_inputs[2*i]['file_name']}, {torch.min(targets[dataset_lbs==i])}")
                    else:
                        losses[f'loss_ce{i}'] = loss
                else:
                    remap_logits = torch.einsum('bchw, nc -> bnhw', logits[dataset_lbs==i], bi_graphs[i])
                        
                    remap_logits = F.interpolate(remap_logits, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                    loss = self.criterion(remap_logits, targets[dataset_lbs==i])
                            
                    if torch.isnan(loss):
                        logger.info(f"file_name:{batched_inputs[2*i]['file_name']}, {torch.min(targets[dataset_lbs==i])}")
                    else:
                        losses[f'loss_ce{i}'] = loss
            else:
                remap_logits[i] = F.interpolate(remap_logits[i], size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                loss = self.criterion(remap_logits[i], targets[dataset_lbs==i])
                        
                if torch.isnan(loss):
                    logger.info(f"file_name:{batched_inputs[2*i]['file_name']}, {torch.min(targets[dataset_lbs==i])}")
                else:
                    losses[f'loss_ce{i}'] = loss                    
        

            if self.with_datasets_aux:
                if self.train_seg_or_gnn == self.GNN:
                    aux_logits = torch.einsum('bchw, nc -> bnhw', outputs['emb'][dataset_lbs==i], unify_prototype[cur_cat-self.datasets_cats[i]:cur_cat])
                else:
                    aux_logits = aux_logits_out[i]
                        
                aux_logits = F.interpolate(aux_logits, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                aux_loss = self.criterion(aux_logits, targets[dataset_lbs==i])
                if torch.isnan(aux_loss):
                    logger.info(f"file_name:{batched_inputs[2*i]['file_name']}, {torch.min(targets[dataset_lbs==i])}")
                    
                else:
                    losses[f'loss_aux{i}'] = aux_loss
                    

                
            if self.with_spa_loss and self.train_seg_or_gnn == self.GNN and self.inFirstGNNStage and self.iters > self.init_gnn_iters:
                if len(bi_graphs)==2*self.n_datasets:
                    spa_loss = torch.pow(torch.norm(bi_graphs[2*i+1], p='fro'), 2)
                else:
                    spa_loss =  torch.pow(torch.norm(bi_graphs[i], p='fro'), 2)
                        
                losses['loss_spa'] = spa_loss

        if self.with_orth_loss and self.train_seg_or_gnn == self.GNN:
            if self.with_datasets_aux:
                losses['loss_orth'] = self.similarity_dsb(unify_prototype[self.total_cats:])
            else:
                losses['loss_orth'] = self.similarity_dsb(unify_prototype)               
        
        if self.with_adj_loss and self.train_seg_or_gnn == self.GNN and self.inFirstGNNStage and self.iters > self.init_gnn_iters:
            self.clac_mcmf(logits, targets, bi_graphs, dataset_lbs, losses)
            # supervise_bi = self.get_multi_thread_mcmf()
            # cur_idx = 0
            # for i in range(self.n_datasets):
            #     if not (dataset_lbs == i).any():
            #         continue
            #     if len(bi_graphs) == 2*self.n_datasets:
            #         if 'loss_adj' not in losses:
            #             losses['loss_adj'] = self.celoss(bi_graphs[2*i+1].T, supervise_bi[cur_idx])
            #         else:
            #             losses['loss_adj'] += self.celoss(bi_graphs[2*i+1].T, supervise_bi[cur_idx])
            #     else:
            #         if 'loss_adj' not in losses:
            #             losses['loss_adj'] = self.celoss(bi_graphs[i].T, supervise_bi[cur_idx])
            #         else:
            #             losses['loss_adj'] += self.celoss(bi_graphs[i].T, supervise_bi[cur_idx])
                    
            #     cur_idx += 1
    
        return losses
        
    def start_multi_thread_mcmf(self, unify_logits, target, bipart, dataset_lbs):
        self.threads = []
        for i in range(self.n_datasets):
            if not (dataset_lbs == i).any():
                continue
            if len(bipart) == 2*self.n_datasets:
                self.threads.append(MCMFThread(self.datasets_cats[i], unify_logits[dataset_lbs == i], target[dataset_lbs == i], bipart[2*i+1]))
            else:
                self.threads.append(MCMFThread(self.datasets_cats[i], unify_logits[dataset_lbs == i], target[dataset_lbs == i], bipart[i]))
                

        # 启动线程
        for thread in self.threads:
            thread.start()

    def clac_mcmf(self, unify_logits, target, bipart, dataset_lbs, losses):
        for i in range(self.n_datasets):
            if not (dataset_lbs == i).any():
                continue
            uni_classes = unify_logits.shape[1]
            mcmf = MinCostMaxFlow_Or(self.datasets_cats[i], uni_classes, n_points=self.n_points, ignore_lb=self.ignore_lb)
            if len(bipart) == 2*self.n_datasets:
                supervise_bi = mcmf(unify_logits[dataset_lbs == i], target[dataset_lbs == i], bipart[2*i+1]).to(unify_logits.device).long()
            else:
                supervise_bi = mcmf(unify_logits[dataset_lbs == i], target[dataset_lbs == i], bipart[i]).to(unify_logits.device).long()
            
            
                
            if len(bipart) == 2*self.n_datasets:
                if 'loss_adj' not in losses:
                    loss = self.celoss(bipart[2*i+1].T, supervise_bi)
                    if not torch.isnan(loss):
                        losses['loss_adj'] = loss
                else:
                    loss = self.celoss(bipart[2*i+1].T, supervise_bi)
                    if not torch.isnan(loss):
                        losses['loss_adj'] += loss
            else:
                if 'loss_adj' not in losses:
                    loss = self.celoss(bipart[i].T, supervise_bi)
                    if not torch.isnan(loss):
                        losses['loss_adj'] = loss
                else:
                    loss = self.celoss(bipart[i].T, supervise_bi)
                    if not torch.isnan(loss):
                        losses['loss_adj'] += loss
                

    def get_multi_thread_mcmf(self):
        for thread in self.threads:
            thread.join()
        
        rets = []
        for thread in self.threads:
            rets.append(thread.ret)
        
        # logger.info(f"rets{rets}")

        return rets
