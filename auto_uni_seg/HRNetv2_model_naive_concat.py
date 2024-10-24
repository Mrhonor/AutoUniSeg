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
from .modeling.GNN.gen_graph_node_feature import gen_graph_node_feature
from .modeling.GNN.ltbgnn_llama import build_GNN_module
from .modeling.backbone.hrnet_backbone import HighResolutionNet
from .modeling.loss.ohem_ce_loss import OhemCELoss
from timm.models.layers import trunc_normal_
import clip
import logging
from detectron2.utils.events import get_event_storage, EventStorage
import numpy as np
import torch.utils.model_zoo as model_zoo

logger = logging.getLogger(__name__)

@META_ARCH_REGISTRY.register()
class HRNet_W48_Naive_Concat_ARCH(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """
    @configurable
    def __init__(self, *, 
                backbone,
                gnn_model,
                sem_seg_head,
                datasets_cats,
                with_datasets_aux,
                ignore_lb,
                ohem_thresh,
                size_divisibility,
                pixel_mean,
                pixel_std,
                graph_node_features,
                finetune_stage1_iters,
                num_unify_classes,
                with_spa_loss,
                loss_weight_dict,
                with_orth_loss,
                with_adj_loss
                ):
        super(HRNet_W48_Naive_Concat_ARCH, self).__init__()
        self.num_unify_classes = num_unify_classes

        self.datasets_cats = datasets_cats
        self.n_datasets = len(self.datasets_cats)
        self.backbone = backbone
        self.gnn_model = None
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        
        self.register_buffer("finetune_stage", torch.ones(1), True)
        self.register_buffer("proto_init", torch.zeros(1), True)

        # self.register_buffer("target_bipart", torch.ParameterList([]), False)

        
        #self.GNN = torch.ones(1)
        #self.SEG = torch.zeros(1)
        self.finetune_stage1_iters = finetune_stage1_iters
        self.with_datasets_aux = with_datasets_aux
        self.proj_head = sem_seg_head # ProjectionHead(dim_in=in_channels, proj_dim=self.output_feat_dim, bn_type=bn_type)
        self.graph_node_features = graph_node_features.cuda()
        self.iters = 0
        self.total_cats = 0
        self.acc_cats = []
        # self.datasets_cats = []
        self.dataset_adapter = []
        for i in range(0, self.n_datasets):
            # self.datasets_cats.append(self.configer.get('dataset'+str(i+1), 'n_cats'))
            self.acc_cats.append(self.total_cats)
            self.total_cats += self.datasets_cats[i]
            self.dataset_adapter.append(None)
            
        
        self.ignore_lb = self.total_cats
 
        self.criterion = OhemCELoss(ohem_thresh, self.ignore_lb)
        
        self.initial = False
        self.inFirstGNNStage = True
        self.temperature = 0.07
 
        #  if self.MODEL_WEIGHTS != None:
        # state = torch.load('output/pretrain_model_30000.pth')
        # self.load_state_dict(state['model_state_dict'], strict=True)
        self.isLoad = False
        self.with_spa_loss = with_spa_loss
        self.with_orth_loss = with_orth_loss
        self.with_adj_loss = with_adj_loss

        self.loss_weight_dict = loss_weight_dict
        self.MSE_sum_loss = torch.nn.MSELoss(reduction='sum')
        self.init_gnn_stage = False
        self.target_bipart = None
        # self.backbone.load_state_dict( model_zoo.load_url("https://download.pytorch.org/models/resnet18-5c106cde.pth"), strict=False)

        # self.get_encode_lb_vec()

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, 720)
        # sem_seg_head = build_sem_seg_head(cfg, backbone.num_features)
        gnn_model = None
        datasets_cats = cfg.DATASETS.DATASETS_CATS
        ignore_lb = cfg.DATASETS.IGNORE_LB
        ohem_thresh = cfg.LOSS.OHEM_THRESH

        with_datasets_aux = cfg.MODEL.GNN.with_datasets_aux
        graph_node_features = gen_graph_node_feature(cfg)
        num_unify_classes = cfg.DATASETS.NUM_UNIFY_CLASS
        finetune_stage1_iters = cfg.MODEL.GNN.FINETUNE_STAGE1_ITERS
        with_spa_loss = cfg.LOSS.WITH_SPA_LOSS
        with_orth_loss = cfg.LOSS.WITH_ORTH_LOSS  
        with_adj_loss = cfg.LOSS.WITH_ADJ_LOSS 
        loss_weight_dict = {"loss_ce0": 1, "loss_ce1": 3, "loss_ce2": 1, "loss_ce3": 1, "loss_ce4": 1, "loss_ce5": 3, "loss_ce6": 3, "loss_aux0": 1, "loss_aux1": 3, "loss_aux2": 1, "loss_aux3": 1, "loss_aux4": 1, "loss_aux5": 3, "loss_aux6": 2, "loss_spa": 0.001, "loss_adj":1, "loss_orth":10}
        # loss_weight_dict = {"loss_ce0": 1, "loss_ce1": 2, "loss_ce2": 1, "loss_ce3": 1, "loss_ce4": 3, "loss_ce5": 3, "loss_ce6": 2, "loss_aux0": 1, "loss_aux1": 3, "loss_aux2": 1, "loss_aux3": 1, "loss_aux4": 1, "loss_aux5": 3, "loss_aux6": 2, "loss_spa": 0.001, "loss_adj":1, "loss_orth":10}
        
        return {
            'backbone': backbone,
            'sem_seg_head': sem_seg_head,
            'gnn_model': gnn_model,
            'datasets_cats': datasets_cats,
            'with_datasets_aux': with_datasets_aux, 
            'ignore_lb': ignore_lb,
            'ohem_thresh': ohem_thresh,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "graph_node_features": graph_node_features,
            "finetune_stage1_iters": finetune_stage1_iters,
            "num_unify_classes": num_unify_classes,
            "with_spa_loss": with_spa_loss,
            "with_orth_loss": with_orth_loss,
            "with_adj_loss": with_adj_loss,
            "loss_weight_dict": loss_weight_dict
        }


    def forward(self, batched_inputs, dataset=0):

        images = [x["image"].cuda() for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # if self.training:
        # images = ImageList.from_tensors(images, self.size_divisibility)
        # else:
        images = ImageList.from_tensors(images, -1)

        if self.training:
            targets = [x["sem_seg"].cuda() for x in batched_inputs]
            targets = self.prepare_targets(targets, images)
            targets = torch.cat(targets, dim=0)
            dataset_lbs = [x["dataset_id"] for x in batched_inputs]
            dataset_lbs = torch.tensor(dataset_lbs).long().cuda()
        else:
            if "dataset_id" in batched_inputs[0]: 
                dataset_lbs = int(batched_inputs[0]["dataset_id"])
            else:
                dataset_lbs = dataset
            # dataset_lbs = 6

        features = self.backbone(images.tensor)
        outputs = self.proj_head(features, dataset_lbs)
        
        if self.training:
                        # bipartite matching-based loss
            remap_logits = outputs['logits']

            losses = {}
            # for id, logit in enumerate(remap_logits):
            for idx in range(self.n_datasets):
                if not (dataset_lbs == idx).any():
                    continue
                logits = F.interpolate(remap_logits[dataset_lbs==idx], size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                this_target = targets[dataset_lbs==idx]
                add_target = this_target + self.acc_cats[idx]
                add_target[this_target==255] = self.ignore_lb
                loss = self.criterion(logits, add_target)

                if torch.isnan(loss):
                    continue
                losses[f'loss_ce{idx}'] = loss
            
            for k in list(losses.keys()):
                if k in self.loss_weight_dict:
                    losses[k] *= self.loss_weight_dict[k]
            #     else:
            #         # remove this loss if not specified in `weight_dict`
            #         losses.pop(k)
            return losses
        else:
            processed_results = []

            logit = outputs['logits']
            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            # dataset_lbs = 0
            # logit = F.softmax(logit, dim=1)
            # max_logits, preds = torch.max(logit, dim=1, keepdim=True)
            # # preds = torch.argmax(uni_logits, dim=0, keepdim=True).long()
            # preds = preds.long()
            # preds = preds - self.acc_cats[dataset_lbs]
            # preds[preds > self.datasets_cats[dataset_lbs]] = self.datasets_cats[dataset_lbs]
            # preds[preds < 0] = self.datasets_cats[dataset_lbs]
            # output = torch.zeros(preds.shape[0], self.datasets_cats[dataset_lbs]+1, preds.shape[2], preds.shape[3]).cuda()
            
            # output.scatter_(1, preds, 1)
            # logit = output
            # print(self.acc_cats[dataset_lbs])
            logit = logit[:, self.acc_cats[dataset_lbs]:self.acc_cats[dataset_lbs]+self.datasets_cats[dataset_lbs], :, :]
            logit = F.interpolate(logit, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="nearest")

            logit = retry_if_cuda_oom(sem_seg_postprocess)(logit, image_size, height, width)                
            # logit = F.softmax(logit, dim=0)
            # max_logits, preds = torch.max(logit, dim=0, keepdim=True)
            # # preds = torch.argmax(uni_logits, dim=0, keepdim=True).long()
            # preds = preds.long()
            # output = torch.zeros(26, preds.shape[1], preds.shape[2]).cuda()
            # preds[max_logits < 0.3] = 25
            # output.scatter_(0, preds, 1)
            # logit = output
            # logger.info(f"logit shape:{logit.shape}")
            processed_results.append({"sem_seg": logit})
            return processed_results             


                    

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
        self.proh_head.set_bipartite_graphs(bi_graphs)

    def set_dataset_adapter(self, dataset_adapter):
        self.dataset_adapter = dataset_adapter
        
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
        return self.proj_head.bipartite_graphs

    def set_bipartite_graphs(self, bigraph):
        self.proj_head.set_bipartite_graphs(bigraph)
        
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

        # entropy loss to push each feature to be similar to only one class prototype (no supervision)
        if reduce == 'mean':
            loss = -1 * torch.mean(F.softmax(z / self.temperature, dim=1) * F.log_softmax(z / self.temperature, dim=1))
        elif reduce == 'sum':
            loss = -1 * torch.sum(F.softmax(z / self.temperature, dim=1) * F.log_softmax(z / self.temperature, dim=1))
            

        return loss