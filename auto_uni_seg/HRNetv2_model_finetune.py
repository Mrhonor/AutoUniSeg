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
class HRNet_W48_Finetune_ARCH(nn.Module):
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
                with_adj_loss,
                specific_dataset_id
                ):
        super(HRNet_W48_Finetune_ARCH, self).__init__()
        self.num_unify_classes = num_unify_classes

        self.datasets_cats = datasets_cats
        self.n_datasets = len(self.datasets_cats)
        self.backbone = backbone
        self.gnn_model = gnn_model
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        
        self.register_buffer("finetune_stage", torch.ones(1), True)
        self.register_buffer("proto_init", torch.zeros(1), True)

        # self.register_buffer("target_bipart", torch.ParameterList([]), False)

        self.finetune_stage1_iters = finetune_stage1_iters
        self.with_datasets_aux = with_datasets_aux
        self.proj_head = sem_seg_head # ProjectionHead(dim_in=in_channels, proj_dim=self.output_feat_dim, bn_type=bn_type)
        self.proj_head.req_grad(True)
        self.graph_node_features = graph_node_features.cuda()
        self.iters = 0
        self.total_cats = 0
        # self.datasets_cats = []
        self.dataset_adapter = []
        for i in range(0, self.n_datasets):
            # self.datasets_cats.append(self.configer.get('dataset'+str(i+1), 'n_cats'))
            self.total_cats += self.datasets_cats[i]
            self.dataset_adapter.append(None)
 
        self.criterion = OhemCELoss(ohem_thresh, ignore_lb)
        
        self.specific_dataset_id = int(specific_dataset_id)
        
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
        gnn_model = build_GNN_module(cfg)
        datasets_cats = cfg.DATASETS.DATASETS_CATS
        ignore_lb = cfg.DATASETS.IGNORE_LB
        ohem_thresh = cfg.LOSS.OHEM_THRESH
        specific_dataset_id = cfg.DATASETS.SPECIFIC_DATASET_ID

        with_datasets_aux = cfg.MODEL.GNN.with_datasets_aux
        graph_node_features = gen_graph_node_feature(cfg)
        init_adj_path = cfg.MODEL.GNN.INIT_ADJ_PATH
        if init_adj_path != None:
            init_adj = torch.load(init_adj_path)
            num_unify_class = init_adj.shape[1]
        else:
            num_unify_class = cfg.DATASETS.NUM_UNIFY_CLASS
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
            "num_unify_classes": num_unify_class,
            "with_spa_loss": with_spa_loss,
            "with_orth_loss": with_orth_loss,
            "with_adj_loss": with_adj_loss,
            "loss_weight_dict": loss_weight_dict,
            "specific_dataset_id": specific_dataset_id
        }


    def forward(self, batched_inputs, dataset=0):
        if self.training:
            self.env_init()
        
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
            targets = [x["sem_seg"].cuda() for x in batched_inputs]
            targets = self.prepare_targets(targets, images)
            targets = torch.cat(targets, dim=0)
        else:
            if "dataset_id" in batched_inputs[0]: 
                dataset_lbs = int(batched_inputs[0]["dataset_id"])
            else:
                dataset_lbs = dataset
            if self.specific_dataset_id >= 0:
                dataset_lbs = self.specific_dataset_id


        

        features = self.backbone(images.tensor)
        outputs = self.proj_head(features, dataset_lbs)
        
        if self.training:
                        # bipartite matching-based loss
            remap_logits = outputs['logits']
            if self.with_datasets_aux:
                aux_logits_out = outputs['aux_logits']
            losses = {}
            for id, logit in enumerate(remap_logits):
                logits = F.interpolate(logit, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                loss = self.criterion(logits, targets[dataset_lbs==id])
                # logger.info(f"loss:{loss}")
                if torch.isnan(loss):
                    continue
                losses[f'loss_ce{id}'] = loss
            
            if self.with_datasets_aux:
                for idx, aux_logits in enumerate(aux_logits_out):
                
                    aux_logits = F.interpolate(aux_logits, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                    aux_loss = self.criterion(aux_logits, targets[dataset_lbs==idx])
                    if torch.isnan(aux_loss):
                        continue
                    losses[f'loss_aux{idx}'] = aux_loss
            for k in list(losses.keys()):
                if k in self.loss_weight_dict:
                    losses[k] *= self.loss_weight_dict[k]
            #     else:
            #         # remove this loss if not specified in `weight_dict`
            #         losses.pop(k)
            return losses
        else:
            processed_results = []
            for logit, input_per_image, image_size, uni_logits in zip(outputs['logits'], batched_inputs, images.image_sizes, outputs['uni_logits']):
                
                # height = images.tensor.shape[2]
                # width = images.tensor.shape[3]
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                
                
                # logit = F.interpolate(logit, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                logit = retry_if_cuda_oom(sem_seg_postprocess)(logit, image_size, height, width)
                uni_logits = retry_if_cuda_oom(sem_seg_postprocess)(uni_logits, image_size, height, width)

                processed_results.append({"sem_seg": logit, "uni_logits": uni_logits})
            return processed_results                      

    def env_init(self):
        if self.initial == False:
            logger.info(f"initial: finetune_stage: {self.finetune_stage}")
            self.backbone.req_grad(True)
            self.proj_head.req_grad(True)
            self.gnn_model.req_grad(False)
            self.backbone.train()
            self.proj_head.train()
            self.gnn_model.eval()
            self.initial = True

        if int(self.proto_init) == 0:
            logger.info(f"initial: finetune_stage: {self.finetune_stage}")
            self.gnn_model.set_init_stage(False)
            unify_prototype, bi_graphs = self.gnn_model.get_optimal_matching(self.graph_node_features, True)
            self.proj_head.set_bipartite_graphs(bi_graphs)
            self.proj_head.set_unify_prototype(unify_prototype.detach().float(), grad=True)
            self.proto_init.data = torch.ones(1)
                
                    

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