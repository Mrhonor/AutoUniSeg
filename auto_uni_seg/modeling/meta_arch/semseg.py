import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import warnings
import numpy as np
import clip

from ...utils.configer import Configer

from timm.models.layers import trunc_normal_
from detectron2.config import configurable
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
import logging
logger = logging.getLogger(__name__)

upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=False)

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1,
                 drop_rate=.0, separable=False):
        super(_BNReluConv, self).__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=batch_norm is True))
        padding = k // 2
        conv_class = SeparableConv2d if separable else nn.Conv2d
        
        self.add_module('conv', conv_class(num_maps_in, num_maps_out, kernel_size=k, padding=padding, bias=bias,
                                           dilation=dilation))
        if drop_rate > 0:
            self.add_module('dropout', nn.Dropout2d(drop_rate, inplace=True))

class _UpsampleBlend(nn.Module):
    def __init__(self, num_features, use_bn=True, use_skip=True, detach_skip=False, fixed_size=None, k=3,
                 separable=False):
        super(_UpsampleBlend, self).__init__()
        self.blend_conv = _BNReluConv(num_features, num_features, k=k, batch_norm=use_bn, separable=separable)
        self.use_skip = use_skip
        self.detach_skip = detach_skip
        
        self.upsampling_method = upsample
        if fixed_size is not None:
            self.upsampling_method = lambda x, size: F.interpolate(x, mode='nearest', size=fixed_size)

    def forward(self, x, skip):
        if self.detach_skip:
            skip = skip.detach()
        skip_size = skip.size()[-2:]
        x = self.upsampling_method(x, skip_size)
        if self.use_skip:
            x = x + skip
        x = self.blend_conv.forward(x)
        return x



class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False, n_bn=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.ModuleList([nn.BatchNorm2d(out_chan, affine=False) for i in range(0, n_bn)])
        self.affine_weight = nn.Parameter(torch.empty(out_chan))
        self.affine_bias = nn.Parameter(torch.empty(out_chan))
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x, dataset_id):
        feat = self.conv(x)
        feat_list = []
        cur_pos = 0
        for i in range(0, len(dataset_id)):
            if dataset_id[i] != dataset_id[cur_pos]:
                feat_ = self.bn[dataset_id[cur_pos]](feat[cur_pos:i])
                feat_list.append(feat_)
                cur_pos = i
        feat_ = self.bn[dataset_id[cur_pos]](feat[cur_pos:])
        feat_list.append(feat_)
        feat = torch.cat(feat_list, dim=0)
        feat = feat * self.affine_weight.reshape(1,-1,1,1) + self.affine_bias.reshape(1,-1,1,1) 
        feat = self.relu(feat)
        return feat

    
@SEM_SEG_HEADS_REGISTRY.register()   
class SemsegModel(nn.Module):
    def __init__(self, cfg, input_shape):
    # configer,  
        super(SemsegModel, self).__init__()
        use_bn=True
        k=1
        bias=True,
        loss_ret_additional=False
        upsample_logits=True
        logit_class=_BNReluConv
        
        multiscale_factors=(.5, .75, 1.5, 2.)
        
        self.configer = Configer(configs=cfg.DATASETS.CONFIGER)
        self.aux_mode = cfg.MODEL.AUX_MODE
        self.datasets_cats = cfg.DATASETS.DATASETS_CATS
        self.n_datasets = len(self.datasets_cats)
        self.with_datasets_aux = cfg.MODEL.SEM_SEG_HEAD.WITH_DATASETS_AUX
        self.total_cats = 0
        for i in range(0, self.n_datasets):
            self.total_cats += self.datasets_cats[i]

        self.output_feat_dim = cfg.MODEL.SEM_SEG_HEAD.OUTPUT_FEAT_DIM
        init_adj_path = cfg.MODEL.GNN.INIT_ADJ_PATH
        if init_adj_path != None:
            init_adj = torch.load(init_adj_path)
            num_unify_class = init_adj.shape[1]
        else:
            num_unify_class = cfg.DATASETS.NUM_UNIFY_CLASS

        # self.logits = logit_class(self.backbone.num_features, self.output_feat_dim, batch_norm=use_bn, k=k, bias=bias) 
        self.logits = logit_class(input_shape, self.output_feat_dim, batch_norm=use_bn, k=k, bias=bias) 
        self.bipartite_graphs = nn.ParameterList([])
        cur_cat = 0 
        for i in range(0, self.n_datasets):
            this_bigraph = torch.zeros(self.datasets_cats[i], self.num_unify_class)
            if self.num_unify_class == self.total_cats:
                for j in range(0, self.datasets_cats[i]):
                    this_bigraph[j, cur_cat+j] = 1
            cur_cat += self.datasets_cats[i]
            self.bipartite_graphs.append(nn.Parameter(
                this_bigraph, requires_grad=False
                ))
            

        self.unify_prototype = nn.Parameter(torch.zeros(self.num_unify_class, self.output_feat_dim),
                                requires_grad=False)
        trunc_normal_(self.unify_prototype, std=0.02)
        
        self.with_datasets_aux = cfg.MODEL.GNN.with_datasets_aux
        if self.with_datasets_aux:
            self.aux_prototype = nn.ParameterList([])
            for i in range(0, self.n_datasets):
                self.aux_prototype.append(nn.Parameter(torch.zeros(self.datasets_cats[i], self.output_feat_dim),
                                        requires_grad=True))
                trunc_normal_(self.aux_prototype[i], std=0.02)


        if self.num_unify_class == self.total_cats:
            self.get_encode_lb_vec()
        self.num_classes = self.num_unify_class
        # self.logits = logit_class(self.backbone.num_features, self.num_classes, batch_norm=use_bn, k=k, bias=bias)


        self.criterion = None
        self.loss_ret_additional = loss_ret_additional
        self.img_req_grad = loss_ret_additional
        self.upsample_logits = upsample_logits
        self.multiscale_factors = multiscale_factors

    def forward(self, features, dataset_ids=0):
        feat, _ = features
        emb = self.logits.forward(feat)
        if self.training:
            logits = torch.einsum('bchw, nc -> bnhw', emb, self.unify_prototype)
            remap_logits = []
            for i in range(self.n_datasets):
                if not (dataset_ids == i).any():
                    continue
                remap_logits.append(torch.einsum('bchw, nc -> bnhw', logits[dataset_ids==i], self.bipartite_graphs[i]))
            
            if self.with_datasets_aux:
                cur_cat = 0
                aux_logits = []
                for i in range(self.n_datasets):
                    aux_logits.append(torch.einsum('bchw, nc -> bnhw', emb, self.aux_prototype[i]))
                    cur_cat += self.datasets_cats[i]
                    
                return {'logits':remap_logits, 'aux_logits':aux_logits, 'emb':emb}
            
            return {'logits':remap_logits, 'emb':emb}
        else:
            logits = torch.einsum('bchw, nc -> bnhw', emb, self.unify_prototype) 
            if not isinstance(dataset_ids, int):
                remap_logits = []
                for i in range(self.n_datasets):
                    if not (dataset_ids == i).any():
                        continue
                    remap_logits.append(torch.einsum('bchw, nc -> bnhw', logits[dataset_ids==i], self.bipartite_graphs[i]))
            else:
                remap_logits = torch.einsum('bchw, nc -> bnhw', logits, self.bipartite_graphs[dataset_ids])
            return {'logits':remap_logits, 'emb':emb}

    def get_encode_lb_vec(self):
        text_feature_vecs = []
        with torch.no_grad():
            clip_model, _ = clip.load("ViT-B/32", device="cpu")
            for i in range(0, self.n_datasets):
                lb_name = self.configer.get("dataset"+str(i+1), "label_names")
                lb_name = [f'a photo of {name} from dataset {i+1}.' for name in lb_name]
                text = clip.tokenize(lb_name) #.cuda()
                text_features = clip_model.encode_text(text).type(torch.float32)
                text_feature_vecs.append(text_features)
                
        text_feature_vecs = torch.cat(text_feature_vecs, dim=0)
        self.unify_prototype.data = text_feature_vecs
        self.unify_prototype.requires_grad=False

    def forward_down(self, image, target_size, image_size):
        return self.backbone.forward_down(image), target_size, image_size

    def forward_up(self, feats, target_size, image_size):
        feats, additional = self.backbone.forward_up(feats)
        features = upsample(feats, target_size)
        logits = self.logits.forward(features)
        logits = upsample(logits, image_size)
        return logits, additional

    def prepare_data(self, batch, image_size, device=torch.device('cuda'), img_key='image'):
        if image_size is None:
            image_size = batch['target_size']
        warnings.warn(f'Image requires grad: {self.img_req_grad}', UserWarning)
        image = batch[img_key].detach().requires_grad_(self.img_req_grad).to(device)
        return {
            'image': image,
            'image_size': image_size,
            'target_size': batch.get('target_size_feats')
        }

    def do_forward(self, batch, image_size=None):
        data = self.prepare_data(batch, image_size)
        logits, additional = self.forward(**data)
        additional['model'] = self
        additional = {**additional, **data}
        return logits, additional

    def loss(self, batch):
        assert self.criterion is not None
        labels = batch['labels'].cuda()
        logits, additional = self.do_forward(batch, image_size=labels.shape[-2:])
        if self.loss_ret_additional:
            return self.criterion(logits, labels, batch=batch, additional=additional), additional
        return self.criterion(logits, labels, batch=batch, additional=additional)

    def random_init_params(self):
        params = self.backbone.random_init_params()
        if self.unify_prototype.requires_grad:
            params.extend(self.unify_prototype)        
        if self.with_datasets_aux:
            if self.aux_prototype[0].requires_grad:
                params.extend(self.aux_prototype)
        # self.logits.parameters(), 
        # if hasattr(self, 'border_logits'):
        #     params += [self.border_logits.parameters()]
        return params

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()

    def ms_forward(self, batch, image_size=None):
        image_size = batch.get('target_size', image_size if image_size is not None else batch['image'].shape[-2:])
        ms_logits = None
        pyramid = [batch['image'].cuda()]
        pyramid += [
            F.interpolate(pyramid[0], scale_factor=sf, mode=self.backbone.pyramid_subsample,
                          align_corners=self.backbone.align_corners) for sf in self.multiscale_factors
        ]
        for image in pyramid:
            batch['image'] = image
            logits, additional = self.do_forward(batch, image_size=image_size)
            if ms_logits is None:
                ms_logits = torch.zeros(logits.size()).to(logits.device)
            ms_logits += F.softmax(logits, dim=1)
        batch['image'] = pyramid[0].cpu()
        return ms_logits / len(pyramid), {}

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
        if self.with_datasets_aux:
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
        
    def set_aux_grad(self, grad=False):

        for i in range(self.n_datasets):

            self.aux_prototype[i].requires_grad=grad


    def get_optim_params(self):
        fine_tune_factor = 4
        optim_params = [
            {'params': self.random_init_params(), 'lr': self.configer.get('lr', 'seg_lr_start'), 'weight_decay': self.configer.get('lr', 'weight_decay')},
            {'params': self.fine_tune_params(), 'lr': self.configer.get('lr', 'seg_lr_start') / fine_tune_factor,
            'weight_decay': self.configer.get('lr', 'weight_decay') / fine_tune_factor},
        ]
        return optim_params
