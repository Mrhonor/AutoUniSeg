import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ...utils.configer import Configer

from timm.models.layers import trunc_normal_
import clip
import logging

# backbone_url = './res/hrnetv2_w48_imagenet_pretrained.pth'
logger = logging.getLogger(__name__)
def BNReLU(num_features, bn_type=None, **kwargs):
    if bn_type == 'torchbn':
        return nn.Sequential(
            nn.BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )
    elif bn_type == 'torchsyncbn':
        return nn.Sequential(
            nn.SyncBatchNorm(num_features, **kwargs),
            nn.ReLU()
        )
    elif bn_type == 'syncbn':
        from lib.extensions.syncbn.module import BatchNorm2d
        return nn.Sequential(
            BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )
    elif bn_type == 'sn':
        from lib.extensions.switchablenorms.switchable_norm import SwitchNorm2d
        return nn.Sequential(
            SwitchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )
    elif bn_type == 'gn':
        return nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=num_features, **kwargs),
            nn.ReLU()
        )
    elif bn_type == 'fn':
        Log.error('Not support Filter-Response-Normalization: {}.'.format(bn_type))
        exit(1)
    elif bn_type == 'inplace_abn':
        torch_ver = torch.__version__[:3]
        # Log.info('Pytorch Version: {}'.format(torch_ver))
        if torch_ver == '0.4':
            from lib.extensions.inplace_abn.bn import InPlaceABNSync
            return InPlaceABNSync(num_features, **kwargs)
        elif torch_ver in ('1.0', '1.1'):
            from lib.extensions.inplace_abn_1.bn import InPlaceABNSync
            return InPlaceABNSync(num_features, **kwargs)
        elif torch_ver == '1.2':
            from inplace_abn import InPlaceABNSync
            return InPlaceABNSync(num_features, **kwargs)

    else:
        Log.error('Not support BN type: {}.'.format(bn_type))
        exit(1)


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, up_factor=8, proj='convmlp', bn_type='torchsyncbn', up_sample=False, down_sample=False, n_bn=1):
        super(ProjectionHead, self).__init__()

        # Log.info('proj_dim: {}'.format(proj_dim))
        self.n_bn = n_bn

        self.up_sample = up_sample
        if self.up_sample:
            self.Upsample = nn.Upsample(scale_factor=up_factor, mode='nearest')
            

        if proj == 'linear':
            raise Exception("Not Imp error")
            # if down_sample:
            #     raise Exception("Not Imp error")
                
            # self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            if down_sample:
                # self.conv1 = ConvBNReLU(dim_in, dim_in*2, ks=3, stride=1, padding=1, n_bn=self.n_bn)
                # self.conv_last = nn.Conv2d(dim_in*2, proj_dim, kernel_size=1, bias=True)
                
                self.proj = nn.Sequential(
                    nn.Conv2d(dim_in, dim_in*2, kernel_size=3, stride=1, padding=1),
                    BNReLU(dim_in*2, bn_type=bn_type),
                    nn.Conv2d(dim_in*2, proj_dim, kernel_size=1, bias=True)
                )
            else:
                # self.conv1 = ConvBNReLU(dim_in, dim_in*2, ks=3, stride=1, padding=1, n_bn=self.n_bn)
                # self.conv_last = nn.Conv2d(dim_in*2, proj_dim, kernel_size=1, bias=True)
                
                
                self.proj = nn.Sequential(
                    nn.Conv2d(dim_in, dim_in, kernel_size=1),
                    BNReLU(dim_in, bn_type=bn_type),
                    nn.Conv2d(dim_in, proj_dim, kernel_size=1)
                )

    def forward(self, x):
        # feats = self.conv1(dataset, x, *other_x)
        # feats = [self.conv_last(feat) for feat in feats]
        feat = self.proj(x)

        feat = F.normalize(feat, p=2, dim=1)
        if self.up_sample:
            feat = self.Upsample(feat)

        return feat

@SEM_SEG_HEADS_REGISTRY.register()
class HRNet_W48_naive_concat(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """
    @configurable
    def __init__(self, *, 
                datasets_cats,
                aux_mode,
                output_feat_dim, 
                num_unify_class,
                with_datasets_aux,
                bn_type,
                input_shape,
                configer):
        super(HRNet_W48_naive_concat, self).__init__()
        self.aux_mode = aux_mode
        # self.num_unify_classes = num_unify_classes')
        self.datasets_cats = datasets_cats
        self.n_datasets = len(self.datasets_cats)
        # self.backbone = backbone
        self.output_feat_dim = output_feat_dim
        self.configer = configer

        # extra added layers
        in_channels = input_shape  # 48 + 96 + 192 + 384

        self.proj_head = ProjectionHead(dim_in=in_channels, proj_dim=self.output_feat_dim, bn_type=bn_type)

        self.total_cats = 0
        # self.datasets_cats = []
        for i in range(0, self.n_datasets):
            # self.datasets_cats.append(self.configer.get('dataset'+str(i+1), 'n_cats'))
            self.total_cats += self.datasets_cats[i]
        
        self.num_unify_class = num_unify_class
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
            

        self.unify_prototype = nn.Parameter(torch.zeros(num_unify_class, self.output_feat_dim),
                                requires_grad=True)
        trunc_normal_(self.unify_prototype, std=0.02)
        
        self.with_datasets_aux = with_datasets_aux
        self.init_weights()    

        

    @classmethod
    def from_config(cls, cfg, input_shape):
        # backbone = build_backbone(cfg)
        aux_mode = cfg.MODEL.AUX_MODE
        num_unify_class = cfg.DATASETS.NUM_UNIFY_CLASS
        datasets_cats = cfg.DATASETS.DATASETS_CATS
        output_feat_dim = cfg.MODEL.SEM_SEG_HEAD.OUTPUT_FEAT_DIM
        with_datasets_aux = cfg.MODEL.GNN.with_datasets_aux
        bn_type = cfg.MODEL.SEM_SEG_HEAD.BN_TYPE
        configer = Configer(configs=cfg.DATASETS.CONFIGER)
        return {
            # 'backbone': backbone,
            'aux_mode': aux_mode,
            'datasets_cats': datasets_cats,
            'output_feat_dim': output_feat_dim,
            'with_datasets_aux': with_datasets_aux, 
            'bn_type': bn_type,
            'input_shape': input_shape,
            'num_unify_class': num_unify_class,
            'configer': configer,
        }



    def forward(self, x, dataset_ids=0):
        # x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        emb = self.proj_head(feats)

        logits = torch.einsum('bchw, nc -> bnhw', emb, self.unify_prototype.to(emb.dtype))
            
        return {'logits':logits, 'emb':emb}

    def req_grad(self, isFrooze):
        for name, param in self.named_parameters():
            if 'bipartite_graphs' in name:
                continue
            param.requires_grad = isFrooze

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
        
        if unify_prototype.shape[0] != self.unify_prototype.shape[0]:
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
        print(f"text_feature shape:{text_feature_vecs.shape}")
        self.unify_prototype.data = text_feature_vecs
        self.unify_prototype.requires_grad=False
                
