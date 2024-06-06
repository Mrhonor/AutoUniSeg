import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from itertools import chain
import torch.utils.checkpoint as cp
from collections import defaultdict
from math import log2
import numpy as np

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

__all__ = ['ResNet', 'resnet18', 'resnet34', 'BasicBlock']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

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

class BNReLUConv(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False, n_bn=1):
        ## n_bn bn层数量，对应混合的数据集数量
        super(BNReLUConv, self).__init__()
        self.bn = nn.ModuleList([nn.BatchNorm2d(in_chan, affine=False) for i in range(0, n_bn)])
        ## 采用共享的affine parameter
        self.affine_weight = nn.Parameter(torch.empty(in_chan))
        self.affine_bias = nn.Parameter(torch.empty(in_chan))

        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x, dataset_id):
        # feat = self.conv(x)
        feat = x
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
        feat = self.conv(feat)
        return feat

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

class _UpsampleBlend_mulbn(nn.Module):
    def __init__(self, num_features, use_bn=True, use_skip=True, detach_skip=False, fixed_size=None, k=3,
                 separable=False, n_bn=1):
        super(_UpsampleBlend_mulbn, self).__init__()
        self.blend_conv = BNReLUConv(num_features, num_features, ks=k, n_bn=n_bn)
        self.use_skip = use_skip
        self.detach_skip = detach_skip
        self.upsampling_method = upsample
        if fixed_size is not None:
            self.upsampling_method = lambda x, size: F.interpolate(x, mode='nearest', size=fixed_size)

    def forward(self, x, skip, dataset_id):
        if self.detach_skip:
            skip = skip.detach()
        skip_size = skip.size()[-2:]
        x = self.upsampling_method(x, skip_size)
        if self.use_skip:
            x = x + skip
        x = self.blend_conv.forward(x, dataset_id)
        return x

class ConvBN(nn.Module):
    ## ConvBNReLU类去掉ReLu层
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False, n_bn=1):
        ## n_bn bn层数量，对应混合的数据集数量
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.ModuleList([nn.BatchNorm2d(out_chan, affine=True) for i in range(0, n_bn)])
        ## 采用共享的affine parameter
        self.affine_weight = nn.Parameter(torch.empty(out_chan))
        self.affine_bias = nn.Parameter(torch.empty(out_chan))
        

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
        # feat = feat * self.affine_weight.reshape(1,-1,1,1) + self.affine_bias.reshape(1,-1,1,1) 
        return feat
        

        
        ## TODO 此处可以优化，不同数据集的图像过卷积层可以拼接到一起，过BN层再分离
        # feat = self.conv(x)
        # feat_list = [None] * len(dataset_id)
        # for i in set(dataset_id.cpu().numpy()):
        #     feat_ = self.bn[i](feat[dataset_id == i])
        #     feat_ = feat_ * self.affine_weight.reshape(1,-1,1,1) + self.affine_bias.reshape(1,-1,1,1) 
        #     j = 0
        #     for index, val in enumerate(dataset_id):
        #         if val == i:
        #             feat_list[index] = feat_[j][None]
        # feat = torch.cat(feat_list, dim=0)
        # return feat
        
        # if len(other_x) != 0:
        #     batch_size = [x.shape[0]]
        #     for i in range(0, len(other_x)):
        #         batch_size.append(other_x[i].shape[0])
        #         x = torch.cat((x, other_x[i]), 0)
        #     feat = self.conv(x)
        #     feats = []
        #     begin_index = 0
        #     for i in range(0, len(other_x) + 1):
        #         end_index = begin_index + batch_size[i]
        #         feat_ = self.bn[i](feat[begin_index: end_index])
                
        #         ## affine param
        #         feat_ = feat_ * self.affine_weight.reshape(1,-1,1,1) + self.affine_bias.reshape(1,-1,1,1)
                
        #         feats.append(feat_)
        #         begin_index = end_index
        # else:
        #     feat = self.conv(x)
        #     feat = self.bn[dataset](feat)
            
        #     ## affine param
        #     feat = feat * self.affine_weight.reshape(1,-1,1,1) + self.affine_bias.reshape(1,-1,1,1)
            
        #     feats = [feat]
        

    def SetLastBNAttr(self, attr):
        self.affine_weight.last_bn = attr
        self.affine_bias.last_bn = attr
        # for bn in self.bn:
        #     bn.last_bn = attr

def convkxk(in_planes, out_planes, stride=1, k=3):
    """kxk convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=stride, padding=k // 2, bias=False)


def _bn_function_factory(conv, norm, relu=None):
    def bn_function(x):
        x = norm(conv(x))
        if relu is not None:
            x = relu(x)
        return x

    return bn_function

def _mulbn_function_factory(conv, norm, affine_weight, affine_bias, relu=None):
    def bn_function(x, dataset_id):
        feat = conv(x)
        feat_list = []
        cur_pos = 0
        for i in range(0, len(dataset_id)):
            if dataset_id[i] != dataset_id[cur_pos]:
                feat_ = norm[dataset_id[cur_pos]](feat[cur_pos:i])
                feat_list.append(feat_)
                cur_pos = i
        feat_ = norm[dataset_id[cur_pos]](feat[cur_pos:])
        feat_list.append(feat_)
        feat = torch.cat(feat_list, dim=0)
        # feat = feat * affine_weight.reshape(1,-1,1,1) + affine_bias.reshape(1,-1,1,1) 
        return feat
        
        # x = norm(conv(x))
        # if relu is not None:
        #     x = relu(x)
        # return x

    return bn_function


def do_efficient_fwd(block, x, efficient):
    # return block(x)
    if efficient and x.requires_grad:
        return cp.checkpoint(block, x)
    else:
        return block(x)
    
def do_efficient_fwd_mulbn(block, x, efficient, dataset_id):
    # return block(x)
    if efficient and x.requires_grad:
        return cp.checkpoint(block, x, dataset_id)
    else:
        return block(x, dataset_id)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True, bn_class=nn.BatchNorm2d, levels=3):
        super(BasicBlock, self).__init__()
        self.conv1 = convkxk(inplanes, planes, stride)
        self.bn1 = nn.ModuleList([bn_class(planes) for _ in range(levels)])
        self.relu_inp = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = convkxk(planes, planes)
        self.bn2 = nn.ModuleList([bn_class(planes) for _ in range(levels)])
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient
        self.num_levels = levels

    def forward(self, x, level):
        residual = x

        bn_1 = _bn_function_factory(self.conv1, self.bn1[level], self.relu_inp)
        bn_2 = _bn_function_factory(self.conv2, self.bn2[level])

        out = do_efficient_fwd(bn_1, x, self.efficient)
        out = do_efficient_fwd(bn_2, out, self.efficient)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        relu = self.relu(out)

        return relu, out

    # def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
    #                           missing_keys, unexpected_keys, error_msgs):
    #     super(BasicBlock, self)._load_from_state_dict(state_dict, prefix, local_metadata, False, missing_keys,
    #                                                   unexpected_keys, error_msgs)
    #     missing_keys = []
    #     unexpected_keys = []
    #     for bn in self.bn1:
    #         bn._load_from_state_dict(state_dict, prefix + 'bn1.', local_metadata, strict, missing_keys, unexpected_keys,
    #                                  error_msgs)
    #     for bn in self.bn2:
    #         bn._load_from_state_dict(state_dict, prefix + 'bn2.', local_metadata, strict, missing_keys, unexpected_keys,
    #                                  error_msgs)
                                    
class MulBNBlock(nn.Module):
    expansion = 1

    def __init__(self, configer, inplanes, planes, stride=1, downsample=None, efficient=True, bn_class=nn.BatchNorm2d, levels=3):
        super(MulBNBlock, self).__init__()
        self.configer = configer
        self.n_datasets = configer.get("n_datasets")
        self.conv1 = convkxk(inplanes, planes, stride)
        # print('inplanes:', inplanes)
        # print('planes:', planes)
        self.bn1 = nn.ModuleList([nn.ModuleList([bn_class(planes, affine=True) for _ in range(self.n_datasets)]) for _ in range(levels)])
        self.relu_inp = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = convkxk(planes, planes)
        self.bn2 = nn.ModuleList([nn.ModuleList([bn_class(planes, affine=True) for _ in range(self.n_datasets)]) for _ in range(levels)])
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient
        self.num_levels = levels
                
        self.affine_weight1 = nn.ParameterList([nn.Parameter(torch.empty(planes)) for _ in range(levels)])
        self.affine_bias1 = nn.ParameterList([nn.Parameter(torch.empty(planes)) for _ in range(levels)])
                
        self.affine_weight2 = nn.ParameterList([nn.Parameter(torch.empty(planes)) for _ in range(levels)])
        self.affine_bias2 = nn.ParameterList([nn.Parameter(torch.empty(planes)) for _ in range(levels)])

        

    def forward(self, x, level, dataset_id):
        residual = x

        bn_1 = _mulbn_function_factory(self.conv1, self.bn1[level], self.affine_weight1[level], self.affine_bias1[level], self.relu_inp)
        bn_2 = _mulbn_function_factory(self.conv2, self.bn2[level], self.affine_weight2[level], self.affine_bias2[level])

        out = do_efficient_fwd_mulbn(bn_1, x, self.efficient, dataset_id)
        out = do_efficient_fwd_mulbn(bn_2, out, self.efficient, dataset_id)

        if self.downsample is not None:
            residual = self.downsample(x, dataset_id)
        out += residual
        relu = self.relu(out)

        return relu, out

    # def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
    #                           missing_keys, unexpected_keys, error_msgs):
    #     super(MulBNBlock, self)._load_from_state_dict(state_dict, prefix, local_metadata, False, missing_keys,
    #                                                   unexpected_keys, error_msgs)
    #     missing_keys = []
    #     unexpected_keys = []
    #     for bn in self.bn1:
    #         bn._load_from_state_dict(state_dict, prefix + 'bn1.', local_metadata, strict, missing_keys, unexpected_keys,
    #                                  error_msgs)
    #     for bn in self.bn2:
    #         bn._load_from_state_dict(state_dict, prefix + 'bn2.', local_metadata, strict, missing_keys, unexpected_keys,
    #                                  error_msgs)

@BACKBONE_REGISTRY.register()
class SnpResNet(Backbone):
    def _make_layer(self, block, planes, blocks, stride=1, bn_class=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                bn_class(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.efficient, bn_class=bn_class,
                            levels=self.pyramid_levels))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_class=bn_class, levels=self.pyramid_levels, efficient=self.efficient))

        return nn.Sequential(*layers)

    def __init__(self, cfg, input_shape):
        block=BasicBlock
        layers=[2, 2, 2, 2]
        num_features=128 
        pyramid_levels=3
        use_bn=True
        k_bneck=1
        k_upsample=3
        efficient=True
        upsample_skip=True
        detach_upsample_skips=()
        detach_upsample_in=False
        align_corners=None
        pyramid_subsample='bicubic' 
        target_size=None
        output_stride=4
        
        self.inplanes = 64
        self.efficient = efficient
        super(SnpResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn_class = nn.BatchNorm2d if use_bn else Identity
        # self.register_buffer('img_mean', torch.tensor(mean).view(1, -1, 1, 1))
        # self.register_buffer('img_std', torch.tensor(std).view(1, -1, 1, 1))
        # if scale != 1:
        #     self.register_buffer('img_scale', torch.tensor(scale).view(1, -1, 1, 1).float())

        self.pyramid_levels = pyramid_levels
        self.num_features = num_features
        self.replicated = False

        self.align_corners = align_corners
        self.pyramid_subsample = pyramid_subsample

        self.bn1 = nn.ModuleList([bn_class(64) for _ in range(pyramid_levels)])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        bottlenecks = []
        self.layer1 = self._make_layer(block, 64, layers[0], bn_class=bn_class)
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, bn_class=bn_class)
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, bn_class=bn_class)
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, bn_class=bn_class)
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)]

        num_bn_remove = max(0, int(log2(output_stride) - 2))
        self.num_skip_levels = self.pyramid_levels + 3 - num_bn_remove
        bottlenecks = bottlenecks[num_bn_remove:]

        self.fine_tune = ['conv1', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'bn1']

        self.upsample_bottlenecks = nn.ModuleList(bottlenecks[::-1])
        num_pyr_modules = 2 + pyramid_levels - num_bn_remove
        self.target_size = target_size
        if self.target_size is not None:
            h, w = target_size
            target_sizes = [(h // 2 ** i, w // 2 ** i) for i in range(2, 2 + num_pyr_modules)][::-1]
        else:
            target_sizes = [None] * num_pyr_modules
        self.upsample_blends = nn.ModuleList(
            [_UpsampleBlend(num_features,
                            use_bn=use_bn,
                            use_skip=upsample_skip,
                            detach_skip=i in detach_upsample_skips,
                            fixed_size=ts,
                            k=k_upsample)
             for i, ts in enumerate(target_sizes)])
        self.detach_upsample_in = detach_upsample_in

        self.random_init = ['upsample_bottlenecks', 'upsample_blends']

        self.features = num_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def random_init_params(self): 
        params = []
        for f in self.random_init:
            params.extend(f.parameters())
        return params

    def fine_tune_params(self):
        params = []
        for f in self.fine_tune:
            params.extend(f.parameters())
        return params

    def forward_resblock(self, x, layers, idx):
        skip = None
        for l in layers:
            x = l(x) if not isinstance(l, BasicBlock) else l(x, idx)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward_down(self, image, skips, idx=-1):
        x = self.conv1(image)
        x = self.bn1[idx](x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x, skip = self.forward_resblock(x, self.layer1, idx)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2, idx)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3, idx)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer4, idx)
        features += [skip]

        skip_feats = [b(f) for b, f in zip(self.upsample_bottlenecks, reversed(features))]

        for i, s in enumerate(reversed(skip_feats)):
            skips[idx + i] += [s]

        return skips

    def forward(self, image):
        # if isinstance(self.bn1[0], nn.BatchNorm2d):
        #     if hasattr(self, 'img_scale'):
        #         image /= self.img_scale
        #     image -= self.img_mean
        #     image /= self.img_std
        pyramid = [image]
        for l in range(1, self.pyramid_levels):
            if self.target_size is not None:
                ts = list([si // 2 ** l for si in self.target_size])
                pyramid += [
                    F.interpolate(image, size=ts, mode=self.pyramid_subsample, align_corners=self.align_corners)]
            else:
                pyramid += [F.interpolate(image, scale_factor=1 / 2 ** l, mode=self.pyramid_subsample,
                                          align_corners=self.align_corners)]
        skips = [[] for _ in range(self.num_skip_levels)]
        additional = {'pyramid': pyramid}
        for idx, p in enumerate(pyramid):
            skips = self.forward_down(p, skips, idx=idx)
        skips = skips[::-1]
        x = skips[0][0]
        if self.detach_upsample_in:
            x = x.detach()
        for i, (sk, blend) in enumerate(zip(skips[1:], self.upsample_blends)):
            x = blend(x, sum(sk))
        return x, additional

    # def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
    #                           missing_keys, unexpected_keys, error_msgs):
    #     super(ResNet, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
    #                                               unexpected_keys, error_msgs)
    #     for bn in self.bn1:
    #         bn._load_from_state_dict(state_dict, prefix + 'bn1.', local_metadata, strict, missing_keys, unexpected_keys,
    #                                  error_msgs)

class ResNet_mulbn(nn.Module):
    def _make_layer(self, block, planes, blocks, stride=1, bn_class=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     nn.Conv2d(self.inplanes, planes * block.expansion,
            #               kernel_size=1, stride=stride, bias=False),
            #     bn_class(planes * block.expansion),
            # )
            downsample = ConvBN(self.inplanes, planes * block.expansion, ks=1, stride=stride, padding=0, n_bn=self.n_datasets)

        layers = []
        layers.append(block(self.configer, self.inplanes, planes, stride, downsample, self.efficient, bn_class=bn_class,
                            levels=self.pyramid_levels))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.configer, self.inplanes, planes, bn_class=bn_class, levels=self.pyramid_levels, efficient=self.efficient))

        return nn.Sequential(*layers)

    def __init__(self, configer, block, layers, *, num_features=128, pyramid_levels=3, use_bn=True, k_bneck=1, k_upsample=3,
                 efficient=False, upsample_skip=True, detach_upsample_skips=(), detach_upsample_in=False,
                 align_corners=None, pyramid_subsample='bicubic', target_size=None,
                 output_stride=4, **kwargs):
        self.configer = configer
        self.n_datasets = self.configer.get("n_datasets")
        self.inplanes = 64
        self.efficient = efficient
        super(ResNet_mulbn, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn_class = nn.BatchNorm2d if use_bn else Identity
        # self.register_buffer('img_mean', torch.tensor(mean).view(1, -1, 1, 1))
        # self.register_buffer('img_std', torch.tensor(std).view(1, -1, 1, 1))
        # if scale != 1:
        #     self.register_buffer('img_scale', torch.tensor(scale).view(1, -1, 1, 1).float())

        self.pyramid_levels = pyramid_levels
        self.num_features = num_features
        self.replicated = False

        self.align_corners = align_corners
        self.pyramid_subsample = pyramid_subsample

        self.bn1 = nn.ModuleList([nn.ModuleList([bn_class(64, affine=False) for _ in range(self.n_datasets)]) for _ in range(pyramid_levels)])
        self.affine_weight1 = nn.ParameterList([nn.Parameter(torch.empty(64)) for _ in range(pyramid_levels)])
        self.affine_bias1 = nn.ParameterList([nn.Parameter(torch.empty(64)) for _ in range(pyramid_levels)])
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        bottlenecks = []
        self.layer1 = self._make_layer(block, 64, layers[0], bn_class=bn_class)
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, bn_class=bn_class)
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, bn_class=bn_class)
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, bn_class=bn_class)
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)]

        num_bn_remove = max(0, int(log2(output_stride) - 2))
        self.num_skip_levels = self.pyramid_levels + 3 - num_bn_remove
        bottlenecks = bottlenecks[num_bn_remove:]

        self.fine_tune = [self.conv1, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4, self.bn1]

        self.upsample_bottlenecks = nn.ModuleList(bottlenecks[::-1])
        num_pyr_modules = 2 + pyramid_levels - num_bn_remove
        self.target_size = target_size
        if self.target_size is not None:
            h, w = target_size
            target_sizes = [(h // 2 ** i, w // 2 ** i) for i in range(2, 2 + num_pyr_modules)][::-1]
        else:
            target_sizes = [None] * num_pyr_modules
        self.upsample_blends = nn.ModuleList(
            [_UpsampleBlend_mulbn(num_features,
                            use_bn=use_bn,
                            use_skip=upsample_skip,
                            detach_skip=i in detach_upsample_skips,
                            fixed_size=ts,
                            k=k_upsample,
                            n_bn=self.n_datasets)
             for i, ts in enumerate(target_sizes)])
        self.detach_upsample_in = detach_upsample_in

        self.random_init = [self.upsample_bottlenecks, self.upsample_blends, self.affine_weight1, self.affine_bias1]

        self.features = num_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
        for name, param in self.named_parameters():
            if name.find('affine_weight') != -1:
                nn.init.ones_(param)
            elif name.find('affine_bias') != -1:
                nn.init.zeros_(param)

    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    def fine_tune_params(self):
        return chain(*[f.parameters() for f in self.fine_tune])

    def forward_resblock(self, x, layers, idx, dataset_id):
        skip = None
        for l in layers:
            x = l(x) if not isinstance(l, MulBNBlock) else l(x, idx, dataset_id)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward_down(self, image, skips, dataset_id, idx=-1):
        # x = self.conv1(image)
        # feat_list = [None] * len(dataset_id)
        # for i in set(dataset_id.cpu().numpy()):
        #     feat_ = self.bn1[idx][i](x[dataset_id == i])
        #     feat_ = feat_ * self.affine_weight[idx].reshape(1,-1,1,1) + self.affine_bias[idx].reshape(1,-1,1,1) 
        #     j = 0
        #     for index, val in enumerate(dataset_id):
        #         if val == i:
        #             feat_list[index] = feat_[j][None]
        # feat = torch.cat(feat_list, dim=0)
        feat = self.conv1(image)
        feat_list = []
        cur_pos = 0
        for i in range(0, len(dataset_id)):
            if dataset_id[i] != dataset_id[cur_pos]:
                feat_ = self.bn1[idx][dataset_id[cur_pos]](feat[cur_pos:i])
                feat_list.append(feat_)
                cur_pos = i
        feat_ = self.bn1[idx][dataset_id[cur_pos]](feat[cur_pos:])
        feat_list.append(feat_)
        feat = torch.cat(feat_list, dim=0)
        feat = feat * self.affine_weight1[idx].reshape(1,-1,1,1) + self.affine_bias1[idx].reshape(1,-1,1,1) 

        # x = self.bn1[idx](x)
        x = self.relu(feat)
        x = self.maxpool(x)

        features = []
        x, skip = self.forward_resblock(x, self.layer1, idx, dataset_id)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2, idx, dataset_id)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3, idx, dataset_id)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer4, idx, dataset_id)
        features += [skip]

        skip_feats = [b(f) for b, f in zip(self.upsample_bottlenecks, reversed(features))]

        for i, s in enumerate(reversed(skip_feats)):
            skips[idx + i] += [s]

        return skips

    def forward(self, image, dataset_id):
        # if isinstance(self.bn1[0], nn.BatchNorm2d):
        #     if hasattr(self, 'img_scale'):
        #         image /= self.img_scale
        #     image -= self.img_mean
        #     image /= self.img_std
        pyramid = [image]
        for l in range(1, self.pyramid_levels):
            if self.target_size is not None:
                ts = list([si // 2 ** l for si in self.target_size])
                pyramid += [
                    F.interpolate(image, size=ts, mode=self.pyramid_subsample, align_corners=self.align_corners)]
            else:
                pyramid += [F.interpolate(image, scale_factor=1 / 2 ** l, mode=self.pyramid_subsample,
                                          align_corners=self.align_corners)]
        skips = [[] for _ in range(self.num_skip_levels)]
        additional = {'pyramid': pyramid}
        for idx, p in enumerate(pyramid):
            skips = self.forward_down(p, skips, dataset_id, idx=idx)
        skips = skips[::-1]
        x = skips[0][0]
        if self.detach_upsample_in:
            x = x.detach()
        for i, (sk, blend) in enumerate(zip(skips[1:], self.upsample_blends)):
            x = blend(x, sum(sk), dataset_id)
        return x, additional

    # def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
    #                           missing_keys, unexpected_keys, error_msgs):
    #     super(ResNet_mulbn, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
    #                                               unexpected_keys, error_msgs)
    #     for bn in self.bn1:
    #         bn._load_from_state_dict(state_dict, prefix + 'bn1.', local_metadata, strict, missing_keys, unexpected_keys,
    #                                  error_msgs)

def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    # model_zoo.load_url(model_urls['resnet18'])
    # if pretrained:
    return model

def resnet18_mulbn(configer, pretrained=True, **kwargs):
    """Constructs a mul-bn ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_mulbn(configer, MulBNBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model
