'''
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from:
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/model/segmentors/encoder_decoder.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.config import configurable
from ..backbone.afformer import afformer_tiny, afformer_base, afformer_small
from timm.models.layers import trunc_normal_

def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    print(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

@BACKBONE_REGISTRY.register()
class AFFormerEncoderDecoder(Backbone):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """
    @configurable
    def __init__(self, *,
                 cfg=None,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(AFFormerEncoderDecoder, self).__init__()
        with open(cfg.MODEL.AFFORMER_CONFIG, 'r') as f:
            aff_cfg =  f.read()
        local_namespace = {}
        exec(aff_cfg, {}, local_namespace)
          
        aff_cfg = local_namespace['afformer_model'] 
        
        if aff_cfg['backbone']['type'] == 'afformer_tiny':
            self.backbone = afformer_tiny(cfg=aff_cfg)
        elif aff_cfg['backbone']['type'] == 'afformer_base':
            self.backbone = afformer_base(cfg=aff_cfg)
        elif aff_cfg['backbone']['type'] == 'afformer_small':
            self.backbone = afformer_small(cfg=aff_cfg)
        else:
            raise Exception("backbone type not supported")
            
        if pretrained is not None:
            self.backbone.pretrained = pretrained
        
        if neck is not None:
            self.with_neck = True
            raise Exception("neck is not None")
            # self.neck = builder.build_neck(neck)
        else:
            self.with_neck = False
            
        self._init_decode_head(cfg)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg


    @classmethod
    def from_config(cls, cfg, input_shape):

        return {
            'cfg': cfg,
            'neck': None,
            'auxiliary_head': None,
            'train_cfg': dict(),
            'test_cfg': dict(mode='whole'),
            'pretrained': cfg.get('pretrained', None),
            'init_cfg': None
        }
        

    def _init_decode_head(self, cfg):
        """Initialize ``decode_head``"""
        self.decode_head = build_sem_seg_head(cfg, 0)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(build_sem_seg_head(head_cfg))
            else:
                self.auxiliary_head = build_sem_seg_head(auxiliary_head)
        else:
            self.with_auxiliary_head = False

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward(self, img, img_metas, gt_semantic_seg, return_loss=True, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        if self.training and return_loss:
            x = self.extract_feat(img)

            losses = dict()

            loss_decode = self._decode_head_forward_train(x, img_metas,
                                                        gt_semantic_seg)
            losses.update(loss_decode)

            if self.with_auxiliary_head:
                loss_aux = self._auxiliary_head_forward_train(
                    x, img_metas, gt_semantic_seg)
                losses.update(loss_aux)

            return losses
        else:
            return self.encode_decode(img, img_metas)

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred


@BACKBONE_REGISTRY.register()
class AFFormerMdsEncoderDecoder(Backbone):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """
    @configurable
    def __init__(self, *,
                 cfg=None,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(AFFormerEncoderDecoder, self).__init__()
        with open(cfg.MODEL.AFFORMER_CONFIG, 'r') as f:
            aff_cfg =  f.read()
        local_namespace = {}
        exec(aff_cfg, {}, local_namespace)
          
        aff_cfg = local_namespace['afformer_model'] 
        
        if aff_cfg['backbone']['type'] == 'afformer_tiny':
            self.backbone = afformer_tiny(cfg=aff_cfg)
        elif aff_cfg['backbone']['type'] == 'afformer_base':
            self.backbone = afformer_base(cfg=aff_cfg)
        elif aff_cfg['backbone']['type'] == 'afformer_small':
            self.backbone = afformer_small(cfg=aff_cfg)
        else:
            raise Exception("backbone type not supported")
            
        if pretrained is not None:
            self.backbone.pretrained = pretrained
        
        if neck is not None:
            self.with_neck = True
            raise Exception("neck is not None")
            # self.neck = builder.build_neck(neck)
        else:
            self.with_neck = False
            
        self._init_decode_head(cfg)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.num_unify_class = cfg.DATASETS.NUM_UNIFY_CLASS
        self.datasets_cats = cfg.DATASETS.DATASETS_CATS
        self.n_datasets = len(self.datasets_cats)
        self.output_feat_dim = aff_cfg['decode_head']['num_classes']
        self.total_cats = 0
        # self.datasets_cats = []
        for i in range(0, self.n_datasets):
            # self.datasets_cats.append(self.configer.get('dataset'+str(i+1), 'n_cats'))
            self.total_cats += self.datasets_cats[i]
               
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
                                requires_grad=False)
        trunc_normal_(self.unify_prototype, std=0.02)
        


    @classmethod
    def from_config(cls, cfg, input_shape):

        return {
            'cfg': cfg,
            'neck': None,
            'auxiliary_head': None,
            'train_cfg': dict(),
            'test_cfg': dict(mode='whole'),
            'pretrained': cfg.get('pretrained', None),
            'init_cfg': None
        }
        

    def _init_decode_head(self, cfg):
        """Initialize ``decode_head``"""
        self.decode_head = build_sem_seg_head(cfg, 0)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(build_sem_seg_head(head_cfg))
            else:
                self.auxiliary_head = build_sem_seg_head(auxiliary_head)
        else:
            self.with_auxiliary_head = False

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        emb = self._decode_head_forward_test(x, img_metas)
        
        if self.training:
            logits = torch.einsum('bchw, nc -> bnhw', emb, self.unify_prototype.to(emb.dtype))
            remap_logits = []
            for i in range(self.n_datasets):
                if not (dataset_ids == i).any():
                    continue
                remap_logits.append(torch.einsum('bchw, nc -> bnhw', logits[dataset_ids==i], self.bipartite_graphs[i]))
            
            if self.with_datasets_aux:
                cur_cat = 0
                aux_logits = []
                for i in range(self.n_datasets):
                    aux_logits.append(torch.einsum('bchw, nc -> bnhw', emb[dataset_ids==i], self.aux_prototype[i].to(emb.dtype)))
                    cur_cat += self.datasets_cats[i]
                    
                return {'logits':remap_logits, 'aux_logits':aux_logits, 'emb':emb}
            
            return {'logits':remap_logits, 'emb':emb}
        else:
            # logger.info(f'emb : dtype{emb.dtype}, unify_prototype : dtype{self.unify_prototype.dtype}')
            
            logits = torch.einsum('bchw, nc -> bnhw', emb, self.unify_prototype.to(emb.dtype)) 
            if not isinstance(dataset_ids, int):
                remap_logits = []
                for i in range(self.n_datasets):
                    if not (dataset_ids == i).any():
                        continue
                    remap_logits.append(torch.einsum('bchw, nc -> bnhw', logits[dataset_ids==i], self.bipartite_graphs[i]))
            else:
                remap_logits = [torch.einsum('bchw, nc -> bnhw', logits, self.bipartite_graphs[dataset_ids])]
                
            return {'logits':remap_logits, 'emb':emb, 'uni_logits':logits[None]}

        
        # out = resize(
        #     input=out,
        #     size=img.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None, dataset_ids)

        return seg_logit

    def forward(self, img, img_metas, gt_semantic_seg, dataset_ids, return_loss=True, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        if self.training and return_loss:
            x = self.extract_feat(img)

            losses = dict()

            loss_decode = self._decode_head_forward_train(x, img_metas,
                                                        gt_semantic_seg)
            losses.update(loss_decode)

            if self.with_auxiliary_head:
                loss_aux = self._auxiliary_head_forward_train(
                    x, img_metas, gt_semantic_seg)
                losses.update(loss_aux)

            return losses
        else:
            return self.encode_decode(img, img_metas, dataset_ids)

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale, dataset_ids):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta, dataset_ids)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale, dataset_ids):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta, dataset_ids)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred


@BACKBONE_REGISTRY.register()
class AFFormerMulheadEncoderDecoder(Backbone):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """
    @configurable
    def __init__(self, *,
                 cfg=None,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(AFFormerMulheadEncoderDecoder, self).__init__()
        with open(cfg.MODEL.AFFORMER_CONFIG, 'r') as f:
            aff_cfg =  f.read()
        local_namespace = {}
        exec(aff_cfg, {}, local_namespace)
          
        aff_cfg = local_namespace['afformer_model'] 
        
        if aff_cfg['backbone']['type'] == 'afformer_tiny':
            self.backbone = afformer_tiny(cfg=aff_cfg)
        elif aff_cfg['backbone']['type'] == 'afformer_base':
            self.backbone = afformer_base(cfg=aff_cfg)
        elif aff_cfg['backbone']['type'] == 'afformer_small':
            self.backbone = afformer_small(cfg=aff_cfg)
        else:
            raise Exception("backbone type not supported")
            
        if pretrained is not None:
            self.backbone.pretrained = pretrained
        
        if neck is not None:
            self.with_neck = True
            raise Exception("neck is not None")
            # self.neck = builder.build_neck(neck)
        else:
            self.with_neck = False
            
        self._init_decode_head(cfg)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.num_unify_class = cfg.DATASETS.NUM_UNIFY_CLASS
        self.datasets_cats = cfg.DATASETS.DATASETS_CATS
        self.n_datasets = len(self.datasets_cats)
        self.output_feat_dim = aff_cfg['decode_head']['num_classes']
        self.total_cats = 0
        # self.datasets_cats = []
        for i in range(0, self.n_datasets):
            # self.datasets_cats.append(self.configer.get('dataset'+str(i+1), 'n_cats'))
            self.total_cats += self.datasets_cats[i]
               
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
            

        self.unify_prototype = nn.ParameterList([nn.Parameter(torch.zeros(n_cat, self.output_feat_dim),
                                requires_grad=True) for n_cat in self.datasets_cats])
        _= [trunc_normal_(proto, std=0.02) for proto in self.unify_prototype]
        


    @classmethod
    def from_config(cls, cfg, input_shape):

        return {
            'cfg': cfg,
            'neck': None,
            'auxiliary_head': None,
            'train_cfg': dict(),
            'test_cfg': dict(mode='whole'),
            'pretrained': cfg.get('pretrained', None),
            'init_cfg': None
        }
        

    def _init_decode_head(self, cfg):
        """Initialize ``decode_head``"""
        self.decode_head = build_sem_seg_head(cfg, 0)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(build_sem_seg_head(head_cfg))
            else:
                self.auxiliary_head = build_sem_seg_head(auxiliary_head)
        else:
            self.with_auxiliary_head = False

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas, dataset_ids):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        emb = self._decode_head_forward_test(x, img_metas)
        
        remap_logits = []
        uni_logits = []
        if self.training:
            for i in range(self.n_datasets):
                if not (dataset_ids == i).any():
                    continue
                logits = torch.einsum('bchw, nc -> bnhw', emb[dataset_ids == i], self.unify_prototype[i])
                remap_logits.append(logits)
            return {'logits':remap_logits}
        else:
            cats_unify_prototype = torch.cat([*self.unify_prototype], dim=0)
            if not isinstance(dataset_ids, int):
                for i in range(self.n_datasets):
                    if not (dataset_ids == i).any():
                        continue
                logits = torch.einsum('bchw, nc -> bnhw', emb[dataset_ids == i], self.unify_prototype[i])
                
                remap_logits.append(logits)
                uni_logit = torch.einsum('bchw, nc -> bnhw', emb[dataset_ids == i], cats_unify_prototype)
                uni_logits.append(uni_logit)
            else:
                logits = torch.einsum('bchw, nc -> bnhw', emb, self.unify_prototype[dataset_ids])
                remap_logits.append(logits)
                uni_logit = torch.einsum('bchw, nc -> bnhw', emb, cats_unify_prototype)
                uni_logits.append(uni_logit)
            
            
            
            return {'logits':remap_logits, 'uni_logits':uni_logits}
                

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward(self, img, img_metas, gt_semantic_seg, dataset_ids, return_loss=True, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        if self.training and return_loss:
            x = self.extract_feat(img)

            losses = dict()

            loss_decode = self._decode_head_forward_train(x, img_metas,
                                                        gt_semantic_seg)
            losses.update(loss_decode)

            if self.with_auxiliary_head:
                loss_aux = self._auxiliary_head_forward_train(
                    x, img_metas, gt_semantic_seg)
                losses.update(loss_aux)

            return losses
        else:
            return self.encode_decode(img, img_metas, dataset_ids)

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

