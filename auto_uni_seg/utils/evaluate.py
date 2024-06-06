#!/usr/bin/python
# -*- encoding: utf-8 -*-


import sys

sys.path.insert(0, '.')
import os
import os.path as osp


import logging
import argparse
import math
from tabulate import tabulate

from tqdm import tqdm
import numpy as np
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import ExitStack, contextmanager

from detectron2.data import MetadataCatalog
from detectron2.engine.hooks import HookBase
import datetime
from detectron2.utils.logger import log_every_n_seconds

CITY_ID = 0
CAM_ID = 1
A2D2_ID = 2

def is_distributed():
    return torch.distributed.is_initialized()


def get_round_size(size, divisor=32):
    return [math.ceil(el / divisor) * divisor for el in size]

class MscEvalV0(object):

    def __init__(self, scales=(0.5, ), flip=False, ignore_label=255):
        self.scales = scales
        self.flip = flip
        self.ignore_label = ignore_label

    def __call__(self, net, dl, n_classes, dataset_id):
        ## evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for i, (imgs, label) in diter:
            N, _, H, W = label.shape
            label = label.squeeze(1).cuda()
            size = label.size()[-2:]
            probs = torch.zeros(
                    (N, n_classes, H, W), dtype=torch.float32).cuda().detach()

            for scale in self.scales:
                sH, sW = int(scale * H), int(scale * W)
                sH, sW = get_round_size((sH, sW))
                im_sc = F.interpolate(imgs, size=(sH, sW),
                        mode='bilinear', align_corners=True)

                im_sc = im_sc.cuda()
                
                logits = net(im_sc, dataset=dataset_id)[0]
                logits = F.interpolate(logits, size=size,
                        mode='bilinear', align_corners=True)
                probs += torch.softmax(logits, dim=1)
                if self.flip:
                    im_sc = torch.flip(im_sc, dims=(3, ))
                    logits = net(im_sc, dataset=dataset_id)[0]
                    logits = torch.flip(logits, dims=(3, ))
                    logits = F.interpolate(logits, size=size,
                            mode='bilinear', align_corners=True)
                    probs += torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            

            keep = label != self.ignore_label
            hist += torch.tensor(np.bincount(
                label.cpu().numpy()[keep.cpu().numpy()] * n_classes + preds.cpu().numpy()[keep.cpu().numpy()],
                minlength=n_classes ** 2
            )).cuda().view(n_classes, n_classes)
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        print(ious)
        miou = np.nanmean(ious.detach().cpu().numpy())
        return miou.item()

class MscEvalV0_Contrast(object):

    def __init__(self, configer, scales=(0.5, ), flip=False, ignore_label=255, ori_scales=False):
        self.configer = configer
        # self.num_unify_classes = self.configer.get('num_unify_classes')
        # self.class_Remaper = ClassRemap(configer=self.configer)
        self.scales = scales
        self.flip = flip
        self.ignore_label = ignore_label
        self.ori_scales = ori_scales

        # self.lb_map = torch.tensor(np.load('mapi_relabel.npy')).cuda()
        # print(self.lb_map)

    def __call__(self, net, dl, n_classes, dataset_id):
        # n_classes = 43
        ## evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        # hist = torch.zeros(118, 118).cuda().detach()
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for i, (imgs, label) in diter:
            N, _, H, W = label.shape

            label = label.squeeze(1).cuda()
            size = label.size()[-2:]
            # probs = torch.zeros(
            #         (N, self.num_unify_classes, H, W), dtype=torch.float32).cuda().detach()
            probs = None
            # probs = torch.zeros(
            #         (N, 150, H, W), dtype=torch.float32).cuda().detach()


            for scale in self.scales:

                
                sH, sW = int(scale * H), int(scale * W)
                sH, sW = get_round_size((sH, sW))
                im_sc = F.interpolate(imgs, size=(sH, sW),
                        mode='bilinear', align_corners=True)

                im_sc = im_sc.cuda()
                
                logits = net(im_sc, dataset=dataset_id)
                # logits = net(im_sc, dataset_id * torch.ones(N, dtype=torch.long))
                N, _, lH, lW = logits.shape
                if self.ori_scales:
                    logits = F.interpolate(logits, size=size,
                            mode='bilinear', align_corners=True)
                    if probs is None:
                        probs = torch.zeros(
                            (N, n_classes, H, W), dtype=torch.float32).cuda().detach()
                else:
                    label = F.interpolate(label.float().unsqueeze(1), size=(lH, lW),
                            mode='nearest').squeeze(1).long()
                    if probs is None:
                        probs = torch.zeros(
                            (N, n_classes, lH, lW), dtype=torch.float32).cuda().detach()
                    # label = F.interpolate()
                

                probs += torch.softmax(logits, dim=1)
                if self.flip:
                    im_sc = torch.flip(im_sc, dims=(3, ))
                    logits = net(im_sc, dataset=dataset_id)
                    logits = torch.flip(logits, dims=(3, ))
                    logits = F.interpolate(logits, size=size,
                            mode='bilinear', align_corners=True)
                    probs += torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            keep = label != self.ignore_label
            
            # print(np.max(label.cpu().numpy()[keep.cpu().numpy()]))

            hist += torch.tensor(np.bincount(
                label.cpu().numpy()[keep.cpu().numpy()] * n_classes + preds.cpu().numpy()[keep.cpu().numpy()],
                minlength=n_classes ** 2
            )).cuda().view(n_classes, n_classes)
            # hist += torch.tensor(np.bincount(
            #     label.cpu().numpy()[keep.cpu().numpy()] * 118 + preds.cpu().numpy()[keep.cpu().numpy()],
            #     minlength=118 ** 2
            # )).cuda().view(118, 118)
                
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        print(ious)
        miou = np.nanmean(ious.detach().cpu().numpy())
        return miou.item()

class MscEvalV0_unlabel(object):

    def __init__(self, configer, scales=(0.5, ), flip=False, ignore_label=255, ori_scales=False):
        self.configer = configer
        # self.num_unify_classes = self.configer.get('num_unify_classes')
        # self.class_Remaper = ClassRemap(configer=self.configer)
        self.scales = scales
        self.flip = flip
        self.ignore_label = ignore_label
        self.ori_scales = ori_scales

        # self.lb_map = torch.tensor(np.load('mapi_relabel.npy')).cuda()
        # print(self.lb_map)

    def __call__(self, net, dl, n_classes, dataset_id):
        # n_classes = 43
        ## evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        # hist = torch.zeros(118, 118).cuda().detach()
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for i, (imgs, label) in diter:
            N, _, H, W = label.shape

            label = label.squeeze(1).cuda()
            size = label.size()[-2:]
            # probs = torch.zeros(
            #         (N, self.num_unify_classes, H, W), dtype=torch.float32).cuda().detach()
            probs = None
            # probs = torch.zeros(
            #         (N, 150, H, W), dtype=torch.float32).cuda().detach()


            for scale in self.scales:

                
                sH, sW = int(scale * H), int(scale * W)
                sH, sW = get_round_size((sH, sW))
                im_sc = F.interpolate(imgs, size=(sH, sW),
                        mode='bilinear', align_corners=True)

                im_sc = im_sc.cuda()
                
                ori_logits = net(im_sc, dataset=dataset_id)
                # logits = net(im_sc, dataset_id * torch.ones(N, dtype=torch.long))
                N, D, lH, lW = ori_logits.shape
                logits = ori_logits[:, :n_classes, :, :]
                if self.ori_scales:
                    logits = F.interpolate(logits, size=size,
                            mode='bilinear', align_corners=True)
                    if probs is None:
                        probs = torch.zeros(
                            (N, n_classes, H, W), dtype=torch.float32).cuda().detach()
                else:
                    label = F.interpolate(label.float().unsqueeze(1), size=(lH, lW),
                            mode='nearest').squeeze(1).long()
                    if probs is None:
                        probs = torch.zeros(
                            (N, n_classes, lH, lW), dtype=torch.float32).cuda().detach()
                    # label = F.interpolate()
                

                probs += torch.softmax(logits, dim=1)
                if self.flip:
                    im_sc = torch.flip(im_sc, dims=(3, ))
                    logits = net(im_sc, dataset=dataset_id)
                    logits = torch.flip(logits, dims=(3, ))
                    logits = F.interpolate(logits, size=size,
                            mode='bilinear', align_corners=True)
                    probs += torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            keep = label != self.ignore_label
            
            # print(np.max(label.cpu().numpy()[keep.cpu().numpy()]))

            hist += torch.tensor(np.bincount(
                label.cpu().numpy()[keep.cpu().numpy()] * n_classes + preds.cpu().numpy()[keep.cpu().numpy()],
                minlength=n_classes ** 2
            )).cuda().view(n_classes, n_classes)
            # hist += torch.tensor(np.bincount(
            #     label.cpu().numpy()[keep.cpu().numpy()] * 118 + preds.cpu().numpy()[keep.cpu().numpy()],
            #     minlength=118 ** 2
            # )).cuda().view(118, 118)
                
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        print(ious)
        miou = np.nanmean(ious.detach().cpu().numpy())
        return miou.item()

class MscEvalV0_AutoLink(object):

    def __init__(self, configer, scales=(0.5, ), flip=False, ignore_label=255):
        self.configer = configer
        self.n_datasets = self.configer.get('n_datasets')
        self.scales = scales
        self.flip = flip
        self.ignore_label = ignore_label

    def __call__(self, net, dl, n_classes, dataset_id):
        ## evaluate
        # hist = torch.zeros(n_classes, n_classes).cuda().detach()
        datasets_remap = []
        # hist = torch.zeros(n_classes, n_classes).cuda().detach()
        # if dist.is_initialized() and dist.get_rank() != 0:
        diter = enumerate(dl)
        # else:
        #     diter = enumerate(tqdm(dl))
        for i, (imgs, label) in diter:
            N, _, H, W = label.shape

            label = label.squeeze(1).cuda()
            size = label.size()[-2:]
            # print(size)

            scale = self.scales[0]
            sH, sW = int(scale * H), int(scale * W)
            sH, sW = get_round_size((sH, sW))
            im_sc = F.interpolate(imgs, size=(sH, sW),
                    mode='bilinear', align_corners=True)

            im_sc = im_sc.cuda()
            all_logits = net(im_sc)
            for index in range(0, self.n_datasets):
                if index == dataset_id:
                    if len(datasets_remap) <= index:
                        datasets_remap.append(torch.eye(n_classes).cuda().detach())
                     
                    continue
                
                n_cats = self.configer.get('dataset'+str(index+1), 'n_cats')
                this_data_hist = torch.zeros(n_classes, n_cats).cuda().detach()

                logits = F.interpolate(all_logits[index], size=size,
                        mode='bilinear', align_corners=True)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                keep = label != self.ignore_label
        
                this_data_hist = torch.tensor(np.bincount(
                    label.cpu().numpy()[keep.cpu().numpy()] * n_cats + preds.cpu().numpy()[keep.cpu().numpy()],
                    minlength=n_classes * n_cats
                )).cuda().view(n_classes, n_cats)
                if len(datasets_remap) <= index:
                    datasets_remap.append(this_data_hist)
                else:
                    datasets_remap[index] += this_data_hist
                         
        return [torch.argmax(hist, dim=1) for hist in datasets_remap]
        # if dist.is_initialized():
        #     dist.all_reduce(hist, dist.ReduceOp.SUM)
            
        # ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        # print(ious)
        # miou = np.nanmean(ious.detach().cpu().numpy())
        # return miou.item()


@torch.no_grad()
def find_unuse_label(configer, net, dl, n_classes, dataset_id):
        ## evaluate
    # hist = torch.zeros(n_classes, n_classes).cuda().detach()
    # datasets_remap = []
    ignore_label = 255
    n_datasets = configer.get("n_datasets")
    total_cats = 0
    net.aux_mode = 'train'
    net.eval()
    unify_prototype = net.unify_prototype
    # print(unify_prototype.shape)
    bipart_graph = net.bipartite_graphs
    
    for i in range(0, n_datasets):
        total_cats += configer.get("dataset"+str(i+1), "n_cats")
    total_cats = int(total_cats * configer.get('GNN', 'unify_ratio'))

    hist = torch.zeros(n_classes, total_cats).cuda().detach()
    if dist.is_initialized() and dist.get_rank() != 0:
        diter = enumerate(dl)
    else:
        diter = enumerate(tqdm(dl))
        
    
    with torch.no_grad():
        for i, (imgs, label) in diter:
            N, _, H, W = label.shape
            if H > 2048 or W > 2048:
                H = 2048
                W = 2048
                

            label = label.squeeze(1).cuda()
            size = label.shape[-2:]

            im_sc = F.interpolate(imgs, size=(H, W),
                    mode='bilinear', align_corners=True)

            im_sc = im_sc.cuda()
            
            emb = net(im_sc, dataset=dataset_id)
        
            logits = torch.einsum('bchw, nc -> bnhw', emb['seg'], unify_prototype)

            logits = F.interpolate(logits, size=size,
                    mode='bilinear', align_corners=True)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            keep = label != ignore_label

            hist += torch.tensor(np.bincount(
                label.cpu().numpy()[keep.cpu().numpy()] * total_cats + preds.cpu().numpy()[keep.cpu().numpy()],
                minlength=n_classes * total_cats
            )).cuda().view(n_classes, total_cats)

    max_value, max_index = torch.max(bipart_graph[dataset_id], dim=0)
    # print(max_value)
    n_cat = configer.get(f'dataset{dataset_id+1}', 'n_cats')
    
    # torch.set_printoptions(profile="full")
    # print(hist)

    buckets = {}
    for index, j in enumerate(max_index):
        if max_value[index] == 0:
            continue
        
        if int(j) not in buckets:
            buckets[int(j)] = [index]
        else:
            buckets[int(j)].append(index)

    for index in range(0, n_cat):
        if index not in buckets:
            print('index not in buckets:', index)
            buckets[index] = []

    for index, val in buckets.items():
        total_num = 0
        for i in val:
            total_num += hist[index][i]
        new_val = []
        if total_num != 0:
            for i in val:
                rate = hist[index][i] / total_num
                if rate > 1e-5:
                    new_val.append(i)
        
        buckets[index] = new_val

    net.train()
    return buckets 


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

class iter_info_hook(HookBase):
    @torch.no_grad()
    def before_step(self):
        if is_distributed():
            model = self.trainer.model.module
        else:
            model = self.trainer.model
        model.iters = self.trainer.iter


def print_bipartite(datasets_cats, n_datasets, bi_graphs, total_cats, datasets_name):
    
    logger = logging.getLogger(__name__)

    
    total_buckets = [[] for _ in range(total_cats)]
    for i in range(0, n_datasets):
        meta = MetadataCatalog.get(datasets_name[i])
        lb_name = meta.stuff_classes
        max_value, max_index = torch.max(bi_graphs[i], dim=0)
        n_cat = datasets_cats[i]
        
        buckets = {}
        for index, j in enumerate(max_index):
            if max_value[index] < 1e-4:
               continue
            
            if int(j) not in buckets:
                buckets[int(j)] = [index]
                
            else:
                buckets[int(j)].append(index)
            
            total_buckets[index].append(lb_name[int(j)])
            
        logger.info("dataset {}:".format(datasets_name[i]))    
    
        for index in range(0, n_cat):
            if index not in buckets:
                buckets[index] = []
            logger.info("\"{}\": {}".format(lb_name[index], buckets[index]))    
       
    for index in range(0, total_cats):
        logger.info("\"{}\": {}".format(index, total_buckets[index]))
    
    return 

class find_unuse_hook(HookBase):
    def after_step(self):
        if is_distributed():
            model = self.trainer.model.module
        else:
            model = self.trainer.model
        if self.trainer.iter > self.trainer.cfg.MODEL.GNN.FINETUNE_STAGE1_ITERS and int(model.finetune_stage) == 1:
            logger = logging.getLogger(__name__)
            logger.info("find_unuse_hook")
            cfg = self.trainer.cfg
            model.finetune_stage = torch.zeros(1)
            bipart_graph = model.get_bipart_graph()
            callbacks = None
            ignore_label = 255
            datasets_cats = self.trainer.cfg.DATASETS.DATASETS_CATS
            n_datasets = len(datasets_cats)
            ignore_index = self.trainer.cfg.DATASETS.IGNORE_LB
            total_cats = 0
            for i in range(0, n_datasets):
                total_cats += datasets_cats[i]
            init_adj_path = cfg.MODEL.GNN.INIT_ADJ_PATH
            if init_adj_path != None:
                init_adj = torch.load(init_adj_path)
                num_unify_class = init_adj.shape[1]
            else:
                num_unify_class = cfg.DATASETS.NUM_UNIFY_CLASS
            datasets_name = self.trainer.cfg.DATASETS.TRAIN
            # dls = get_data_loader(configer, aux_mode='train', distributed=is_dist, stage=2)
            print_bipartite(datasets_cats, n_datasets, bipart_graph, total_cats, datasets_name)

            loaded_map = {}
            for dataset_idx, dataset_name in enumerate(self.trainer.cfg.DATASETS.EVAL):
                logger.info("evaluating dataset {}:".format(i+1))    

                data_loader = self.trainer.build_test_loader(self.trainer.cfg, dataset_name)
        
                n_classes = datasets_cats[dataset_idx]
                hist = torch.zeros(n_classes, num_unify_class).cuda()    
                
                with torch.no_grad():
                    total = len(data_loader)
                    num_warmup = min(5, total - 1)
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0
                    with ExitStack() as stack:
                        if isinstance(model, nn.Module):
                            stack.enter_context(inference_context(model))
                        stack.enter_context(torch.no_grad())

                        start_data_time = time.perf_counter()
                        dict.get(callbacks or {}, "on_start", lambda: None)()
                        for idx, inputs in enumerate(data_loader):
                            total_data_time += time.perf_counter() - start_data_time
                            if idx == num_warmup:
                                start_time = time.perf_counter()
                                total_data_time = 0
                                total_compute_time = 0
                                total_eval_time = 0
                            for x in inputs:
                                im = x["image"]
                                if im.shape[-2] > 2200 or im.shape[-1] > 2200:
                                    x["image"] = F.interpolate(im[None].float(), size=(int(im.shape[-2]*0.5), int(im.shape[-1]*0.5)), mode='bilinear', align_corners=True).squeeze(0)
                                    x["sem_seg"] = F.interpolate(x["sem_seg"].float()[None][None], size=(int(im.shape[-2]*0.5), int(im.shape[-1]*0.5)), mode='nearest').squeeze().long()
                                    x["height"] = int(x["height"]*0.5)
                                    x["width"] = int(x["width"]*0.5)
                            start_compute_time = time.perf_counter()
                            dict.get(callbacks or {}, "before_inference", lambda: None)()
                            outputs = model(inputs)
                            dict.get(callbacks or {}, "after_inference", lambda: None)()
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            total_compute_time += time.perf_counter() - start_compute_time

                            start_eval_time = time.perf_counter()
                            labels = [x["sem_seg"][None].cuda() for x in inputs]

                            logits = [output["uni_logits"][None] for output in outputs]
                            
                            for lb, lg in zip(labels, logits):
                                # logit
                                lb = F.interpolate(lb.unsqueeze(1).float(), size=(lg.shape[2], lg.shape[3]),
                                        mode='nearest').squeeze(1).long()

                                probs = torch.softmax(lg, dim=1)
                                preds = torch.argmax(probs, dim=1)
                                                   
                                keep = lb != ignore_label

                                hist += torch.tensor(np.bincount(
                                    lb.cpu().numpy()[keep.cpu().numpy()] * num_unify_class + preds.cpu().numpy()[keep.cpu().numpy()],
                                    minlength=n_classes * num_unify_class
                                )).cuda().view(n_classes, num_unify_class) 
                            total_eval_time += time.perf_counter() - start_eval_time

                            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                            data_seconds_per_iter = total_data_time / iters_after_start
                            compute_seconds_per_iter = total_compute_time / iters_after_start
                            eval_seconds_per_iter = total_eval_time / iters_after_start
                            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                                log_every_n_seconds(
                                    logging.INFO,
                                    (
                                        f"Inference done {idx + 1}/{total}. "
                                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                                        f"ETA={eta}"
                                    ),
                                    n=5,
                                )
                            start_data_time = time.perf_counter()
                        dict.get(callbacks or {}, "on_end", lambda: None)()

                    # Measure the time only for this worker (before the synchronization barrier)
                    total_time = time.perf_counter() - start_time
                    total_time_str = str(datetime.timedelta(seconds=total_time))
                    # NOTE this format is parsed by grep
                    logger.info(
                        "Total inference time: {} ({:.6f} s / iter per device)".format(
                            total_time_str, total_time / (total - num_warmup)
                        )
                    )
                    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
                    logger.info(
                        "Total inference pure compute time: {} ({:.6f} s / iter per device)".format(
                            total_compute_time_str, total_compute_time / (total - num_warmup)
                        )
                    )
                
                max_value, max_index = torch.max(bipart_graph[dataset_idx], dim=0)
                # print(max_value)
                
                # torch.set_printoptions(profile="full")
                # print(hist)

                buckets = {}
                for index, j in enumerate(max_index):
                    if max_value[index] == 0:
                        continue
                    
                    if int(j) not in buckets:
                        buckets[int(j)] = [index]
                    else:
                        buckets[int(j)].append(index)

                for index in range(0, n_classes):
                    if index not in buckets:
                        logger.info(f'index not in buckets: {index}')
                        buckets[index] = []

                for index, val in buckets.items():
                    total_num = 0
                    for i in val:
                        total_num += hist[index][i]
                    new_val = []
                    if total_num != 0:
                        for i in val:
                            rate = hist[index][i] / total_num
                            if rate > 0.01:
                                # new_val.append([i, rate])
                                new_val.append(i)
                    else:
                        for i in val:
                            # new_val.append([i, 0])
                            new_val.append(i)
                    
                    buckets[index] = new_val
                    
                    
                for index in range(0, n_classes):
                    if index not in buckets:
                        logger.info(f'index not in buckets: {index}')
                        buckets[index] = []
                    logger.info("\"{}\": {}".format(index, buckets[index]))    
                
                loaded_map[f'dataset{dataset_idx}'] = buckets

            bi_graphs = []
            for dataset_id in range(0, n_datasets):
                n_cats = datasets_cats[dataset_id]
                this_bi_graph = torch.zeros(n_cats, num_unify_class)
                for key, val in loaded_map['dataset'+str(dataset_id)].items():
                    this_bi_graph[int(key)][val] = 1
                    
                bi_graphs.append(this_bi_graph.cuda())

            logger.info("finished find unuse hook")    
            model.set_bipartite_graphs(bi_graphs) 

class eval_link_hook(HookBase):
    @torch.no_grad()
    def after_step(self):
        if is_distributed():
            model = self.trainer.model.module
        else:
            model = self.trainer.model
        if self.trainer.iter % 5000 == 0 and model.train_seg_or_gnn==model.GNN and not model.init_gnn_stage:
            logger = logging.getLogger(__name__)
            logger.info(f"eval link at iteration {self.trainer.iter}!")
            # org_aux = net.aux_mode
            cfg = self.trainer.cfg
            # model = self.trainer.model
            bipart_graph = model.get_bipart_graph()
            ignore_label = 255
            datasets_cats = self.trainer.cfg.DATASETS.DATASETS_CATS
            n_datasets = len(datasets_cats)
            ignore_index = self.trainer.cfg.DATASETS.IGNORE_LB
            total_cats = 0
            callbacks = None
            for i in range(0, n_datasets):
                total_cats += datasets_cats[i]
            init_adj_path = cfg.MODEL.GNN.INIT_ADJ_PATH
            if init_adj_path != None:
                init_adj = torch.load(init_adj_path)
                num_unify_class = init_adj.shape[1]
            else:
                num_unify_class = cfg.DATASETS.NUM_UNIFY_CLASS
            heads, mious, target_bipart = [], [], []

            for dataset_idx, dataset_name in enumerate(self.trainer.cfg.DATASETS.EVAL):
                data_loader = self.trainer.build_test_loader(self.trainer.cfg, dataset_name)
        
                n_classes = datasets_cats[dataset_idx]
                hist = torch.zeros(n_classes, num_unify_class).cuda()
                # hist_origin = torch.zeros(n_classes, n_classes).cuda().detach()    
                
                with torch.no_grad():
                    total = len(data_loader)
                    num_warmup = min(5, total - 1)
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0
                    with ExitStack() as stack:
                        if isinstance(model, nn.Module):
                            stack.enter_context(inference_context(model))
                        stack.enter_context(torch.no_grad())

                        start_data_time = time.perf_counter()
                        dict.get(callbacks or {}, "on_start", lambda: None)()
                        for idx, inputs in enumerate(data_loader):
                            total_data_time += time.perf_counter() - start_data_time
                            if idx == num_warmup:
                                start_time = time.perf_counter()
                                total_data_time = 0
                                total_compute_time = 0
                                total_eval_time = 0

                            start_compute_time = time.perf_counter()
                            for x in inputs:
                                im = x["image"]
                                if im.shape[-2] > 3500 or im.shape[-1] > 3500:
                                    x["image"] = F.interpolate(im[None], size=(int(im.shape[-2]*0.75), int(im.shape[-1]*0.75)), mode='bilinear', align_corners=True).squeeze(0)
                                    x["sem_seg"] = F.interpolate(x["sem_seg"].float()[None][None], size=(int(im.shape[-2]*0.75), int(im.shape[-1]*0.75)), mode='nearest').squeeze().long()
                            dict.get(callbacks or {}, "before_inference", lambda: None)()
                            outputs = model(inputs)
                            dict.get(callbacks or {}, "after_inference", lambda: None)()
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            total_compute_time += time.perf_counter() - start_compute_time

                            start_eval_time = time.perf_counter()
                            labels = [x["sem_seg"][None].cuda() for x in inputs]

                            logits = [output["uni_logits"] for output in outputs]
                            
                            for lb, lg in zip(labels, logits):
                                # logger.info(f"lb:{lb.shape}, lg:{lg.shape}")

                                lb = F.interpolate(lb.unsqueeze(1).float(), size=(lg.shape[2], lg.shape[3]),
                                        mode='nearest').squeeze(1).long()

                                probs = torch.softmax(lg, dim=1)
                                preds = torch.argmax(probs, dim=1)
                                logger = logging.getLogger(__name__) 
                                keep = lb != ignore_label
                                # logger.info(f"lb:{lb.shape}, keep:{keep.shape}, preds:{preds.shape}")

                                hist += torch.tensor(np.bincount(
                                    lb.cpu().numpy()[keep.cpu().numpy()] * num_unify_class + preds.cpu().numpy()[keep.cpu().numpy()],
                                    minlength=n_classes * num_unify_class
                                )).cuda().view(n_classes, num_unify_class) 
                            total_eval_time += time.perf_counter() - start_eval_time

                            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                            data_seconds_per_iter = total_data_time / iters_after_start
                            compute_seconds_per_iter = total_compute_time / iters_after_start
                            eval_seconds_per_iter = total_eval_time / iters_after_start
                            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                                log_every_n_seconds(
                                    logging.INFO,
                                    (
                                        f"Inference done {idx + 1}/{total}. "
                                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                                        f"ETA={eta}"
                                    ),
                                    n=5,
                                )
                            start_data_time = time.perf_counter()
                        dict.get(callbacks or {}, "on_end", lambda: None)()

                    # Measure the time only for this worker (before the synchronization barrier)
                    total_time = time.perf_counter() - start_time
                    total_time_str = str(datetime.timedelta(seconds=total_time))
                    # NOTE this format is parsed by grep
                    logger.info(
                        "Total inference time: {} ({:.6f} s / iter per device)".format(
                            total_time_str, total_time / (total - num_warmup)
                        )
                    )
                    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
                    logger.info(
                        "Total inference pure compute time: {} ({:.6f} s / iter per device)".format(
                            total_compute_time_str, total_compute_time / (total - num_warmup)
                        )
                    )                    
                     

                max_value, max_index = torch.max(bipart_graph[dataset_idx], dim=0)
                
                # torch.set_printoptions(profile="full")
                # print(hist)

                bipart = ignore_index * torch.ones_like(bipart_graph[dataset_idx])
                buckets = {}
                for index, j in enumerate(max_index):
                    if max_value[index] == 0:
                        continue
                    
                    if int(j) not in buckets:
                        buckets[int(j)] = [index]
                    else:
                        buckets[int(j)].append(index)

                for index in range(0, n_classes):
                    if index not in buckets:
                        print('index not in buckets:', index)
                        buckets[index] = []

                for index, val in buckets.items():
                    total_num = 0
                    # sum_num = torch.sum(hist[index])
                    for v in val:
                        total_num += hist[index][v]
                    # remove_val = []
                    # affirm_val = []
                    if total_num != 0:
                        for v in val:
                            rate = hist[index][v] / total_num
                            if rate < 1e-2:
                                bipart[index][v] = 0
                                # remove_val.append(i)
                            elif rate > 0.5:
                                bipart[index][v] = 1
                                # affirm_val.append(i)
                    
                    # buckets[index] = new_val
                target_bipart.append(bipart)

            # net.train()
            # target_bipart.cat(target_bipart, dim=0)
            model.set_target_bipart(target_bipart)
            

def print_unify_label_space(build_test_loader, model, cfg):

    logger = logging.getLogger(__name__)
    
    bipart_graph = model.get_bipart_graph()
    callbacks = None
    ignore_label = 255
    datasets_cats = cfg.DATASETS.DATASETS_CATS
    n_datasets = len(datasets_cats)
    ignore_index = cfg.DATASETS.IGNORE_LB
    total_cats = 0
    for i in range(0, n_datasets):
        total_cats += datasets_cats[i]
    num_unify_class = cfg.DATASETS.NUM_UNIFY_CLASS
    datasets_name = cfg.DATASETS.TRAIN
    # dls = get_data_loader(configer, aux_mode='train', distributed=is_dist, stage=2)
    print_bipartite(datasets_cats, n_datasets, bipart_graph, total_cats, datasets_name)

    loaded_map = {}
    for dataset_idx, dataset_name in enumerate(cfg.DATASETS.EVAL):
        logger.info("evaluating dataset {}:".format(i+1))    

        data_loader = build_test_loader(cfg, dataset_name)

        n_classes = datasets_cats[dataset_idx]
        hist = torch.zeros(n_classes, num_unify_class).cuda()    
        
        with torch.no_grad():
            total = len(data_loader)
            num_warmup = min(5, total - 1)
            start_time = time.perf_counter()
            total_data_time = 0
            total_compute_time = 0
            total_eval_time = 0
            with ExitStack() as stack:
                if isinstance(model, nn.Module):
                    stack.enter_context(inference_context(model))
                stack.enter_context(torch.no_grad())

                start_data_time = time.perf_counter()
                dict.get(callbacks or {}, "on_start", lambda: None)()
                for idx, inputs in enumerate(data_loader):
                    total_data_time += time.perf_counter() - start_data_time
                    if idx == num_warmup:
                        start_time = time.perf_counter()
                        total_data_time = 0
                        total_compute_time = 0
                        total_eval_time = 0

                    start_compute_time = time.perf_counter()
                    dict.get(callbacks or {}, "before_inference", lambda: None)()
                    outputs = model(inputs)
                    dict.get(callbacks or {}, "after_inference", lambda: None)()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    total_compute_time += time.perf_counter() - start_compute_time

                    start_eval_time = time.perf_counter()
                    labels = [x["sem_seg"][None].cuda() for x in inputs]

                    logits = [output["uni_logits"] for output in outputs]
                    
                    for lb, lg in zip(labels, logits):
                        # logger.info(lb.shape)
                        # logger.info(lg.shape)
                        lb = F.interpolate(lb.unsqueeze(1).float(), size=(lg.shape[2], lg.shape[3]),
                                mode='nearest').squeeze(1).long()

                        probs = torch.softmax(lg, dim=1)
                        preds = torch.argmax(probs, dim=1)
                                            
                        keep = lb != ignore_label

                        hist += torch.tensor(np.bincount(
                            lb.cpu().numpy()[keep.cpu().numpy()] * num_unify_class + preds.cpu().numpy()[keep.cpu().numpy()],
                            minlength=n_classes * num_unify_class
                        )).cuda().view(n_classes, num_unify_class) 
                    total_eval_time += time.perf_counter() - start_eval_time

                    iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                    data_seconds_per_iter = total_data_time / iters_after_start
                    compute_seconds_per_iter = total_compute_time / iters_after_start
                    eval_seconds_per_iter = total_eval_time / iters_after_start
                    total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                    if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                        eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                        log_every_n_seconds(
                            logging.INFO,
                            (
                                f"Inference done {idx + 1}/{total}. "
                                f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                                f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                                f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                                f"Total: {total_seconds_per_iter:.4f} s/iter. "
                                f"ETA={eta}"
                            ),
                            n=5,
                        )
                    start_data_time = time.perf_counter()
                dict.get(callbacks or {}, "on_end", lambda: None)()

            # Measure the time only for this worker (before the synchronization barrier)
            total_time = time.perf_counter() - start_time
            total_time_str = str(datetime.timedelta(seconds=total_time))
            # NOTE this format is parsed by grep
            logger.info(
                "Total inference time: {} ({:.6f} s / iter per device)".format(
                    total_time_str, total_time / (total - num_warmup)
                )
            )
            total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
            logger.info(
                "Total inference pure compute time: {} ({:.6f} s / iter per device)".format(
                    total_compute_time_str, total_compute_time / (total - num_warmup)
                )
            )

        max_value, max_index = torch.max(bipart_graph[dataset_idx], dim=0)
        # print(max_value)
        
        # torch.set_printoptions(profile="full")
        # print(hist)

        buckets = {}
        for index, j in enumerate(max_index):
            if max_value[index] == 0:
                continue
            
            if int(j) not in buckets:
                buckets[int(j)] = [index]
            else:
                buckets[int(j)].append(index)

        for index in range(0, n_classes):
            if index not in buckets:
                logger.info(f'index not in buckets: {index}')
                buckets[index] = []

        for index, val in buckets.items():
            total_num = 0
            for i in val:
                total_num += hist[index][i]
            new_val = []
            if total_num != 0:
                for i in val:
                    rate = hist[index][i] / total_num
                    if rate > 0.001:
                        new_val.append([i, rate])
                        # new_val.append(i)
            else:
                for i in val:
                    new_val.append([i, 0])
                    # new_val.append(i)
            
            buckets[index] = new_val
            
            
            for index in range(0, n_classes):
                if index not in buckets:
                    buckets[index] = []
                print("\"{}\": {}".format(index, buckets[index]))    
            
            loaded_map[f'dataset{dataset_idx}'] = buckets
    #     break
    # n_datasets = 1
    bi_graphs = []
    for dataset_id in range(0, n_datasets):
        n_cats = datasets_cats[dataset_id]
        this_bi_graph = torch.zeros(n_cats, num_unify_class)
        for key, val in loaded_map['dataset'+str(dataset_id)].items():
            for idx, v in val:
                this_bi_graph[int(key)][int(idx)] = v
            
        bi_graphs.append(this_bi_graph.cuda())

            
    
    datasets = ['city', 'mapi', 'sun', 'bdd', 'idd', 'ade', 'coco']
    city_lb = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
    mapi_lb = ["Bird", "Ground Animal", "Curb", "Fence", "Guard Rail", "Barrier", "Wall", "Bike Lane", "Crosswalk - Plain", "Curb Cut", "Parking", "Pedestrian Area", "Rail Track", "Road", "Service Lane", "Sidewalk", "Bridge", "Building", "Tunnel", "Person", "Bicyclist", "Motorcyclist", "Other Rider", "Lane Marking - Crosswalk", "Lane Marking - General", "Mountain", "Sand", "Sky", "Snow", "Terrain", "Vegetation", "Water", "Banner", "Bench", "Bike Rack", "Billboard", "Catch Basin", "CCTV Camera", "Fire Hydrant", "Junction Box", "Manhole", "Phone Booth", "Pothole", "Street Light", "Pole", "Traffic Sign Frame", "Utility Pole", "Traffic Light", "Traffic Sign (Back)", "Traffic Sign (Front)", "Trash Can", "Bicycle", "Boat", "Bus", "Car", "Caravan", "Motorcycle", "On Rails", "Other Vehicle", "Trailer", "Truck", "Wheeled Slow", "Car Mount", "Ego Vehicle"]
    sun_lb = [ "bag", "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter", "blinds", "desk", "shelves", "curtain", "dresser", "pillow", "mirror", "floor mat", "clothes", "ceiling", "books", "refridgerator", "television", "paper", "towel", "shower curtain", "box", "whiteboard", "person", "night stand", "toilet", "sink", "lamp", "bathtub"]
    bdd_lb = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
    idd_lb = ["road", "drivable fallback or parking", "sidewalk", "non-drivable fallback or rail track", "person or animal", "out of roi or rider", "motorcycle", "bicycle", "autorickshaw", "car", "truck", "bus", "trailer or caravan or vehicle fallback", "curb", "wall", "fence", "guard rail", "billboard", "traffic sign", "traffic light", "polegroup or pole", "obs-str-bar-fallback", "building", "tunnel or bridge", "vegetation", "sky or fallback background"]
    ade_lb = ['flag', 'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock']
    coco_lb = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "banner", "blanket", "bridge", "cardboard", "counter", "curtain", "door-stuff", "floor-wood", "flower", "fruit", "gravel", "house", "light", "mirror-stuff", "net", "pillow", "platform", "playingfield", "railroad", "river", "road", "roof", "sand", "sea", "shelf", "snow", "stairs", "tent", "towel", "wall-brick", "wall-stone", "wall-tile", "wall-wood", "water-other", "window-blind", "window-other", "tree-merged", "fence-merged", "ceiling-merged", "sky-other-merged", "cabinet-merged", "table-merged", "floor-other-merged", "pavement-merged", "mountain-merged", "grass-merged", "dirt-merged", "paper-merged", "food-other-merged", "building-other-merged", "rock-merged", "wall-other-merged", "rug-merged"]

    for idx in range(num_unify_class):
        maps_name = f"{idx} : "
        for set_id in range(n_datasets):
            map_id = torch.argmax(bi_graphs[set_id][:,idx])
            if bi_graphs[set_id][map_id][idx] == 0:
                continue
            this_name = eval(datasets[set_id]+'_lb')[map_id]
            maps_name += f"{datasets[set_id]}: {this_name} ({bi_graphs[set_id][map_id][idx]}); "
        logger.info(maps_name)

    # model.set_bipartite_graphs(bi_graphs) 