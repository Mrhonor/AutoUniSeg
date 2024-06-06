
import sys

sys.path.insert(0, '.')
import os
import os.path as osp


import logging
import argparse
import math

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import ExitStack, contextmanager


from detectron2.engine.hooks import HookBase
import datetime
from detectron2.utils.logger import log_every_n_seconds
from .MCMF_build_for_unseen import MinCostMaxFlow_Unseen
from detectron2.data import MetadataCatalog

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

def eval_for_mseg_datasets(trainer_build_test_loader, cfg, model):
    """
    Build a bipartite graph for unseen classes.

    Args:
        cfg (CfgNode): config.
        model (nn.Module): model.

    Returns:
        tuple: (graph, num_classes).
    """
    logger = logging.getLogger(__name__)

    # org_aux = net.aux_mode

    bipart_graph = model.get_bipart_graph()
    
    datasets_cats = cfg.DATASETS.DATASETS_CATS
    n_datasets = len(datasets_cats)
    ignore_index = cfg.DATASETS.IGNORE_LB
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
    target_bipart = []

    for dataset_idx, dataset_name in enumerate(cfg.DATASETS.TEST):
        data_loader = trainer_build_test_loader(cfg, dataset_name)
        meta = MetadataCatalog.get(dataset_name.replace("sem_seg_val", "mseg_sem_seg_val"))
        stuff_classes = meta.stuff_classes
        trainId_to_msegId = meta.trainId_to_msegId
        n_classes = len(stuff_classes)
        logger.info(trainId_to_msegId)
        lb_map = n_classes*torch.ones(512).cuda()
        for k,v in trainId_to_msegId.items():
            if v != 255:
                lb_map[k] = v
        logger.info(lb_map)
        # target_lb_map = np.arange(256).astype(np.uint8)
        # lookup_table = MetadataCatalog.get(dataset_name).thing_dataset_id_to_contiguous_id
        # for k, v in lookup_table.items():
        #     target_lb_map[k] = v
        
        hist = torch.zeros(n_classes+1, n_classes+1).cuda()
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
                    dict.get(callbacks or {}, "before_inference", lambda: None)()
                    outputs = model(inputs, dataset=1)
                    dict.get(callbacks or {}, "after_inference", lambda: None)()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    total_compute_time += time.perf_counter() - start_compute_time

                    start_eval_time = time.perf_counter()
                    labels = [x["sem_seg"][None].cuda() for x in inputs]

                    logits = [output["sem_seg"] for output in outputs]
                    
                    for lb, lg in zip(labels, logits):
                        lg = lg[None]
                        lb = lb.long()
                        lb = lb_map[lb].long().cpu().numpy()
                        # lb = F.interpolate(lb.unsqueeze(1).float(), size=(lg.shape[2], lg.shape[3]),
                        #         mode='nearest').squeeze(1).long()
                        # logger.info(f"lg:{lg.shape}")
                        probs = torch.softmax(lg, dim=1)
                        preds = torch.argmax(probs, dim=1)
                        preds = lb_map[preds].long()
                        
                        keep = lb != ignore_index
                        

                        hist += torch.tensor(np.bincount(
                            lb[keep] * (n_classes+1) + preds.cpu().numpy()[keep],
                            minlength=(n_classes+1) * (n_classes+1)
                        )).cuda().view(n_classes+1, n_classes+1) 
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
                ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
                miou = np.nanmean(ious.cpu().numpy())
                logger.info(
                    "dataset {} miou: {:.6f}, iou for each class:{})".format(
                        dataset_name, miou, ious
                    )
                )
                

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
   