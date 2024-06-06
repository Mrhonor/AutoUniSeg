
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
import torch
import cv2
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
torch.set_printoptions(profile="full")
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

def find_specific_class(trainer_build_test_loader, cfg, model, target_id, target_uni_id, target_dataset_id):
    """
    Build a bipartite graph for unseen classes.

    Args:
        cfg (CfgNode): config.
        model (nn.Module): model.

    Returns:
        tuple: (graph, num_classes).
    """
    logger = logging.getLogger(__name__)
    # show_number = 10
    # target_dataset_id = 4
    # org_aux = net.aux_mode

    bipart_graph = model.get_bipart_graph()
    datasets_cats = []
    dataset_name = cfg.DATASETS.TRAIN[target_dataset_id]
    stuff_classes = MetadataCatalog.get(dataset_name).stuff_classes
    datasets_cats.append(len(stuff_classes))
    #  = [11]#cfg.DATASETS.DATASETS_CATS
    n_datasets = len(datasets_cats)
    ignore_index = cfg.DATASETS.IGNORE_LB
    callbacks = None
    
    target_bipart = []
    dataset_idx = 0

    data_loader = trainer_build_test_loader(cfg, dataset_name)
    logger.info(f"find in {data_loader}")
    metadata_uni = MetadataCatalog.get(
            "uni"
        )
    n_classes = datasets_cats[dataset_idx]
    num_unfiy_class = cfg.DATASETS.NUM_UNIFY_CLASS
    if num_unfiy_class < n_classes:
        num_unfiy_class = n_classes
    hist = torch.zeros(n_classes, num_unfiy_class).cuda()
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
                flag = True
                for lb in inputs:
                    if (lb["sem_seg"] == target_id).any():
                        flag = False
                        break
                if flag:
                    continue
                for i, x in enumerate(inputs):
                    if x["image"].shape[1] > 3000 or x["image"].shape[2] > 3000:
                        inputs[i]["image"] = F.interpolate(x["image"][None], size=(int(x["image"].shape[1]*0.5), int(x["image"].shape[2]*0.5)))[0]
                        inputs[i]['height'] = int(x["image"].shape[1]*0.5)
                        inputs[i]['width'] = int(x["image"].shape[2]*0.5)
                        # logger.info(x["image"].shape)

                dict.get(callbacks or {}, "before_inference", lambda: None)()
                outputs = model(inputs)
                dict.get(callbacks or {}, "after_inference", lambda: None)()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time

                start_eval_time = time.perf_counter()
                labels = [x["sem_seg"][None].cuda() for x in inputs]

                logits = [output["uni_logits"] for output in outputs]
                file_names = [x["file_name"] for x in inputs]
                images = [x["image"] for x in inputs]
                # image = inputs[0]["image"].permute(1, 2, 0)
                
                for name, lg, im in zip(file_names, logits, images):
                    lg = lg
                    # lb = lb.long()
                    # lb = F.interpolate(lb.unsqueeze(1).float(), size=(lg.shape[2], lg.shape[3]),
                    #         mode='nearest').squeeze(1).long()

                    probs = torch.softmax(lg, dim=0)
                    preds = torch.argmax(probs, dim=0)
                    flag = True
                    for tg in target_uni_id:
                        if (preds == tg).any():
                            flag = False
                            break
                    if flag:
                        continue
                    im = F.interpolate(im[None], size=(preds.shape[0], preds.shape[1]))[0]
                    image = im.permute(1, 2, 0)
                    visualizer = Visualizer(image, metadata_uni, instance_mode=ColorMode.IMAGE)
                    vis_output = visualizer.draw_sem_seg(
                        preds.cpu()
                    )
                    name = name.split('/')[-1].replace('.jpg', '.png')
                    if osp.exists(f"output/{dataset_name}/") == False:
                        os.makedirs(f"output/{dataset_name}/")
                    vis_output.save(f"output/{dataset_name}/{name}")
                    logger.info(f"save output/{dataset_name}/{name}")
                    
                    
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
        
  

