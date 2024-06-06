
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

import torch
import cv2
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

def build_bipartite_graph_for_unseen(trainer_build_test_loader, cfg, model):
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

    datasets_cats = []
    for dataset_name in cfg.DATASETS.TRAIN:
        stuff_classes = MetadataCatalog.get(dataset_name).stuff_classes
        datasets_cats.append(len(stuff_classes))
    #  = [11]#cfg.DATASETS.DATASETS_CATS
    n_datasets = len(datasets_cats)
    ignore_index = cfg.DATASETS.IGNORE_LB
    callbacks = None
    
    target_bipart = []

    for dataset_idx, dataset_name in enumerate(cfg.DATASETS.EVAL):
        data_loader = trainer_build_test_loader(cfg, dataset_name)

        n_classes = datasets_cats[dataset_idx]
        init_adj_path = cfg.MODEL.GNN.INIT_ADJ_PATH
        if init_adj_path != None:
            init_adj = torch.load(init_adj_path)
            num_unify_class = init_adj.shape[1]
        else:
            num_unify_class = cfg.DATASETS.NUM_UNIFY_CLASS
        

        if num_unify_class < n_classes:
            num_unify_class = n_classes
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
                        lg = lg[None]
                        lb = lb.long()
                        # lb = F.interpolate(lb.unsqueeze(1).float(), size=(lg.shape[2], lg.shape[3]),
                        #         mode='nearest').squeeze(1).long()

                        probs = torch.softmax(lg, dim=1)
                        preds = torch.argmax(probs, dim=1)
                        logger = logging.getLogger(__name__) 
                        keep = lb != ignore_index

                        
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
            
        logger.info(hist)
        
        cost_matrix = torch.zeros(num_unify_class, n_classes)
        for uni_class in range(num_unify_class):
            total_pres = torch.sum(hist[:, uni_class])
            for set_class in range(n_classes):
                total_target = torch.sum(hist[set_class])
                if total_target == 0:
                    cost_matrix[uni_class, set_class] = 1
                    continue
                iou = hist[set_class, uni_class] / (total_pres + total_target - hist[set_class, uni_class])
                cost_matrix[uni_class, set_class] = 1 - iou
        logger.info(cost_matrix.shape)
        mcmf = MinCostMaxFlow_Unseen()
        src, target = mcmf(cost_matrix) 
        
        # torch.set_printoptions(profile="full")
        # print(hist)
                

        bipart = datasets_cats[dataset_idx] * torch.ones(448)
        for s, t in zip(src, target):
            bipart[s] = t
            # buckets[index] = new_val
        logger.info(bipart)
        target_bipart.append(bipart.cuda())

    # net.train()
    # target_bipart.cat(target_bipart, dim=0)
    model.set_dataset_adapter(target_bipart)
    # return graph, num_classes

from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

def build_bipartite_graph_for_unseen_for_manually(trainer_build_test_loader, cfg, model):
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
    callbacks = None
    init_adj_path = cfg.MODEL.GNN.INIT_ADJ_PATH
    if init_adj_path != None:
        init_adj = torch.load(init_adj_path)
        num_unify_class = init_adj.shape[1]
    else:
        num_unify_class = cfg.DATASETS.NUM_UNIFY_CLASS
    target_bipart = []
    image_list = {}
    for i in range(num_unify_class):
        image_list[i] = []
        
    for dataset_idx, dataset_name in enumerate(cfg.DATASETS.TRAIN):
        data_loader = trainer_build_test_loader(cfg, dataset_name)
        metadata = MetadataCatalog.get(
            'uni' 
        )
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
                    dict.get(callbacks or {}, "before_inference", lambda: None)()
                    outputs = model(inputs)
                    dict.get(callbacks or {}, "after_inference", lambda: None)()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    total_compute_time += time.perf_counter() - start_compute_time

                    start_eval_time = time.perf_counter()
                    labels = [x["sem_seg"][None].cuda() for x in inputs]

                    logits = [output["uni_logits"] for output in outputs]
                    image = inputs[0]["image"].permute(1, 2, 0)
                    visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)

                    for lb, lg in zip(labels, logits):
                        lg = lg[None]
                        lb = lb.long()
                        # lb = F.interpolate(lb.unsqueeze(1).float(), size=(lg.shape[2], lg.shape[3]),
                        #         mode='nearest').squeeze(1).long()

                        probs = torch.softmax(lg, dim=1)
                        preds = torch.argmax(probs, dim=1)
                        unique_pred = torch.unique(preds)
                        vis_output = visualizer.draw_sem_seg(
                            preds[0].cpu()
                        )
                        file_name = inputs[0]['file_name'].split('/')[-1]
                        if osp.exists(f"output/{dataset_name}/") == False:
                            os.makedirs(f"output/{dataset_name}/")
                        vis_output.save(f"output/{dataset_name}/{file_name}")
                        # has_saved = False
                        # src_file_name = None
                        # for pred_id in unique_pred:
                            
                        #     if len(image_list[int(pred_id)]) < 10:
                        #         file_name = inputs[0]['file_name'].split('/')[-1]
                        #         image_list[int(pred_id)].append(file_name)
                        #         if osp.exists(f"output/{dataset_name}/{int(pred_id)}") == False:
                        #             os.makedirs(f"output/{dataset_name}/{int(pred_id)}")
                        #         if has_saved == False:
                        #             vis_output.save(f"output/{dataset_name}/{int(pred_id)}/{file_name}")
                        #             src_file_name = osp.join(os.getcwd(), f"output/{dataset_name}/{int(pred_id)}/{file_name}")
                        #             has_saved = True
                        #         else:
                        #             os.symlink(src_file_name, f"output/{dataset_name}/{int(pred_id)}/{file_name}")
                                
                        keep = lb != ignore_index
                        

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
            
        datasets = ['city', 'mapi', 'sun', 'bdd', 'idd', 'ade', 'coco']
        city_lb = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
        mapi_lb = ["Bird", "Ground Animal", "Curb", "Fence", "Guard Rail", "Barrier", "Wall", "Bike Lane", "Crosswalk - Plain", "Curb Cut", "Parking", "Pedestrian Area", "Rail Track", "Road", "Service Lane", "Sidewalk", "Bridge", "Building", "Tunnel", "Person", "Bicyclist", "Motorcyclist", "Other Rider", "Lane Marking - Crosswalk", "Lane Marking - General", "Mountain", "Sand", "Sky", "Snow", "Terrain", "Vegetation", "Water", "Banner", "Bench", "Bike Rack", "Billboard", "Catch Basin", "CCTV Camera", "Fire Hydrant", "Junction Box", "Manhole", "Phone Booth", "Pothole", "Street Light", "Pole", "Traffic Sign Frame", "Utility Pole", "Traffic Light", "Traffic Sign (Back)", "Traffic Sign (Front)", "Trash Can", "Bicycle", "Boat", "Bus", "Car", "Caravan", "Motorcycle", "On Rails", "Other Vehicle", "Trailer", "Truck", "Wheeled Slow", "Car Mount", "Ego Vehicle"]
        sun_lb = [ "bag", "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter", "blinds", "desk", "shelves", "curtain", "dresser", "pillow", "mirror", "floor mat", "clothes", "ceiling", "books", "refridgerator", "television", "paper", "towel", "shower curtain", "box", "whiteboard", "person", "night stand", "toilet", "sink", "lamp", "bathtub"]
        bdd_lb = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
        idd_lb = ["road", "drivable fallback or parking", "sidewalk", "non-drivable fallback or rail track", "person or animal", "out of roi or rider", "motorcycle", "bicycle", "autorickshaw", "car", "truck", "bus", "trailer or caravan or vehicle fallback", "curb", "wall", "fence", "guard rail", "billboard", "traffic sign", "traffic light", "polegroup or pole", "obs-str-bar-fallback", "building", "tunnel or bridge", "vegetation", "sky or fallback background"]
        ade_lb = ["flag", "wall", "building, edifice", "sky", "floor, flooring", "tree", "ceiling", "road, route", "bed ", "windowpane, window ", "grass", "cabinet", "sidewalk, pavement", "person, individual, someone, somebody, mortal, soul", "earth, ground", "door, double door", "table", "mountain, mount", "plant, flora, plant life", "curtain, drape, drapery, mantle, pall", "chair", "car, auto, automobile, machine, motorcar", "water", "painting, picture", "sofa, couch, lounge", "shelf", "house", "sea", "mirror", "rug, carpet, carpeting", "field", "armchair", "seat", "fence, fencing", "desk", "rock, stone", "wardrobe, closet, press", "lamp", "bathtub, bathing tub, bath, tub", "railing, rail", "cushion", "base, pedestal, stand", "box", "column, pillar", "signboard, sign", "chest of drawers, chest, bureau, dresser", "counter", "sand", "sink", "skyscraper", "fireplace, hearth, open fireplace", "refrigerator, icebox", "grandstand, covered stand", "path", "stairs, steps", "runway", "case, display case, showcase, vitrine", "pool table, billiard table, snooker table", "pillow", "screen door, screen", "stairway, staircase", "river", "bridge, span", "bookcase", "blind, screen", "coffee table, cocktail table", "toilet, can, commode, crapper, pot, potty, stool, throne", "flower", "book", "hill", "bench", "countertop", "stove, kitchen stove, range, kitchen range, cooking stove", "palm, palm tree", "kitchen island", "computer, computing machine, computing device, data processor, electronic computer, information processing system", "swivel chair", "boat", "bar", "arcade machine", "hovel, hut, hutch, shack, shanty", "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle", "towel", "light, light source", "truck, motortruck", "tower", "chandelier, pendant, pendent", "awning, sunshade, sunblind", "streetlight, street lamp", "booth, cubicle, stall, kiosk", "television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box", "airplane, aeroplane, plane", "dirt track", "apparel, wearing apparel, dress, clothes", "pole", "land, ground, soil", "bannister, banister, balustrade, balusters, handrail", "escalator, moving staircase, moving stairway", "ottoman, pouf, pouffe, puff, hassock", "bottle", "buffet, counter, sideboard", "poster, posting, placard, notice, bill, card", "stage", "van", "ship", "fountain", "conveyer belt, conveyor belt, conveyer, conveyor, transporter", "canopy", "washer, automatic washer, washing machine", "plaything, toy", "swimming pool, swimming bath, natatorium", "stool", "barrel, cask", "basket, handbasket", "waterfall, falls", "tent, collapsible shelter", "bag", "minibike, motorbike", "cradle", "oven", "ball", "food, solid food", "step, stair", "tank, storage tank", "trade name, brand name, brand, marque", "microwave, microwave oven", "pot, flowerpot", "animal, animate being, beast, brute, creature, fauna", "bicycle, bike, wheel, cycle ", "lake", "dishwasher, dish washer, dishwashing machine", "screen, silver screen, projection screen", "blanket, cover", "sculpture", "hood, exhaust hood", "sconce", "vase", "traffic light, traffic signal, stoplight", "tray", "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin", "fan", "pier, wharf, wharfage, dock", "crt screen", "plate", "monitor, monitoring device", "bulletin board, notice board", "shower", "radiator", "glass, drinking glass", "clock"]
        coco_lb = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "banner", "blanket", "bridge", "cardboard", "counter", "curtain", "door-stuff", "floor-wood", "flower", "fruit", "gravel", "house", "light", "mirror-stuff", "net", "pillow", "platform", "playingfield", "railroad", "river", "road", "roof", "sand", "sea", "shelf", "snow", "stairs", "tent", "towel", "wall-brick", "wall-stone", "wall-tile", "wall-wood", "water-other", "window-blind", "window-other", "tree-merged", "fence-merged", "ceiling-merged", "sky-other-merged", "cabinet-merged", "table-merged", "floor-other-merged", "pavement-merged", "mountain-merged", "grass-merged", "dirt-merged", "paper-merged", "food-other-merged", "building-other-merged", "rock-merged", "wall-other-merged", "rug-merged"]


        cost_matrix = torch.zeros(num_unify_class, n_classes)
        for uni_class in range(num_unify_class):
            total_pres = torch.sum(hist[:, uni_class])
            for set_class in range(n_classes):
                total_target = torch.sum(hist[set_class])
                if total_target == 0:
                    cost_matrix[uni_class, set_class] = 1
                    continue
                iou = hist[set_class, uni_class] / (total_pres + total_target - hist[set_class, uni_class])
                cost_matrix[uni_class, set_class] = 1 - iou
        # logger.info(cost_matrix)
        mcmf = MinCostMaxFlow_Unseen()
        src, target = mcmf(cost_matrix) 

        bipart = datasets_cats[dataset_idx] * torch.ones(231)
        for s, t in zip(src, target):
            bipart[s] = t
            # buckets[index] = new_val
        logger.info(bipart)
        # target_bipart.append(bipart.cuda())

        for i in range(hist.shape[1]):
            
            for idx, bi in enumerate(bipart_graph):
                this_col = bi[:, i]
                if not (this_col == 1).any():
                    logger.info(f"{i}:{datasets[idx]}:None")
                    bipart[s] = 255
                else:
                    pos = (list(this_col.cpu().numpy())).index(1)
                    this_lb = eval(datasets[idx]+'_lb')
                    logger.info(f"{i}:{datasets[idx]}:{this_lb[pos]}")
                    
            logger.info(f"{i}:{hist[:,i]}")
            

