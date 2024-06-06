import sys

sys.path.insert(0, '.')
import os.path as osp
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from collections import defaultdict
from pycocotools import mask as maskutils
import numba
import time

import pickle
from contextlib import ExitStack, contextmanager


from detectron2.engine.hooks import HookBase
import datetime
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import MetadataCatalog
import logging
from ..modeling.GNN.gen_graph_node_feature import gen_graph_node_feature
from sklearn.cluster import DBSCAN  # DBSCAN API




def create_uni_label_space_by_text(cfg):
    

    datasets = cfg.DATASETS.EVAL # ['city', 'mapi', 'sun', 'bdd', 'idd', 'ade', 'coco']
    city_lb = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
    mapi_lb = ["Bird", "Ground Animal", "Curb", "Fence", "Guard Rail", "Barrier", "Wall", "Bike Lane", "Crosswalk - Plain", "Curb Cut", "Parking", "Pedestrian Area", "Rail Track", "Road", "Service Lane", "Sidewalk", "Bridge", "Building", "Tunnel", "Person", "Bicyclist", "Motorcyclist", "Other Rider", "Lane Marking - Crosswalk", "Lane Marking - General", "Mountain", "Sand", "Sky", "Snow", "Terrain", "Vegetation", "Water", "Banner", "Bench", "Bike Rack", "Billboard", "Catch Basin", "CCTV Camera", "Fire Hydrant", "Junction Box", "Manhole", "Phone Booth", "Pothole", "Street Light", "Pole", "Traffic Sign Frame", "Utility Pole", "Traffic Light", "Traffic Sign (Back)", "Traffic Sign (Front)", "Trash Can", "Bicycle", "Boat", "Bus", "Car", "Caravan", "Motorcycle", "On Rails", "Other Vehicle", "Trailer", "Truck", "Wheeled Slow", "Car Mount", "Ego Vehicle"]
    sun_lb = [ "bag", "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter", "blinds", "desk", "shelves", "curtain", "dresser", "pillow", "mirror", "floor mat", "clothes", "ceiling", "books", "refridgerator", "television", "paper", "towel", "shower curtain", "box", "whiteboard", "person", "night stand", "toilet", "sink", "lamp", "bathtub"]
    bdd_lb = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
    idd_lb = ["road", "drivable fallback or parking", "sidewalk", "non-drivable fallback or rail track", "person or animal", "out of roi or rider", "motorcycle", "bicycle", "autorickshaw", "car", "truck", "bus", "trailer or caravan or vehicle fallback", "curb", "wall", "fence", "guard rail", "billboard", "traffic sign", "traffic light", "polegroup or pole", "obs-str-bar-fallback", "building", "tunnel or bridge", "vegetation", "sky or fallback background"]
    ade_lb = ["flag", "wall", "building, edifice", "sky", "floor, flooring", "tree", "ceiling", "road, route", "bed ", "windowpane, window ", "grass", "cabinet", "sidewalk, pavement", "person, individual, someone, somebody, mortal, soul", "earth, ground", "door, double door", "table", "mountain, mount", "plant, flora, plant life", "curtain, drape, drapery, mantle, pall", "chair", "car, auto, automobile, machine, motorcar", "water", "painting, picture", "sofa, couch, lounge", "shelf", "house", "sea", "mirror", "rug, carpet, carpeting", "field", "armchair", "seat", "fence, fencing", "desk", "rock, stone", "wardrobe, closet, press", "lamp", "bathtub, bathing tub, bath, tub", "railing, rail", "cushion", "base, pedestal, stand", "box", "column, pillar", "signboard, sign", "chest of drawers, chest, bureau, dresser", "counter", "sand", "sink", "skyscraper", "fireplace, hearth, open fireplace", "refrigerator, icebox", "grandstand, covered stand", "path", "stairs, steps", "runway", "case, display case, showcase, vitrine", "pool table, billiard table, snooker table", "pillow", "screen door, screen", "stairway, staircase", "river", "bridge, span", "bookcase", "blind, screen", "coffee table, cocktail table", "toilet, can, commode, crapper, pot, potty, stool, throne", "flower", "book", "hill", "bench", "countertop", "stove, kitchen stove, range, kitchen range, cooking stove", "palm, palm tree", "kitchen island", "computer, computing machine, computing device, data processor, electronic computer, information processing system", "swivel chair", "boat", "bar", "arcade machine", "hovel, hut, hutch, shack, shanty", "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle", "towel", "light, light source", "truck, motortruck", "tower", "chandelier, pendant, pendent", "awning, sunshade, sunblind", "streetlight, street lamp", "booth, cubicle, stall, kiosk", "television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box", "airplane, aeroplane, plane", "dirt track", "apparel, wearing apparel, dress, clothes", "pole", "land, ground, soil", "bannister, banister, balustrade, balusters, handrail", "escalator, moving staircase, moving stairway", "ottoman, pouf, pouffe, puff, hassock", "bottle", "buffet, counter, sideboard", "poster, posting, placard, notice, bill, card", "stage", "van", "ship", "fountain", "conveyer belt, conveyor belt, conveyer, conveyor, transporter", "canopy", "washer, automatic washer, washing machine", "plaything, toy", "swimming pool, swimming bath, natatorium", "stool", "barrel, cask", "basket, handbasket", "waterfall, falls", "tent, collapsible shelter", "bag", "minibike, motorbike", "cradle", "oven", "ball", "food, solid food", "step, stair", "tank, storage tank", "trade name, brand name, brand, marque", "microwave, microwave oven", "pot, flowerpot", "animal, animate being, beast, brute, creature, fauna", "bicycle, bike, wheel, cycle ", "lake", "dishwasher, dish washer, dishwashing machine", "screen, silver screen, projection screen", "blanket, cover", "sculpture", "hood, exhaust hood", "sconce", "vase", "traffic light, traffic signal, stoplight", "tray", "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin", "fan", "pier, wharf, wharfage, dock", "crt screen", "plate", "monitor, monitoring device", "bulletin board, notice board", "shower", "radiator", "glass, drinking glass", "clock", "rug-merged"]
    coco_lb = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "banner", "blanket", "bridge", "cardboard", "counter", "curtain", "door-stuff", "floor-wood", "flower", "fruit", "gravel", "house", "light", "mirror-stuff", "net", "pillow", "platform", "playingfield", "railroad", "river", "road", "roof", "sand", "sea", "shelf", "snow", "stairs", "tent", "towel", "wall-brick", "wall-stone", "wall-tile", "wall-wood", "water-other", "window-blind", "window-other", "tree-merged", "fence-merged", "ceiling-merged", "sky-other-merged", "cabinet-merged", "table-merged", "floor-other-merged", "pavement-merged", "mountain-merged", "grass-merged", "dirt-merged", "paper-merged", "food-other-merged", "building-other-merged", "rock-merged", "wall-other-merged"]
    wilddash_lb = ['ego vehicle', 'road', 'sidewalk', 'building', 'wall', 'fence', 'guard rail', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pickup', 'van', 'billboard', 'street-light', 'road-marking', 'void']
    n_datasets = len(datasets)

    num_cats = cfg.DATASETS.DATASETS_CATS
    num_cats_by_name = {}
    for d, n_cat in zip(datasets, num_cats):
        num_cats_by_name[d] = n_cat
    total_cats = sum(num_cats)
    cnt = 0
    dataset_range = {}
    for d, c in zip(datasets, num_cats):
        dataset_range[d] = range(cnt, cnt + c)
        cnt = cnt + c
    print('dataset_range', dataset_range)
    id2source = np.concatenate(
    [np.ones(len(dataset_range[d]), dtype=np.int32) * i \
        for i, d in enumerate(datasets)]
    ).tolist()
    predid2name, id2sourceid, id2sourceindex, id2sourcename = [], [], [], []
    names = []
    for d in datasets:
        meta = MetadataCatalog.get(d)
        stuff_class = meta.stuff_classes
        predid2name.extend([d + '_' + lb_name for lb_name in stuff_class])
        id2sourceid.extend([i for i in range(len(stuff_class))])
        id2sourceindex.extend([i for i in range(len(stuff_class))])
        id2sourcename.extend([d for _ in range(len(stuff_class))])
        names.extend([d + '_' + lb_name for lb_name in stuff_class])

    def Get_Predhist_by_llm():
        graph_node_features = gen_graph_node_feature(cfg).float()
        def compute_cosine(a_vec, b_vec):
            norms1 = torch.norm(a_vec, dim=1, keepdim=True)
            norms2 = torch.norm(b_vec, dim=1, keepdim=True)
            
            # norm_a = torch.norm(a, dim=1, keepdim=True)

            normalized_a = a_vec / norms1


            normalized_b = b_vec / norms2

            cos_sim = torch.mm(normalized_a, normalized_b.t())
            
            return cos_sim
        
        predHist = compute_cosine(graph_node_features, graph_node_features)
        # for idx, d in enumerate(datasets):
        #     this_hist = {}
        #     this_emb = graph_node_features[dataset_range[d]]
        #     for idx2, d2 in enumerate(datasets):
        #         other_emb = graph_node_features[dataset_range[d2]]
        #         this_hist[d2] = compute_cosine(graph_node_features, graph_node_features) * 100
        #     predHist[d] = this_hist
        return predHist, graph_node_features

    predHist, feats = Get_Predhist_by_llm()
    predHist = 1. - predHist
    predHist[predHist<0] = 0
    print(torch.min(predHist))
    # print(predHist)
    # data = np.arange(predHist.shape[0])

    dbscan = DBSCAN(eps=0.105,
                    min_samples=1,
                    metric='precomputed')
    result = dbscan.fit_predict(predHist)
    print(result)

    names = []
    for d in datasets:
        meta = MetadataCatalog.get(d)
        stuff_class = meta.stuff_classes
        names.extend(stuff_class)
    merged = [False for _ in range(len(names))]
    print_order = datasets
    heads = datasets
    head_str = 'key'
    for head in heads:
        head_str = head_str + ', {}'.format(head)
    print(head_str)
    cnt = 0
    for i in range(max(result)):
        inds = np.where(result == i)[0]
        dataset_name = {d: '' for d in datasets}
        for ind in inds:
            
            d = datasets[id2source[ind]]
            if len(dataset_name[d]) != 0:
                continue
                # raise Exception("Categories from the same data set cannot be grouped into one class")
            name = names[ind]
            dataset_name[d] = name
            merged[ind] = True
        # if name == 'background':
        #   continue
        unified_name = dataset_name[print_order[0]].replace(',', '_')
        for d in print_order[1:]:
            unified_name = unified_name + '_{}'.format(dataset_name[d].replace(',', '_'))
        print(unified_name, end='')
        cnt = cnt + 1
        for d in print_order:
            # if d == 'oid':
            #     print(', {}, {}'.format(oidname2freebase[dataset_name[d]], dataset_name[d]), end='')
            # else:
            # print(', {}'.format(dataset_name[d]), end='')
            # print("!:", dataset_name[d])
            if dataset_name[d] != '':
                meta = MetadataCatalog.get(d)
                stuff_class = meta.stuff_classes
                print(', {}'.format(stuff_class.index(dataset_name[d])), end='')
            else:
                print(', {}'.format(dataset_name[d]), end='')
        print()
    for ind in range(len(names)):
        if not merged[ind]:
            dataset_name = {d: '' for d in datasets}
            d = datasets[id2source[ind]]
            name = names[ind]
            # if name == 'background':
            #   continue
            dataset_name[d] = name
            unified_name = dataset_name[print_order[0]].replace(',', '_')
            for d in print_order[1:]:
                unified_name = unified_name + '_{}'.format(dataset_name[d].replace(',', '_'))
            print(unified_name, end='')
            cnt = cnt + 1
            for d in print_order:
                # if d == 'oid':
                #     print(', {}, {}'.format(oidname2freebase[dataset_name[d]], dataset_name[d]), end='')
                # else:
                # print(', {}'.format(dataset_name[d]), end='')
                if dataset_name[d] != '':
                    meta = MetadataCatalog.get(d)
                    stuff_class = meta.stuff_classes
                    print(', {}'.format(stuff_class.index(dataset_name[d])), end='')
                else:
                    print(', {}'.format(dataset_name[d]), end='')
            print()
    print(f'cats: {cnt}')