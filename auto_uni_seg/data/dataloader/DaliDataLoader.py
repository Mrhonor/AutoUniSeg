#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import cv2
import numpy as np

import types
from random import shuffle
from ...utils.configer import Configer

import threading
from detectron2.structures import BitMasks, Instances
import logging
from ..dataset_mappers.semantic_dataset_mapper import SemanticDatasetMapper
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader

Log = logging.getLogger(__name__)
cityscapes_cv2_labels_info = [
    {"hasInstances": False, "category": "void", "catid": 0, "name": "unlabeled", "ignoreInEval": True, "id": 0, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "ego vehicle", "ignoreInEval": True, "id": 1, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "rectification border", "ignoreInEval": True, "id": 2, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "out of roi", "ignoreInEval": True, "id": 3, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "static", "ignoreInEval": True, "id": 4, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "dynamic", "ignoreInEval": True, "id": 5, "color": [111, 74, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "ground", "ignoreInEval": True, "id": 6, "color": [81, 0, 81], "trainId": 255},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "road", "ignoreInEval": False, "id": 7, "color": [128, 64, 128], "trainId": 0},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "sidewalk", "ignoreInEval": False, "id": 8, "color": [244, 35, 232], "trainId": 1},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "parking", "ignoreInEval": True, "id": 9, "color": [250, 170, 160], "trainId": 255},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "rail track", "ignoreInEval": True, "id": 10, "color": [230, 150, 140], "trainId": 255},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "building", "ignoreInEval": False, "id": 11, "color": [70, 70, 70], "trainId": 2},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "wall", "ignoreInEval": False, "id": 12, "color": [102, 102, 156], "trainId": 3},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "fence", "ignoreInEval": False, "id": 13, "color": [190, 153, 153], "trainId": 4},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "guard rail", "ignoreInEval": True, "id": 14, "color": [180, 165, 180], "trainId": 255},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "bridge", "ignoreInEval": True, "id": 15, "color": [150, 100, 100], "trainId": 255},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "tunnel", "ignoreInEval": True, "id": 16, "color": [150, 120, 90], "trainId": 255},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "pole", "ignoreInEval": False, "id": 17, "color": [153, 153, 153], "trainId": 5},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "polegroup", "ignoreInEval": True, "id": 18, "color": [153, 153, 153], "trainId": 255},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "traffic light", "ignoreInEval": False, "id": 19, "color": [250, 170, 30], "trainId": 6},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "traffic sign", "ignoreInEval": False, "id": 20, "color": [220, 220, 0], "trainId": 7},
    {"hasInstances": False, "category": "nature", "catid": 4, "name": "vegetation", "ignoreInEval": False, "id": 21, "color": [107, 142, 35], "trainId": 8},
    {"hasInstances": False, "category": "nature", "catid": 4, "name": "terrain", "ignoreInEval": False, "id": 22, "color": [152, 251, 152], "trainId": 9},
    {"hasInstances": False, "category": "sky", "catid": 5, "name": "sky", "ignoreInEval": False, "id": 23, "color": [70, 130, 180], "trainId": 10},
    {"hasInstances": True, "category": "human", "catid": 6, "name": "person", "ignoreInEval": False, "id": 24, "color": [220, 20, 60], "trainId": 11},
    {"hasInstances": True, "category": "human", "catid": 6, "name": "rider", "ignoreInEval": False, "id": 25, "color": [255, 0, 0], "trainId": 12},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "car", "ignoreInEval": False, "id": 26, "color": [0, 0, 142], "trainId": 13},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "truck", "ignoreInEval": False, "id": 27, "color": [0, 0, 70], "trainId": 14},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "bus", "ignoreInEval": False, "id": 28, "color": [0, 60, 100], "trainId": 15},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "caravan", "ignoreInEval": True, "id": 29, "color": [0, 0, 90], "trainId": 255},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "trailer", "ignoreInEval": True, "id": 30, "color": [0, 0, 110], "trainId": 255},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "train", "ignoreInEval": False, "id": 31, "color": [0, 80, 100], "trainId": 16},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "motorcycle", "ignoreInEval": False, "id": 32, "color": [0, 0, 230], "trainId": 17},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "bicycle", "ignoreInEval": False, "id": 33, "color": [119, 11, 32], "trainId": 18},
    {"hasInstances": False, "category": "vehicle", "catid": 7, "name": "license plate", "ignoreInEval": True, "id": -1, "color": [0, 0, 142], "trainId": 255}
]

sunrgbd_labels_info = [
{"name": "unlabeled", "id": 0, "trainId": 255},
{"name": "wall", "id": 1, "trainId": 1},
{"name": "floor", "id": 2, "trainId": 2},
{"name": "cabinet", "id": 3, "trainId": 3},
{"name": "bed", "id": 4, "trainId": 4},
{"name": "chair", "id": 5, "trainId": 5},
{"name": "sofa", "id": 6, "trainId": 6},
{"name": "table", "id": 7, "trainId": 7},
{"name": "door", "id": 8, "trainId": 8},
{"name": "window", "id": 9, "trainId": 9},
{"name": "bookshelf", "id": 10, "trainId": 10},
{"name": "picture", "id": 11, "trainId": 11},
{"name": "counter", "id": 12, "trainId": 12},
{"name": "blinds", "id": 13, "trainId": 13},
{"name": "desk", "id": 14, "trainId": 14},
{"name": "shelves", "id": 15, "trainId": 15},
{"name": "curtain", "id": 16, "trainId": 16},
{"name": "dresser", "id": 17, "trainId": 17},
{"name": "pillow", "id": 18, "trainId": 18},
{"name": "mirror", "id": 19, "trainId": 19},
{"name": "floor mat", "id": 20, "trainId": 20},
{"name": "clothes", "id": 21, "trainId": 21},
{"name": "ceiling", "id": 22, "trainId": 22},
{"name": "books", "id": 23, "trainId": 23},
{"name": "refridgerator", "id": 24, "trainId": 24},
{"name": "television", "id": 25, "trainId": 25},
{"name": "paper", "id": 26, "trainId": 26},
{"name": "towel", "id": 27, "trainId": 27},
{"name": "shower curtain", "id": 28, "trainId": 28},
{"name": "box", "id": 29, "trainId": 29},
{"name": "whiteboard", "id": 30, "trainId": 30},
{"name": "person", "id": 31, "trainId": 31},
{"name": "night stand", "id": 32, "trainId": 32},
{"name": "toilet", "id": 33, "trainId": 33},
{"name": "sink", "id": 34, "trainId": 34},
{"name": "lamp", "id": 35, "trainId": 35},
{"name": "bathtub", "id": 36, "trainId": 36},
{"name": "bag", "id": 37, "trainId": 0},
]

idd_cv2_labels_info = [
    {"name": "person", "id": 0, "color": [0, 0, 0], "trainId": 4},
    {"name": "truck", "id": 1, "color": [0, 0, 0], "trainId": 10},
    {"name": "fence", "id": 2, "color": [0, 0, 0], "trainId": 15},
    {"name": "billboard", "id": 3, "color": [0, 0, 0], "trainId": 17},
    {"name": "bus", "id": 4, "color": [0, 0, 0], "trainId": 11},
    {"name": "out of roi", "id": 5, "color": [0, 0, 0], "trainId": 255},
    {"name": "curb", "id": 6, "color": [0, 0, 0], "trainId": 13},
    {"name": "obs-str-bar-fallback", "id": 7, "color": [0, 0, 0], "trainId": 21},
    {"name": "tunnel", "id": 8, "color": [0, 0, 0], "trainId": 23},
    {"name": "non-drivable fallback", "id": 9, "color": [0, 0, 0], "trainId": 3},
    {"name": "bridge", "id": 10, "color": [0, 0, 0], "trainId": 23},
    {"name": "road", "id": 11, "color": [0, 0, 0], "trainId": 0},
    {"name": "wall", "id": 12, "color": [0, 0, 0], "trainId": 14},
    {"name": "traffic sign", "id": 13, "color": [0, 0, 0], "trainId": 18},
    {"name": "trailer", "id": 14, "color": [0, 0, 0], "trainId": 12},
    {"name": "animal", "id": 15, "color": [0, 0, 0], "trainId": 4},
    {"name": "building", "id": 16, "color": [0, 0, 0], "trainId": 22},
    {"name": "sky", "id": 17, "color": [0, 0, 0], "trainId": 25},
    {"name": "drivable fallback", "id": 18, "color": [0, 0, 0], "trainId": 1},
    {"name": "guard rail", "id": 19, "color": [0, 0, 0], "trainId": 16},
    {"name": "bicycle", "id": 20, "color": [0, 0, 0], "trainId": 7},
    {"name": "traffic light", "id": 21, "color": [0, 0, 0], "trainId": 19},
    {"name": "polegroup", "id": 22, "color": [0, 0, 0], "trainId": 20},
    {"name": "motorcycle", "id": 23, "color": [0, 0, 0], "trainId": 6},
    {"name": "car", "id": 24, "color": [0, 0, 0], "trainId": 9},
    {"name": "parking", "id": 25, "color": [0, 0, 0], "trainId": 1},
    {"name": "fallback background", "id": 26, "color": [0, 0, 0], "trainId": 25},
    {"name": "license plate", "id": 27, "color": [0, 0, 0], "trainId": 255},
    {"name": "rectification border", "id": 28, "color": [0, 0, 0], "trainId": 255},
    {"name": "train", "id": 29, "color": [0, 0, 0], "trainId": 255},
    {"name": "rider", "id": 30, "color": [0, 0, 0], "trainId": 5},
    {"name": "rail track", "id": 31, "color": [0, 0, 0], "trainId": 3},
    {"name": "sidewalk", "id": 32, "color": [0, 0, 0], "trainId": 2},
    {"name": "caravan", "id": 33, "color": [0, 0, 0], "trainId": 12},
    {"name": "pole", "id": 34, "color": [0, 0, 0], "trainId": 20},
    {"name": "vegetation", "id": 35, "color": [0, 0, 0], "trainId": 24},
    {"name": "autorickshaw", "id": 36, "color": [0, 0, 0], "trainId": 8},
    {"name": "vehicle fallback", "id": 37, "color": [0, 0, 0], "trainId": 12},
    {"name": "unlabel", "id":255, "color":[0,0,0], "trainId": 255},
]

bdd100k_data_labels_info = [
    {"name": "unlabel", "id":255, "color":[0,0,0], "trainId": 255},
    {"name": "road", "id": 0, "color": [0, 0, 0], "trainId": 0},
    {"name": "sidewalk", "id": 1, "color": [0, 0, 0], "trainId": 1},
    {"name": "building", "id": 2, "color": [0, 0, 0], "trainId": 2},
    {"name": "wall", "id": 3, "color": [0, 0, 0], "trainId": 3},
    {"name": "fence", "id": 4, "color": [0, 0, 0], "trainId": 4},
    {"name": "pole", "id": 5, "color": [0, 0, 0], "trainId": 5},
    {"name": "traffic light", "id": 6, "color": [0, 0, 0], "trainId": 6},
    {"name": "traffic sign", "id": 7, "color": [0, 0, 0], "trainId": 7},
    {"name": "vegetation", "id": 8, "color": [0, 0, 0], "trainId": 8},
    {"name": "terrain", "id": 9, "color": [0, 0, 0], "trainId": 9},
    {"name": "sky", "id": 10, "color": [0, 0, 0], "trainId": 10},
    {"name": "person", "id": 11, "color": [0, 0, 0], "trainId": 11},
    {"name": "rider", "id": 12, "color": [0, 0, 0], "trainId": 12},
    {"name": "car", "id": 13, "color": [0, 0, 0], "trainId": 13},
    {"name": "truck", "id": 14, "color": [0, 0, 0], "trainId": 14},
    {"name": "bus", "id": 15, "color": [0, 0, 0], "trainId": 15},
    {"name": "train", "id": 16, "color": [0, 0, 0], "trainId": 16},
    {"name": "motorcycle", "id": 17, "color": [0, 0, 0], "trainId": 17},
    {"name": "bicycle", "id": 18, "color": [0, 0, 0], "trainId": 18},
]

coco_data_labels_info = [
{"name": "person", "id": 1, "trainId": 0},
{"name": "bicycle", "id": 2, "trainId": 1},
{"name": "car", "id": 3, "trainId": 2},
{"name": "motorcycle", "id": 4, "trainId": 3},
{"name": "airplane", "id": 5, "trainId": 4},
{"name": "bus", "id": 6, "trainId": 5},
{"name": "train", "id": 7, "trainId": 6},
{"name": "truck", "id": 8, "trainId": 7},
{"name": "boat", "id": 9, "trainId": 8},
{"name": "traffic light", "id": 10, "trainId": 9},
{"name": "fire hydrant", "id": 11, "trainId": 10},
{"name": "stop sign", "id": 13, "trainId": 11},
{"name": "parking meter", "id": 14, "trainId": 12},
{"name": "bench", "id": 15, "trainId": 13},
{"name": "bird", "id": 16, "trainId": 14},
{"name": "cat", "id": 17, "trainId": 15},
{"name": "dog", "id": 18, "trainId": 16},
{"name": "horse", "id": 19, "trainId": 17},
{"name": "sheep", "id": 20, "trainId": 18},
{"name": "cow", "id": 21, "trainId": 19},
{"name": "elephant", "id": 22, "trainId": 20},
{"name": "bear", "id": 23, "trainId": 21},
{"name": "zebra", "id": 24, "trainId": 22},
{"name": "giraffe", "id": 25, "trainId": 23},
{"name": "backpack", "id": 27, "trainId": 24},
{"name": "umbrella", "id": 28, "trainId": 25},
{"name": "handbag", "id": 31, "trainId": 26},
{"name": "tie", "id": 32, "trainId": 27},
{"name": "suitcase", "id": 33, "trainId": 28},
{"name": "frisbee", "id": 34, "trainId": 29},
{"name": "skis", "id": 35, "trainId": 30},
{"name": "snowboard", "id": 36, "trainId": 31},
{"name": "sports ball", "id": 37, "trainId": 32},
{"name": "kite", "id": 38, "trainId": 33},
{"name": "baseball bat", "id": 39, "trainId": 34},
{"name": "baseball glove", "id": 40, "trainId": 35},
{"name": "skateboard", "id": 41, "trainId": 36},
{"name": "surfboard", "id": 42, "trainId": 37},
{"name": "tennis racket", "id": 43, "trainId": 38},
{"name": "bottle", "id": 44, "trainId": 39},
{"name": "wine glass", "id": 46, "trainId": 40},
{"name": "cup", "id": 47, "trainId": 41},
{"name": "fork", "id": 48, "trainId": 42},
{"name": "knife", "id": 49, "trainId": 43},
{"name": "spoon", "id": 50, "trainId": 44},
{"name": "bowl", "id": 51, "trainId": 45},
{"name": "banana", "id": 52, "trainId": 46},
{"name": "apple", "id": 53, "trainId": 47},
{"name": "sandwich", "id": 54, "trainId": 48},
{"name": "orange", "id": 55, "trainId": 49},
{"name": "broccoli", "id": 56, "trainId": 50},
{"name": "carrot", "id": 57, "trainId": 51},
{"name": "hot dog", "id": 58, "trainId": 52},
{"name": "pizza", "id": 59, "trainId": 53},
{"name": "donut", "id": 60, "trainId": 54},
{"name": "cake", "id": 61, "trainId": 55},
{"name": "chair", "id": 62, "trainId": 56},
{"name": "couch", "id": 63, "trainId": 57},
{"name": "potted plant", "id": 64, "trainId": 58},
{"name": "bed", "id": 65, "trainId": 59},
{"name": "dining table", "id": 67, "trainId": 60},
{"name": "toilet", "id": 70, "trainId": 61},
{"name": "tv", "id": 72, "trainId": 62},
{"name": "laptop", "id": 73, "trainId": 63},
{"name": "mouse", "id": 74, "trainId": 64},
{"name": "remote", "id": 75, "trainId": 65},
{"name": "keyboard", "id": 76, "trainId": 66},
{"name": "cell phone", "id": 77, "trainId": 67},
{"name": "microwave", "id": 78, "trainId": 68},
{"name": "oven", "id": 79, "trainId": 69},
{"name": "toaster", "id": 80, "trainId": 70},
{"name": "sink", "id": 81, "trainId": 71},
{"name": "refrigerator", "id": 82, "trainId": 72},
{"name": "book", "id": 84, "trainId": 73},
{"name": "clock", "id": 85, "trainId": 74},
{"name": "vase", "id": 86, "trainId": 75},
{"name": "scissors", "id": 87, "trainId": 76},
{"name": "teddy bear", "id": 88, "trainId": 77},
{"name": "hair drier", "id": 89, "trainId": 78},
{"name": "toothbrush", "id": 90, "trainId": 79},
{"name": "banner", "id": 92, "trainId": 80},
{"name": "blanket", "id": 93, "trainId": 81},
{"name": "bridge", "id": 95, "trainId": 82},
{"name": "cardboard", "id": 100, "trainId": 83},
{"name": "counter", "id": 107, "trainId": 84},
{"name": "curtain", "id": 109, "trainId": 85},
{"name": "door-stuff", "id": 112, "trainId": 86},
{"name": "floor-wood", "id": 118, "trainId": 87},
{"name": "flower", "id": 119, "trainId": 88},
{"name": "fruit", "id": 122, "trainId": 89},
{"name": "gravel", "id": 125, "trainId": 90},
{"name": "house", "id": 128, "trainId": 91},
{"name": "light", "id": 130, "trainId": 92},
{"name": "mirror-stuff", "id": 133, "trainId": 93},
{"name": "net", "id": 138, "trainId": 94},
{"name": "pillow", "id": 141, "trainId": 95},
{"name": "platform", "id": 144, "trainId": 96},
{"name": "playingfield", "id": 145, "trainId": 97},
{"name": "railroad", "id": 147, "trainId": 98},
{"name": "river", "id": 148, "trainId": 99},
{"name": "road", "id": 149, "trainId": 100},
{"name": "roof", "id": 151, "trainId": 101},
{"name": "sand", "id": 154, "trainId": 102},
{"name": "sea", "id": 155, "trainId": 103},
{"name": "shelf", "id": 156, "trainId": 104},
{"name": "snow", "id": 159, "trainId": 105},
{"name": "stairs", "id": 161, "trainId": 106},
{"name": "tent", "id": 166, "trainId": 107},
{"name": "towel", "id": 168, "trainId": 108},
{"name": "wall-brick", "id": 171, "trainId": 109},
{"name": "wall-stone", "id": 175, "trainId": 110},
{"name": "wall-tile", "id": 176, "trainId": 111},
{"name": "wall-wood", "id": 177, "trainId": 112},
{"name": "water-other", "id": 178, "trainId": 113},
{"name": "window-blind", "id": 180, "trainId": 114},
{"name": "window-other", "id": 181, "trainId": 115},
{"name": "tree-merged", "id": 184, "trainId": 116},
{"name": "fence-merged", "id": 185, "trainId": 117},
{"name": "ceiling-merged", "id": 186, "trainId": 118},
{"name": "sky-other-merged", "id": 187, "trainId": 119},
{"name": "cabinet-merged", "id": 188, "trainId": 120},
{"name": "table-merged", "id": 189, "trainId": 121},
{"name": "floor-other-merged", "id": 190, "trainId": 122},
{"name": "pavement-merged", "id": 191, "trainId": 123},
{"name": "mountain-merged", "id": 192, "trainId": 124},
{"name": "grass-merged", "id": 193, "trainId": 125},
{"name": "dirt-merged", "id": 194, "trainId": 126},
{"name": "paper-merged", "id": 195, "trainId": 127},
{"name": "food-other-merged", "id": 196, "trainId": 128},
{"name": "building-other-merged", "id": 197, "trainId": 129},
{"name": "rock-merged", "id": 198, "trainId": 130},
{"name": "wall-other-merged", "id": 199, "trainId": 131},
{"name": "rug-merged", "id": 200, "trainId": 132},
{"name": "unlabeled", "id":0, "trainId": 255},
]

# class IMLbReaderThread(threading.Thread):
#     def __init__(self, im_lb_path):
#         super(IMLbReaderThread, self).__init__()
#         self.im_lb_path = im_lb_path
#         self.im_lb = []
#         # self.lb_map=lb_map
#         # self.trans_func = trans_func
        

#     def run(self):
#         for im_lb_p in self.im_lb_path:
#             imp, lbp = im_lb_p
#             im = np.fromfile(imp, dtype=np.uint8)
#             lb = np.fromfile(lbp, dtype=np.uint8)

#             self.im_lb.append([im, lb])


# class ExternalInputIterator(object):
#     def __init__(self, batch_size, dataroot, annpath, mode='train'):
#         self.batch_size = batch_size
#         # self.num_instances = num_instances
#         self.shuffled = False
#         if mode == 'train':
#             self.shuffled = True

#         # self.img_seq_length = num_instances

#         self.lb_map = None

#         with open(annpath, 'r') as fr:
#             pairs = fr.read().splitlines()
#         self.img_paths, self.lb_paths = [], []
#         for pair in pairs:
#             imgpth, lbpth = pair.split(',')
#             self.img_paths.append(osp.join(dataroot, imgpth))
#             self.lb_paths.append(osp.join(dataroot, lbpth))

#         assert len(self.img_paths) == len(self.lb_paths)
#         self.len = len(self.img_paths)

#         # self.list_of_pids = list(images_dict.keys())
#         self._num_classes = len(self.img_paths) #len(self.list_of_pids)
#         self.all_indexs = list(range(self._num_classes))
#         self.n = self.__len__()


#     def __iter__(self):
#         self.i = 0
#         if self.shuffled:
#             shuffle(self.all_indexs)
#         return self

#     def __len__(self):
#         return len(self.all_indexs)

#     @staticmethod
#     def image_open(path):
#         return np.fromfile(path, dtype=np.uint8)

#     def __next__(self):
#         if self.i >= self.n:
#             self.__iter__()
#             raise StopIteration

#         # batch_images = []
#         # batch_labels = []

#         leave_num = self.n - self.i
#         current_batch_size = min(self.batch_size, leave_num) 
#         imp, lbp = [], []
#         for _ in range(current_batch_size):
#             tmp_index = self.all_indexs[self.i]
#             # p_id = self.list_of_pids[tmp_index]
#             imp.append(self.img_paths[tmp_index])
#             lbp.append(self.lb_paths[tmp_index])

#             self.i += 1

#         batch_images, batch_labels = self.get_image_label(imp, lbp)
#         # batch_labels.append(np.fromfile(lbp, dtype=np.uint8))

#         # batch_data = []
#         # for ins_i in range(self.num_instances):
#         #     elem = []
#         #     for batch_idx in range(current_batch_size):
#         #         elem.append(batch_images[batch_idx][ins_i])
#         #     batch_data.append(elem)

#         return batch_images, batch_labels
    
#     def get_image_label(self, impth, lbpth):
#         threads = []
#         for i in range(0, len(impth), 2):
#             if i+1 < len(impth):
#                 im_lb_path = [[impth[i], lbpth[i]], [impth[i+1], lbpth[i+1]]]
#             else:
#                 im_lb_path = [[impth[i], lbpth[i]]]
#             threads.append(IMLbReaderThread(im_lb_path))

#         for thread in threads:
#             thread.start()

#         for thread in threads:
#             thread.join()
        
#         ims = []
#         lbs = []
#         for thread in threads:
#             im_lbs = thread.im_lb
#             for im_lb in im_lbs:
#                 im, lb = im_lb
#                 ims.append(im)
#                 lbs.append(lb)

#         return ims, lbs

#     # next = __next__
#     # len = __len__
# class ExternalInputIteratorMul(object):
#     def __init__(self, batch_size, dataroot, annpath, mode='train'):
#         self.n_datasets = len(batch_size)
#         self.batch_size = batch_size
#         # self.num_instances = num_instances
#         self.shuffled = False
#         if mode == 'train':
#             self.shuffled = True

#         # self.img_seq_length = num_instances

#         self.lb_map = None
#         self.img_paths, self.lb_paths = [], []
#         self.len = 0
#         self.n = []
#         for root, anp in zip(dataroot, annpath):
#             with open(anp, 'r') as fr:
#                 pairs = fr.read().splitlines()
#             self.img_path, self.lb_path = [], []
#             for pair in pairs:
#                 imgpth, lbpth = pair.split(',')
#                 self.img_path.append(osp.join(root, imgpth))
#                 self.lb_path.append(osp.join(root, lbpth))

#             assert len(self.img_path) == len(self.lb_path)
#             self.img_paths.append(self.img_path)
#             self.lb_paths.append(self.lb_path)
#             self.len += len(self.img_path)
#             self.n.append(len(self.img_path))
    
#         # self.len = len(self.img_paths)

#         # self.list_of_pids = list(images_dict.keys())
#         # self._num_classes = len(self.img_paths) #len(self.list_of_pids)
#         self.all_indexs = [list(range(len(imp))) for imp in self.img_paths]
        
#         self.i = [0 for _ in self.img_paths]


#     def __iter__(self):
#         self.i = [0 for _ in self.img_paths]
#         if self.shuffled:
#             for i in range(len(self.all_indexs)):
#                 shuffle(self.all_indexs[i]) 
#         return self

#     def __len__(self):
#         return self.len #len(self.all_indexs)

#     @staticmethod
#     def image_open(path):
#         return np.fromfile(path, dtype=np.uint8)

#     def __next__(self):

#         batch_images = []
#         batch_labels = []
#         for idx in range(self.n_datasets):
            
#                 # raise StopIteration
#             # leave_num = self.n[idx] - self.i[idx]
#             # current_batch_size = min(self.batch_size, leave_num) 
#             for _ in range(self.batch_size[idx]):
#                 if self.i[idx] >= self.n[idx]:
#                     # self.__iter__()
#                     self.i[idx] = 0
#                     if self.shuffled:
#                         shuffle(self.all_indexs[idx])
#                 tmp_index = self.all_indexs[idx][self.i[idx]]
#                 # p_id = self.list_of_pids[tmp_index]
#                 imp = self.img_paths[idx][tmp_index]
#                 lbp = self.lb_paths[idx][tmp_index]
#                 batch_images.append(imp)
#                 batch_labels.append(lbp)

#                 self.i[idx] += 1

#             # batch_data = []
#             # for ins_i in range(self.num_instances):
#             #     elem = []
#             #     for batch_idx in range(current_batch_size):
#             #         elem.append(batch_images[batch_idx][ins_i])
#             #     batch_data.append(elem)

#         batch_images, batch_labels = self.get_image_label(batch_images, batch_labels)
#         return batch_images, batch_labels

#     def get_image_label(self, impth, lbpth):
#         threads = []
#         # for i in range(0, len(impth), 2):
#         #     if i+1 < len(impth):
#         #         im_lb_path = [[impth[i], lbpth[i]], [impth[i+1], lbpth[i+1]]]
#         #     else:
#         #         im_lb_path = [[impth[i], lbpth[i]]]
#         #     threads.append(IMLbReaderThread(im_lb_path))
#         for i in range(0, len(impth)):
#             # if i+1 < len(impth):
#             #     im_lb_path = [[impth[i], lbpth[i]], [impth[i+1], lbpth[i+1]]]
#             # else:
#             im_lb_path = [[impth[i], lbpth[i]]]
#             threads.append(IMLbReaderThread(im_lb_path))

#         for thread in threads:
#             thread.start()


#         for thread in threads:
#             thread.join()
        
#         ims = []
#         lbs = []
#         for thread in threads:
#             im_lbs = thread.im_lb
#             for im_lb in im_lbs:
#                 im, lb = im_lb
#                 ims.append(im)
#                 lbs.append(lb)

#         return ims, lbs

# def ExternalSourcePipelineMul(batch_size, num_threads, device_id, external_data, mode='train', scales=(0.5, 1.), size=(768, 768), p=0.5, brightness=0.4, contrast=0.4, saturation=0.4):
    
#     pipe = Pipeline(batch_size, num_threads, device_id, prefetch_queue_depth=4)
#     crop_h, crop_w = size
#     if not brightness is None and brightness >= 0:
#         brightness = [max(1-brightness, 0), 1+brightness]
#     if not contrast is None and contrast >= 0:
#         contrast = [max(1-contrast, 0), 1+contrast]
#     if not saturation is None and saturation >= 0:
#         saturation = [max(1-saturation, 0), 1+saturation]

#     # mean=[0.3257, 0.3690, 0.3223] # city, rgb
#     # std=[0.2112, 0.2148, 0.2115]
#     MEAN = np.asarray([0.3257, 0.3690, 0.3223])[None, None, :]
#     STD = np.asarray([0.2112, 0.2148, 0.2115])[None, None, :]
#     SCALE = 1 / 255.
#     with pipe:
#         jpegs, labels = fn.external_source(source=external_data, num_outputs=2, dtype=types.UINT8)
#         images = fn.decoders.image(jpegs, device="mixed")
#         labels = fn.decoders.image(labels, device="mixed", output_type=types.GRAY)
#         # images = fn.random_resized_crop()
#         # for i in range(len(images)):
#         # print(fn.peek_image_shape(labels))
#         # shape = fn.peek_image_shape(labels)
#         # images = images.gpu()
#         # labels = labels.gpu()
#         if mode == 'train':
#             images = fn.random_resized_crop(images, interp_type=types.INTERP_LINEAR, size=size, seed=1234)
#             labels = fn.random_resized_crop(labels, antialias=False, interp_type=types.INTERP_NN, size=size, seed=1234)

#             brightness_rate = fn.random.uniform(range=(min(brightness), max(brightness)))
#             contrast_rate = fn.random.uniform(range=(min(contrast), max(contrast)))
#             saturation_rate = fn.random.uniform(range=(min(saturation), max(saturation)))
#             images = fn.brightness_contrast(images, brightness=brightness_rate, contrast_center=74, contrast=contrast_rate)
#             images = fn.saturation(images, saturation=saturation_rate)

#         # images = fn.cast(images, dtype=types.FLOAT)
#         # images = fn.normalize(images, scale=1/255)
#         # images = fn.normalize(images, axes=[0,1], mean=mean, stddev=std)
#         images = fn.normalize(
#             images,
#             mean=MEAN / SCALE,
#             stddev=STD,
#             scale=SCALE,
#             dtype=types.FLOAT,
#         )
        
#         # if lb_map is not None: 
#         # print(lb_map)
#         # labels = fn.lookup_table(labels, keys=list(range(len(lb_map))), values=list(lb_map), default_value=255)
#         labels = fn.cast(labels, dtype=types.UINT8)
#         pipe.set_outputs(images, labels)
#     return pipe

# def get_DALI_data_loaderMul(configer, aux_mode='eval', stage=None):
#     mode = aux_mode
#     n_datasets = configer.get('n_datasets')
#     max_iter = configer.get('lr', 'max_iter')
    
#     if mode == 'train':
#         scales = configer.get('train', 'scales')
#         cropsize = configer.get('train', 'cropsize')
        
#         if stage != None:
#             annpath = [configer.get('dataset'+str(i), 'train_im_anns').replace('.txt', f'_{stage}.txt') for i in range(1, n_datasets+1)]
#             print(annpath)
#             batchsize = [configer.get('dataset'+str(i), 'ims_per_gpu') for i in range(1, n_datasets+1)]
#         else:
#             annpath = [configer.get('dataset'+str(i), 'train_im_anns') for i in range(1, n_datasets+1)]
#             batchsize = [configer.get('dataset'+str(i), 'ims_per_gpu') for i in range(1, n_datasets+1)]
#         imroot = [configer.get('dataset'+str(i), 'im_root') for i in range(1, n_datasets+1)]
#         data_reader = [configer.get('dataset'+str(i), 'dataset_name') for i in range(1, n_datasets+1)]
        
#         shuffle = True
#         drop_last = True
#     elif mode == 'eval':
        
#         batchsize = [configer.get('dataset'+str(i), 'eval_ims_per_gpu') for i in range(1, n_datasets+1)]
#         annpath = [configer.get('dataset'+str(i), 'val_im_anns') for i in range(1, n_datasets+1)]
#         imroot = [configer.get('dataset'+str(i), 'im_root') for i in range(1, n_datasets+1)]
#         data_reader = [configer.get('dataset'+str(i), 'dataset_name') for i in range(1, n_datasets+1)]
        
#         shuffle = False
#         drop_last = False
#     elif mode == 'ret_path':
        
#         batchsize = [1 for i in range(1, n_datasets+1)]
#         if stage != None:
#             annpath = [configer.get('dataset'+str(i), 'train_im_anns').replace('.txt', f'_{stage}.txt') for i in range(1, n_datasets+1)]
#             print(annpath)
#         else:
#             annpath = [configer.get('dataset'+str(i), 'train_im_anns') for i in range(1, n_datasets+1)]
            
#         imroot = [configer.get('dataset'+str(i), 'im_root') for i in range(1, n_datasets+1)]
#         data_reader = [configer.get('dataset'+str(i), 'dataset_name') for i in range(1, n_datasets+1)]
        
#         shuffle = False
#         drop_last = False
        

#     ds = ExternalInputIteratorMul(batchsize, imroot, annpath, mode=mode)
         
#     total_bs = 0
#     for bs in batchsize:
#         total_bs += bs
#     pipe = ExternalSourcePipelineMul(batch_size=total_bs, num_threads=64, device_id=0, external_data=ds, mode=mode)
#     # lb_maps = []
#     # for i, data_name in enumerate(data_reader):
#     #     label_info = eval(data_name).labels_info
#     #     lb_map = np.arange(256).astype(np.uint8)
       
#     #     for el in label_info:
#     #         lb_map[el['id']] = el['trainId']
        
#     #     lb_maps.append(lb_map)
        
        

#     if mode == 'train':
#         dl = PyTorchIterator(pipe, last_batch_padded=True, last_batch_policy=LastBatchPolicy.DROP) 
#     else:
#         dl = PyTorchIterator(pipe, last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL)

#     return dl


# def ExternalSourcePipeline(batch_size, num_threads, device_id, external_data, lb_map=None, mode='train', scales=(0.5, 1.), size=(768, 768), p=0.5, brightness=0.4, contrast=0.4, saturation=0.4):
    
#     pipe = Pipeline(batch_size, num_threads, device_id, prefetch_queue_depth=4)
#     crop_h, crop_w = size
#     if not brightness is None and brightness >= 0:
#         brightness = [max(1-brightness, 0), 1+brightness]
#     if not contrast is None and contrast >= 0:
#         contrast = [max(1-contrast, 0), 1+contrast]
#     if not saturation is None and saturation >= 0:
#         saturation = [max(1-saturation, 0), 1+saturation]

#     # mean=[0.3257, 0.3690, 0.3223] # city, rgb
#     # std=[0.2112, 0.2148, 0.2115]
#     MEAN = np.asarray([0.3257, 0.3690, 0.3223])[None, None, :]
#     STD = np.asarray([0.2112, 0.2148, 0.2115])[None, None, :]
#     SCALE = 1 / 255.
#     with pipe:
#         jpegs, labels = fn.external_source(source=external_data, num_outputs=2, dtype=types.UINT8)
#         images = fn.decoders.image(jpegs, device="mixed")
#         labels = fn.decoders.image(labels, device="cpu", output_type=types.GRAY)
#         # images = fn.random_resized_crop()
#         # for i in range(len(images)):
#         # print(fn.peek_image_shape(labels))
#         shape = fn.shapes(labels)
#         # images = images.gpu()
#         labels = labels.gpu()
#         if mode == 'train':
#             scale = np.random.uniform(min(scales), max(scales))
            
#             images = fn.resize(images, interp_type=types.INTERP_LINEAR, resize_x=shape[1]*scale, resize_y=shape[0]*scale)
#             labels = fn.resize(labels, antialias=False, interp_type=types.INTERP_NN, size=shape[:2]*scale)

#             crop_pos_x = fn.random.uniform(range=(0, 1))
#             crop_pos_y = fn.random.uniform(range=(0, 1))            

#             images = fn.crop(images, crop=size, crop_pos_x=crop_pos_x, crop_pos_y=crop_pos_y, out_of_bounds_policy='pad')
#             labels = fn.crop(labels, crop=size, crop_pos_x=crop_pos_x, crop_pos_y=crop_pos_y, fill_values=255, out_of_bounds_policy='pad')            
#         # labels = fn.decoders.image(labels, device="mixed", output_type=types.GRAY)
#         # # images = fn.random_resized_crop()
#         # # for i in range(len(images)):
#         # # print(fn.peek_image_shape(labels))
#         # # shape = fn.peek_image_shape(labels)
#         # # images = images.gpu()
#         # # labels = labels.gpu()
#         # if mode == 'train':
#         #     images = fn.random_resized_crop(images, interp_type=types.INTERP_LINEAR, size=size, seed=1234)
#         #     labels = fn.random_resized_crop(labels, antialias=False, interp_type=types.INTERP_NN, size=size, seed=1234)

#             brightness_rate = fn.random.uniform(range=(min(brightness), max(brightness)))
#             contrast_rate = fn.random.uniform(range=(min(contrast), max(contrast)))
#             saturation_rate = fn.random.uniform(range=(min(saturation), max(saturation)))
#             images = fn.brightness_contrast(images, brightness=brightness_rate, contrast_center=74, contrast=contrast_rate)
#             images = fn.saturation(images, saturation=saturation_rate)

#         # images = fn.cast(images, dtype=types.FLOAT)
#         # images = fn.normalize(images, scale=1/255)
#         # images = fn.normalize(images, axes=[0,1], mean=mean, stddev=std)
#         images = fn.normalize(
#             images,
#             mean=MEAN / SCALE,
#             stddev=STD,
#             scale=SCALE,
#             dtype=types.FLOAT,
#         )
        
#         if lb_map is not None: 
#         # print(lb_map)
#             labels = fn.lookup_table(labels, keys=list(range(len(lb_map))), values=list(lb_map), default_value=255)
#         labels = fn.cast(labels, dtype=types.UINT8)
#         pipe.set_outputs(images, labels)
#     return pipe

# def get_DALI_data_loader(configer, aux_mode='eval', stage=None):
#     mode = aux_mode
#     n_datasets = configer.get('n_datasets')
#     max_iter = configer.get('lr', 'max_iter')
    
#     if mode == 'train':
#         scales = configer.get('train', 'scales')
#         cropsize = configer.get('train', 'cropsize')
        
#         if stage != None:
#             annpath = [configer.get('dataset'+str(i), 'train_im_anns').replace('.txt', f'_{stage}.txt') for i in range(1, n_datasets+1)]
#             print(annpath)
#             batchsize = [configer.get('dataset'+str(i), 'ims_per_gpu') for i in range(1, n_datasets+1)]
#         else:
#             annpath = [configer.get('dataset'+str(i), 'train_im_anns') for i in range(1, n_datasets+1)]
#             batchsize = [configer.get('dataset'+str(i), 'ims_per_gpu') for i in range(1, n_datasets+1)]
#         imroot = [configer.get('dataset'+str(i), 'im_root') for i in range(1, n_datasets+1)]
#         data_reader = [configer.get('dataset'+str(i), 'dataset_name') for i in range(1, n_datasets+1)]
        
#         shuffle = True
#         drop_last = True
#     elif mode == 'eval':
        
#         batchsize = [configer.get('dataset'+str(i), 'eval_ims_per_gpu') for i in range(1, n_datasets+1)]
#         annpath = [configer.get('dataset'+str(i), 'val_im_anns') for i in range(1, n_datasets+1)]
#         imroot = [configer.get('dataset'+str(i), 'im_root') for i in range(1, n_datasets+1)]
#         data_reader = [configer.get('dataset'+str(i), 'dataset_name') for i in range(1, n_datasets+1)]
        
#         shuffle = False
#         drop_last = False
#     elif mode == 'ret_path':
        
#         batchsize = [1 for i in range(1, n_datasets+1)]
#         if stage != None:
#             annpath = [configer.get('dataset'+str(i), 'train_im_anns').replace('.txt', f'_{stage}.txt') for i in range(1, n_datasets+1)]
#             print(annpath)
#         else:
#             annpath = [configer.get('dataset'+str(i), 'train_im_anns') for i in range(1, n_datasets+1)]
            
#         imroot = [configer.get('dataset'+str(i), 'im_root') for i in range(1, n_datasets+1)]
#         data_reader = [configer.get('dataset'+str(i), 'dataset_name') for i in range(1, n_datasets+1)]
        
#         shuffle = False
#         drop_last = False
        

#     ds = [ExternalInputIterator(bs, root, path, mode=mode)
#           for bs, root, path in zip(batchsize, imroot, annpath)]
    
#     pipes = []
#     for i, data_name in enumerate(data_reader):
#         label_info = eval(data_name+'_labels_info')
#         lb_map = np.arange(256).astype(np.uint8)
#         # print(lb_map)
#         for el in label_info:
#             lb_map[el['id']] = el['trainId']
#         pipe = ExternalSourcePipeline(batch_size=batchsize[i], num_threads=8, device_id=0, external_data=ds[i], lb_map=lb_map, mode=mode)
#         pipes.append(pipe)

#     if mode == 'train':
#         dl = [PyTorchIterator(pipe, last_batch_padded=True, last_batch_policy=LastBatchPolicy.DROP) for pipe in pipes]
#     else:
#         dl = [PyTorchIterator(pipe, last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL) for pipe in pipes]

#     return dl

# class DaLiLoaderAdapter:
#     def __init__(self, cfg, aux_mode='train', dataset_id=None) -> None:
#         self.configer = Configer(configs=cfg.DATASETS.CONFIGER)
#         self.max_iters = cfg.SOLVER.MAX_ITER + 10
#         self.dls = get_DALI_data_loader(self.configer, aux_mode)
#         self.n = 0
#         self.dataset_id = dataset_id
#         self.aux_mode = aux_mode
    
#     def __iter__(self):
#         self.n = 0
#         if self.aux_mode == 'train':
#             self.dl_iters = [iter(dl) for dl in self.dls]
#         else:
#             self.dl_iters = [iter(self.dls[self.dataset_id])]
#         return self
    
#     def __len__(self):
#         if self.aux_mode == 'train':
#             return self.max_iters
#         else:
#             return len(self.dl_iters[0])
        
    
#     def __next__(self):
#         self.n += 1
#         if self.n < self.__len__():
#             ims = []
#             lbs = []
#             for j in range(0,len(self.dl_iters)):

#                 if self.aux_mode == 'train':
#                     try:
#                         data = next(self.dl_iters[j])
#                         im = data[0]['data']
#                         lb = data[0]['label']
#                         if not im.size()[0] == self.configer.get('dataset'+str(j+1), 'ims_per_gpu'):
#                             raise StopIteration
#                         while torch.min(lb) == 255:
#                             data = next(self.dl_iters[j])
#                             im = data[0]['data']
#                             lb = data[0]['label']
#                             if not im.size()[0] == self.configer.get('dataset'+str(j+1), 'ims_per_gpu'):
#                                 raise StopIteration


#                     except StopIteration:
#                         self.dl_iters[j] = iter(self.dls[j])
#                         data = next(self.dl_iters[j])
#                         im = data[0]['data']
#                         lb = data[0]['label']
#                         while torch.min(lb) == 255:
#                             print(f"{j}:stop while")
#                             data = next(self.dl_iters[j])
#                             im = data[0]['data']
#                             lb = data[0]['label']
#                 else:
#                     data = next(self.dl_iters[j])
#                     im = data[0]['data']
#                     lb = data[0]['label']
                            
#                 ims.append(im)
#                 lbs.append(lb)
                
                        
#             im = torch.cat(ims, dim=0)
#             lb = torch.cat(lbs, dim=0)
#             im = im.permute(0,3,1,2).contiguous()

#             if self.aux_mode == 'train':
#                 dataset_lbs = torch.cat([i*torch.ones(this_lb.shape[0], dtype=torch.int) for i,this_lb in enumerate(lbs)], dim=0)
#             else:
#                 dataset_lbs = self.dataset_id
#             lb = torch.squeeze(lb, 3).long()
#             batch_inputs = {
#                 'image': im,
#                 'sem_seg': lb,
#                 'dataset_lbs': dataset_lbs
#             }
#             return batch_inputs
#         else:
#             raise StopIteration
        
    

class LoaderAdapter:
    def __init__(self, cfg, aux_mode='train', dataset_id=None, datasets_name=None) -> None:
        print(f"LoaderAdapter:{aux_mode}, dataset_id:{dataset_id}")
        if datasets_name is not None:
            self.datasets_name = datasets_name
        else:
            if aux_mode == 'train':
                self.datasets_name = cfg.DATASETS.TRAIN
            elif aux_mode == 'eval':
                self.datasets_name = cfg.DATASETS.EVAL
            elif aux_mode == 'unseen':
                self.datasets_name = cfg.DATASETS.TRAIN
            else:
                self.datasets_name = cfg.DATASETS.TEST
        
        dataset = [get_detection_dataset_dicts(
            name,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        ) for name in self.datasets_name]
        # _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])
        
        # if cfg.INPUT.DATASET_MAPPER_NAME == 'SemanticDatasetMapper':
        mapper = []
        for i in range(len(dataset)):
            should_lkt = True

            mapper.append(SemanticDatasetMapper(cfg, True, i, should_lkt))

        if aux_mode == 'train':
            self.dls = [build_detection_train_loader(cfg, mapper=mp, dataset=ds) for ds, mp in zip(dataset, mapper)]
        # elif aux_mode == 'eval':
        #     mapper = SemanticDatasetMapper(cfg, False, dataset_id, True)
        #     Log.info(f"evaluate {self.datasets_name[dataset_id]}")
        #     self.dls = build_detection_test_loader(cfg, dataset_name=self.datasets_name[dataset_id], mapper=mapper)            
        else: # aux_mode == 'eval':
            
            # if dataset_id > 0:
            #     mapper = SemanticDatasetMapper(cfg, False, dataset_id-1, True)
            #     Log.info(f"evaluate {self.datasets_name[dataset_id-1]}")
            #     self.dls = build_detection_test_loader(cfg, dataset_name=self.datasets_name[dataset_id-1], mapper=mapper)
            # else:
            if dataset_id >= len(self.datasets_name):
                mapper = SemanticDatasetMapper(cfg, False, dataset_id, True)
                Log.info(f"evaluate {self.datasets_name[0]}")
                self.dls = build_detection_test_loader(cfg, dataset_name=self.datasets_name[0], mapper=mapper)            
            else:
                mapper = SemanticDatasetMapper(cfg, False, dataset_id, True)
                Log.info(f"evaluate {self.datasets_name[dataset_id]}")
                self.dls = build_detection_test_loader(cfg, dataset_name=self.datasets_name[dataset_id], mapper=mapper)
        # else:
        #     self.dls = build_detection_test_loader(cfg, dataset_name=self.datasets_name[dataset_id])
            
        
        # self.configer = Configer(configs=cfg.DATASETS.CONFIGER)
        self.max_iters = cfg.SOLVER.MAX_ITER + 10
        # self.dls = get_DALI_data_loader(self.configer, aux_mode)
        self.n = 0
        self.dataset_id = dataset_id
        self.aux_mode = aux_mode
    
    def __iter__(self):
        self.n = 0
        if self.aux_mode == 'train':
            self.dl_iters = [iter(dl) for dl in self.dls]
        else:
            self.dl_iters =iter(self.dls)
        return self
    
    def __len__(self):
        if self.aux_mode == 'train':
            return self.max_iters
        else:
            return len(self.dls)
        
    
    def __next__(self):
        self.n += 1
        if self.n < self.__len__():
            datas = []
            ids = []
            if self.aux_mode == 'train':
                for j in range(0,len(self.dl_iters)):
                    try:
                        data = next(self.dl_iters[j])

                    except StopIteration:
                        self.dl_iters[j] = iter(self.dls[j])
                        data = next(self.dl_iters[j])
                    
                    for i in range(len(data)):
                        data[i]['dataset_id'] = j
                    
                    datas.extend(data)
            else:
                data = next(self.dl_iters)
                for i in range(len(data)):
                    data[i]['dataset_id'] = self.dataset_id
                        
                datas.extend(data)


            # if self.aux_mode == 'train':
            #     dataset_lbs = torch.tensor(ids)
            # else:
            #     dataset_lbs = self.dataset_id

            return datas
        else:
            raise StopIteration
        

def Mask2formerAdapter(sem_seg_gts, ignore_label=255):
    out_instances = []
    for sem_seg_gt in sem_seg_gts:
        sem_seg_gt = sem_seg_gt.numpy()
        image_shape = sem_seg_gt.shape
        instances = Instances(image_shape)
        classes = np.unique(sem_seg_gt)
        # remove ignored region
        mod_classes = np.mod(classes, 256)
        mod_classes = mod_classes[mod_classes != ignore_label]
        classes = classes[mod_classes != ignore_label]
        instances.gt_classes = torch.tensor(mod_classes, dtype=torch.int64)

        masks = []
        for class_id in classes:
            masks.append(sem_seg_gt == class_id)

        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks.tensor
        out_instances.append(instances)

    return out_instances