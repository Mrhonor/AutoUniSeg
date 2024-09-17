import os
import os.path as osp
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import cv2
from PIL import Image
import numpy as np
import time


import json
from PIL import Image, ImageDraw
import numpy as np


labels_info = [
    {"name": "unlabeled", "id": 0, "color": [0,0,0], "trainId":255},
    {"name": "ambiguous", "id": 1, "color": [111,74,0], "trainId":255},
    {"name": "sky", "id": 2, "color": [70,130,180], "trainId":0},
    {"name": "road", "id": 3, "color": [128,64,128], "trainId":1},
    {"name": "sidewalk", "id": 4, "color": [244,35,232], "trainId":2},
    {"name": "railtrack", "id": 5, "color": [230,150,140], "trainId":255},
    {"name": "terrain", "id": 6, "color": [152,251,152], "trainId":3},
    {"name": "tree", "id": 7, "color": [87,182,35], "trainId":4},
    {"name": "vegetation", "id": 8, "color": [35,142,35], "trainId":5},
    {"name": "building", "id": 9, "color": [70,70,70], "trainId":6},
    {"name": "infrastructure", "id": 10, "color": [153,153,153], "trainId":7},
    {"name": "fence", "id": 11, "color": [190,153,153], "trainId":8},
    {"name": "billboard", "id": 12, "color": [150,20,20], "trainId":9},
    {"name": "trafficlight", "id": 13, "color": [250,170,30], "trainId":10},
    {"name": "trafficsign", "id": 14, "color": [220,220,0], "trainId":11},
    {"name": "mobilebarrier", "id": 15, "color": [180,180,100], "trainId":12},
    {"name": "firehydrant", "id": 16, "color": [173,153,153], "trainId":13},
    {"name": "chair", "id": 17, "color": [168,153,153], "trainId":14},
    {"name": "trash", "id": 18, "color": [81,0,21], "trainId":15},
    {"name": "trashcan", "id": 19, "color": [81,0,81], "trainId":16},
    {"name": "person", "id": 20, "color": [220,20,60], "trainId":17},
    {"name": "animal", "id": 21, "color": [255,0,0], "trainId":255},
    {"name": "bicycle", "id": 22, "color": [119,11,32], "trainId":255},
    {"name": "motorcycle", "id": 23, "color": [0,0,230], "trainId":18},
    {"name": "car", "id": 24, "color": [0,0,142], "trainId":19},
    {"name": "van", "id": 25, "color": [0,80,100], "trainId":20},
    {"name": "bus", "id": 26, "color": [0,60,100], "trainId":21},
    {"name": "truck", "id": 27, "color": [0,0,70], "trainId":22},
    {"name": "trailer", "id": 28, "color": [0,0,90], "trainId":255},
    {"name": "train", "id": 29, "color": [0,80,100], "trainId":255},
    {"name": "plane", "id": 30, "color": [0,100,100], "trainId":255},
    {"name": "boat", "id": 31, "color": [50,0,90], "trainId":255},
]



class ViperData(Dataset):
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(ViperData, self).__init__()
        # assert mode in ('train', 'eval', 'test')

        self.mode = mode
        self.trans_func = trans_func
        self.n_cats = 32
        self.lb_map = np.arange(256).astype(np.uint8)

        for el in labels_info:
            self.lb_map[el['id']] = el['trainId']

        self.ignore_lb = -1

        with open(annpath, 'r') as fr:
            pairs = fr.read().splitlines()

        self.img_paths, self.lb_paths = [], []
        for pair in pairs:
            imgpth, lbpth = pair.split(',')
            self.img_paths.append(osp.join(dataroot, imgpth))
            self.lb_paths.append(osp.join(dataroot, lbpth))

        assert len(self.img_paths) == len(self.lb_paths)
        self.len = len(self.img_paths)

        
        self.colors = []

        for el in labels_info:
            (r, g, b) = el['color']
            self.colors.append((r, g, b))
            
        self.color2id = dict(zip(self.colors, range(len(self.colors))))

    def __getitem__(self, idx):
        # impth = self.img_paths[idx]
        lbpth = self.lb_paths[idx]
        
        # start = time.time()
        # img = cv2.imread(impth)[:, :, ::-1]

        # img = cv2.resize(img, (1920, 1280))
        label = np.array(Image.open(lbpth).convert('RGB'))
        # end = time.time()
        # print("idx: {}, cv2.imread time: {}".format(idx, end - start))
        # label = np.array(Image.open(lbpth).convert('RGB').resize((1920, 1280),Image.ANTIALIAS))
        
        # start = time.time()
        label = self.convert_labels(label, None)
        # label = Image.fromarray(label)
        new_lbpth = lbpth.replace(".png", "_L.png")
        cv2.imwrite(new_lbpth, label)
        # im = np.array(cv2.imread(new_lbpth, cv2.IMREAD_GRAYSCALE))
        # print(im.shape)
        # print(label.shape)
        # if (im == label).any() == False:
        #     print("wrong!")
        
        return idx
        # end = time.time()
        # print("idx: {}, convert_labels time: {}".format(idx, end - start))

        
        # return img.detach(), label.unsqueeze(0).detach()
        # return img.detach()

    def __len__(self):
        return self.len

    def convert_labels(self, label, impth):
        mask = np.full(label.shape[:2], 2, dtype=np.uint8)
        # mask = np.zeros(label.shape[:2])
        for k, v in self.color2id.items():
            mask[cv2.inRange(label, np.array(k) - 1, np.array(k) + 1) == 255] = v
            
            
            # if v == 30 and cv2.inRange(label, np.array(k) - 1, np.array(k) + 1).any() == True:
            #     label[cv2.inRange(label, np.array(k) - 1, np.array(k) + 1) == 255] = [0, 0, 0]
            #     cv2.imshow(impth, label)
            #     cv2.waitKey(0)
        return mask

if __name__ == "__main__":
    dataroot = "./val/"
    annpath = "index/viper/val.txt"
    mapi = ViperData(dataroot, annpath)
    index = 0
    for i in mapi:
        index += 1
        if index % 100 == 0:
            print(f'finish:{index}')
