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


labels_info = [
{"name": "Bird", "id": 0, "color": [165, 42, 42], "trainId": 0},
{"name": "Ground Animal", "id": 1, "color": [0, 192, 0], "trainId": 1},
{"name": "Ambiguous Barrier", "id": 2, "color": [250, 170, 31], "trainId": 2},
{"name": "Concrete Block", "id": 3, "color": [250, 170, 32], "trainId": 3},
{"name": "Curb", "id": 4, "color": [196, 196, 196], "trainId": 4},
{"name": "Fence", "id": 5, "color": [190, 153, 153], "trainId": 5},
{"name": "Guard Rail", "id": 6, "color": [180, 165, 180], "trainId": 6},
{"name": "Barrier", "id": 7, "color": [90, 120, 150], "trainId": 7},
{"name": "Road Median", "id": 8, "color": [250, 170, 33], "trainId": 8},
{"name": "Road Side", "id": 9, "color": [250, 170, 34], "trainId": 9},
{"name": "Lane Separator", "id": 10, "color": [128, 128, 128], "trainId": 10},
{"name": "Temporary Barrier", "id": 11, "color": [250, 170, 35], "trainId": 11},
{"name": "Wall", "id": 12, "color": [102, 102, 156], "trainId": 12},
{"name": "Bike Lane", "id": 13, "color": [128, 64, 255], "trainId": 13},
{"name": "Crosswalk - Plain", "id": 14, "color": [140, 140, 200], "trainId": 14},
{"name": "Curb Cut", "id": 15, "color": [170, 170, 170], "trainId": 15},
{"name": "Driveway", "id": 16, "color": [250, 170, 36], "trainId": 16},
{"name": "Parking", "id": 17, "color": [250, 170, 160], "trainId": 17},
{"name": "Parking Aisle", "id": 18, "color": [250, 170, 37], "trainId": 18},
{"name": "Pedestrian Area", "id": 19, "color": [96, 96, 96], "trainId": 19},
{"name": "Rail Track", "id": 20, "color": [230, 150, 140], "trainId": 20},
{"name": "Road", "id": 21, "color": [128, 64, 128], "trainId": 21},
{"name": "Road Shoulder", "id": 22, "color": [110, 110, 110], "trainId": 22},
{"name": "Service Lane", "id": 23, "color": [110, 110, 110], "trainId": 23},
{"name": "Sidewalk", "id": 24, "color": [244, 35, 232], "trainId": 24},
{"name": "Traffic Island", "id": 25, "color": [128, 196, 128], "trainId": 25},
{"name": "Bridge", "id": 26, "color": [150, 100, 100], "trainId": 26},
{"name": "Building", "id": 27, "color": [70, 70, 70], "trainId": 27},
{"name": "Garage", "id": 28, "color": [150, 150, 150], "trainId": 28},
{"name": "Tunnel", "id": 29, "color": [150, 120, 90], "trainId": 29},
{"name": "Person", "id": 30, "color": [220, 20, 60], "trainId": 30},
{"name": "Person Group", "id": 31, "color": [220, 20, 60], "trainId": 31},
{"name": "Bicyclist", "id": 32, "color": [255, 0, 0], "trainId": 32},
{"name": "Motorcyclist", "id": 33, "color": [255, 0, 100], "trainId": 33},
{"name": "Other Rider", "id": 34, "color": [255, 0, 200], "trainId": 34},
{"name": "Lane Marking - Dashed Line", "id": 35, "color": [255, 255, 255], "trainId": 35},
{"name": "Lane Marking - Straight Line", "id": 36, "color": [255, 255, 255], "trainId": 36},
{"name": "Lane Marking - Zigzag Line", "id": 37, "color": [250, 170, 29], "trainId": 37},
{"name": "Lane Marking - Ambiguous", "id": 38, "color": [250, 170, 28], "trainId": 38},
{"name": "Lane Marking - Arrow (Left)", "id": 39, "color": [250, 170, 26], "trainId": 39},
{"name": "Lane Marking - Arrow (Other)", "id": 40, "color": [250, 170, 25], "trainId": 40},
{"name": "Lane Marking - Arrow (Right)", "id": 41, "color": [250, 170, 24], "trainId": 41},
{"name": "Lane Marking - Arrow (Split Left or Straight)", "id": 42, "color": [250, 170, 22], "trainId": 42},
{"name": "Lane Marking - Arrow (Split Right or Straight)", "id": 43, "color": [250, 170, 21], "trainId": 43},
{"name": "Lane Marking - Arrow (Straight)", "id": 44, "color": [250, 170, 20], "trainId": 44},
{"name": "Lane Marking - Crosswalk", "id": 45, "color": [255, 255, 255], "trainId": 45},
{"name": "Lane Marking - Give Way (Row)", "id": 46, "color": [250, 170, 19], "trainId": 46},
{"name": "Lane Marking - Give Way (Single)", "id": 47, "color": [250, 170, 18], "trainId": 47},
{"name": "Lane Marking - Hatched (Chevron)", "id": 48, "color": [250, 170, 12], "trainId": 48},
{"name": "Lane Marking - Hatched (Diagonal)", "id": 49, "color": [250, 170, 11], "trainId": 49},
{"name": "Lane Marking - Other", "id": 50, "color": [255, 255, 255], "trainId": 50},
{"name": "Lane Marking - Stop Line", "id": 51, "color": [255, 255, 255], "trainId": 51},
{"name": "Lane Marking - Symbol (Bicycle)", "id": 52, "color": [250, 170, 16], "trainId": 52},
{"name": "Lane Marking - Symbol (Other)", "id": 53, "color": [250, 170, 15], "trainId": 53},
{"name": "Lane Marking - Text", "id": 54, "color": [250, 170, 15], "trainId": 54},
{"name": "Lane Marking (only) - Dashed Line", "id": 55, "color": [255, 255, 255], "trainId": 55},
{"name": "Lane Marking (only) - Crosswalk", "id": 56, "color": [255, 255, 255], "trainId": 56},
{"name": "Lane Marking (only) - Other", "id": 57, "color": [255, 255, 255], "trainId": 57},
{"name": "Lane Marking (only) - Test", "id": 58, "color": [255, 255, 255], "trainId": 58},
{"name": "Mountain", "id": 59, "color": [64, 170, 64], "trainId": 59},
{"name": "Sand", "id": 60, "color": [230, 160, 50], "trainId": 60},
{"name": "Sky", "id": 61, "color": [70, 130, 180], "trainId": 61},
{"name": "Snow", "id": 62, "color": [190, 255, 255], "trainId": 62},
{"name": "Terrain", "id": 63, "color": [152, 251, 152], "trainId": 63},
{"name": "Vegetation", "id": 64, "color": [107, 142, 35], "trainId": 64},
{"name": "Water", "id": 65, "color": [0, 170, 30], "trainId": 65},
{"name": "Banner", "id": 66, "color": [255, 255, 128], "trainId": 66},
{"name": "Bench", "id": 67, "color": [250, 0, 30], "trainId": 67},
{"name": "Bike Rack", "id": 68, "color": [100, 140, 180], "trainId": 68},
{"name": "Catch Basin", "id": 69, "color": [220, 128, 128], "trainId": 69},
{"name": "CCTV Camera", "id": 70, "color": [222, 40, 40], "trainId": 70},
{"name": "Fire Hydrant", "id": 71, "color": [100, 170, 30], "trainId": 71},
{"name": "Junction Box", "id": 72, "color": [40, 40, 40], "trainId": 72},
{"name": "Mailbox", "id": 73, "color": [33, 33, 33], "trainId": 73},
{"name": "Manhole", "id": 74, "color": [100, 128, 160], "trainId": 74},
{"name": "Parking Meter", "id": 75, "color": [20, 20, 255], "trainId": 75},
{"name": "Phone Booth", "id": 76, "color": [142, 0, 0], "trainId": 76},
{"name": "Pothole", "id": 77, "color": [70, 100, 150], "trainId": 77},
{"name": "Signage - Advertisement", "id": 78, "color": [250, 171, 30], "trainId": 78},
{"name": "Signage - Ambiguous", "id": 79, "color": [250, 172, 30], "trainId": 79},
{"name": "Signage - Back", "id": 80, "color": [250, 173, 30], "trainId": 80},
{"name": "Signage - Information", "id": 81, "color": [250, 174, 30], "trainId": 81},
{"name": "Signage - Other", "id": 82, "color": [250, 175, 30], "trainId": 82},
{"name": "Signage - Store", "id": 83, "color": [250, 176, 30], "trainId": 83},
{"name": "Street Light", "id": 84, "color": [210, 170, 100], "trainId": 84},
{"name": "Pole", "id": 85, "color": [153, 153, 153], "trainId": 85},
{"name": "Pole Group", "id": 86, "color": [153, 153, 153], "trainId": 86},
{"name": "Traffic Sign Frame", "id": 87, "color": [128, 128, 128], "trainId": 87},
{"name": "Utility Pole", "id": 88, "color": [0, 0, 80], "trainId": 88},
{"name": "Traffic Cone", "id": 89, "color": [210, 60, 60], "trainId": 89},
{"name": "Traffic Light - General (Single)", "id": 90, "color": [250, 170, 30], "trainId": 90},
{"name": "Traffic Light - Pedestrians", "id": 91, "color": [250, 170, 30], "trainId": 91},
{"name": "Traffic Light - General (Upright)", "id": 92, "color": [250, 170, 30], "trainId": 92},
{"name": "Traffic Light - General (Horizontal)", "id": 93, "color": [250, 170, 30], "trainId": 93},
{"name": "Traffic Light - Cyclists", "id": 94, "color": [250, 170, 30], "trainId": 94},
{"name": "Traffic Light - Other", "id": 95, "color": [250, 170, 30], "trainId": 95},
{"name": "Traffic Sign - Ambiguous", "id": 96, "color": [192, 192, 192], "trainId": 96},
{"name": "Traffic Sign (Back)", "id": 97, "color": [192, 192, 192], "trainId": 97},
{"name": "Traffic Sign - Direction (Back)", "id": 98, "color": [192, 192, 192], "trainId": 98},
{"name": "Traffic Sign - Direction (Front)", "id": 99, "color": [220, 220, 0], "trainId": 99},
{"name": "Traffic Sign (Front)", "id": 100, "color": [220, 220, 0], "trainId": 100},
{"name": "Traffic Sign - Parking", "id": 101, "color": [0, 0, 196], "trainId": 101},
{"name": "Traffic Sign - Temporary (Back)", "id": 102, "color": [192, 192, 192], "trainId": 102},
{"name": "Traffic Sign - Temporary (Front)", "id": 103, "color": [220, 220, 0], "trainId": 103},
{"name": "Trash Can", "id": 104, "color": [140, 140, 20], "trainId": 104},
{"name": "Bicycle", "id": 105, "color": [119, 11, 32], "trainId": 105},
{"name": "Boat", "id": 106, "color": [150, 0, 255], "trainId": 106},
{"name": "Bus", "id": 107, "color": [0, 60, 100], "trainId": 107},
{"name": "Car", "id": 108, "color": [0, 0, 142], "trainId": 108},
{"name": "Caravan", "id": 109, "color": [0, 0, 90], "trainId": 109},
{"name": "Motorcycle", "id": 110, "color": [0, 0, 230], "trainId": 110},
{"name": "On Rails", "id": 111, "color": [0, 80, 100], "trainId": 111},
{"name": "Other Vehicle", "id": 112, "color": [128, 64, 64], "trainId": 112},
{"name": "Trailer", "id": 113, "color": [0, 0, 110], "trainId": 113},
{"name": "Truck", "id": 114, "color": [0, 0, 70], "trainId": 114},
{"name": "Vehicle Group", "id": 115, "color": [0, 0, 142], "trainId": 115},
{"name": "Wheeled Slow", "id": 116, "color": [0, 0, 192], "trainId": 116},
{"name": "Water Valve", "id": 117, "color": [170, 170, 170], "trainId": 117},
{"name": "Car Mount", "id": 118, "color": [32, 32, 32], "trainId": 118},
{"name": "Dynamic", "id": 119, "color": [111, 74, 0], "trainId": 119},
{"name": "Ego Vehicle", "id": 120, "color": [120, 10, 10], "trainId": 120},
{"name": "Ground", "id": 121, "color": [81, 0, 81], "trainId": 121},
{"name": "Static", "id": 122, "color": [111, 111, 0], "trainId": 122},
{"name": "Unlabeled", "id": 123, "color": [0, 0, 0], "trainId": 255},
]

import json
from PIL import Image, ImageDraw
import numpy as np




class MapiData(Dataset):
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(MapiData, self).__init__()
        # assert mode in ('train', 'eval', 'test')

        self.mode = mode
        self.trans_func = trans_func
        self.n_cats = 123
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
    dataroot = "./"
    annpath = "index/mapi/validation.txt"
    mapi = MapiData(dataroot, annpath)
    index = 0
    for i in mapi:
        index += 1
        if index % 100 == 0:
            print(f'finish:{index}')
