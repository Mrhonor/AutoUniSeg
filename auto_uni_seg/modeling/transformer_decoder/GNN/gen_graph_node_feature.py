import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as osp
import sys
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)

import clip

# from lib.get_dataloader import get_data_loader
import cv2
import numpy as np
import matplotlib.pyplot as plt
import clip
from PIL import Image
import torch.nn.functional as F
import random
import pickle
from tqdm import tqdm
import torch.distributed as dist
from ....utils.configer import Configer
# from configer import Configer
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

class BaseDataset(Dataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(BaseDataset, self).__init__()
        # assert mode in ('train', 'eval', 'test')
        self.mode = mode
        self.trans_func = trans_func

        self.lb_map = None

        with open(annpath, 'r') as fr:
            pairs = fr.read().splitlines()
        self.img_paths, self.lb_paths = [], []
        for pair in pairs:
            imgpth, lbpth = pair.split(',')
            self.img_paths.append(osp.join(dataroot, imgpth))
            self.lb_paths.append(osp.join(dataroot, lbpth))

        assert len(self.img_paths) == len(self.lb_paths)
        self.len = len(self.img_paths)

    def __getitem__(self, idx):
        impth, lbpth = self.img_paths[idx], self.lb_paths[idx]        
        label = self.get_label(lbpth)
        if not self.lb_map is None:
            label = self.lb_map[label]
        if self.mode == 'ret_path':
            return impth, label, lbpth

        img = self.get_image(impth)

        im_lb = dict(im=img, lb=label)
        if not self.trans_func is None:
            im_lb = self.trans_func(im_lb)
            
        im_lb = self.to_tensor(im_lb)
        img, label = im_lb['im'], im_lb['lb']
        # self.to_tensor = T.ToTensorCUDA(
        #     mean=(0.3038, 0.3383, 0.3034), # city, rgb
        #     std=(0.2071, 0.2088, 0.2090),
        # )
        # img, label = self.to_tensor(img, label)
        # return img.copy(), label[None].copy()
        
        return img.detach(), label.unsqueeze(0).detach()
        # return img.detach()

    def get_label(self, lbpth):
        return cv2.imread(lbpth, 0)

    def get_image(self, impth):
        img = cv2.imread(impth)[:, :, ::-1]
        return img

    def __len__(self):
        return self.len

class ToTensor(object):
    '''
    mean and std should be of the channel order 'bgr'
    '''
    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        im = im.transpose(2, 0, 1).astype(np.float32)
        im = torch.from_numpy(im).div_(255)
        dtype, device = im.dtype, im.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]
        im = im.sub_(mean).div_(std).clone()
        if not lb is None:
            lb = torch.from_numpy(lb.astype(np.int64))#.clone()
        return dict(im=im, lb=lb)

labels_info = [{'name': 'unlabeled', 'id': 0, 'evaluate': False, 'trainId': 255},
{'name': 'egovehicle', 'id': 1, 'evaluate': True, 'trainId': 0}, 
{'name': 'overlay', 'id': 2, 'evaluate': False, 'trainId': 255},
{'name': 'outofroi', 'id': 3, 'evaluate': False, 'trainId': 255},
{'name': 'static', 'id': 4, 'evaluate': False, 'trainId': 255},
{'name': 'dynamic', 'id': 5, 'evaluate': False, 'trainId': 255},
{'name': 'ground', 'id': 6, 'evaluate': False, 'trainId': 255},
{'name': 'road', 'id': 7, 'evaluate': True, 'trainId': 1},
{'name': 'sidewalk', 'id': 8, 'evaluate': True, 'trainId': 2},
{'name': 'parking', 'id': 9, 'evaluate': False, 'trainId': 255},
{'name': 'railtrack', 'id': 10, 'evaluate': False, 'trainId': 255},
{'name': 'building', 'id': 11, 'evaluate': True, 'trainId': 3},
{'name': 'wall', 'id': 12, 'evaluate': True, 'trainId': 4},
{'name': 'fence', 'id': 13, 'evaluate': True, 'trainId': 5},
{'name': 'guardrail', 'id': 14, 'evaluate': True, 'trainId': 6},
{'name': 'bridge', 'id': 15, 'evaluate': False, 'trainId': 255},
{'name': 'tunnel', 'id': 16, 'evaluate': False, 'trainId': 255},
{'name': 'pole', 'id': 17, 'evaluate': True, 'trainId': 7},
{'name': 'polegroup', 'id': 18, 'evaluate': False, 'trainId': 255},
{'name': 'trafficlight', 'id': 19, 'evaluate': True, 'trainId': 8},
{'name': 'trafficsignfront', 'id': 20, 'evaluate': True, 'trainId': 9},
{'name': 'vegetation', 'id': 21, 'evaluate': True, 'trainId': 10},
{'name': 'terrain', 'id': 22, 'evaluate': True, 'trainId': 11},
{'name': 'sky', 'id': 23, 'evaluate': True, 'trainId': 12},
{'name': 'person', 'id': 24, 'evaluate': True, 'trainId': 13},
{'name': 'rider', 'id': 25, 'evaluate': True, 'trainId': 14},
{'name': 'car', 'id': 26, 'evaluate': True, 'trainId': 15},
{'name': 'truck', 'id': 27, 'evaluate': True, 'trainId': 16},
{'name': 'bus', 'id': 28, 'evaluate': True, 'trainId': 17},
{'name': 'caravan', 'id': 29, 'evaluate': False, 'trainId': 255},
{'name': 'trailer', 'id': 30, 'evaluate': False, 'trainId': 255},
{'name': 'onrails', 'id': 31, 'evaluate': False, 'trainId': 255},
{'name': 'motorcycle', 'id': 32, 'evaluate': True, 'trainId': 18},
{'name': 'bicycle', 'id': 33, 'evaluate': True, 'trainId': 19},
{'name': 'pickup', 'id': 34, 'evaluate': True, 'trainId': 20},
{'name': 'van', 'id': 35, 'evaluate': True, 'trainId': 21},
{'name': 'billboard', 'id': 36, 'evaluate': True, 'trainId': 22},
{'name': 'streetlight', 'id': 37, 'evaluate': True, 'trainId': 23},
{'name': 'roadmarking', 'id': 38, 'evaluate': True, 'trainId': 24},
{'name': 'junctionbox', 'id': 39, 'evaluate': False,'trainId': 255},
{'name': 'mailbox', 'id': 40, 'evaluate': False, 'trainId': 255},
{'name': 'manhole', 'id': 41, 'evaluate': False, 'trainId': 255},
{'name': 'phonebooth', 'id': 42, 'evaluate': False, 'trainId': 255},
{'name': 'pothole', 'id': 43, 'evaluate': False, 'trainId': 255},
{'name': 'bikerack', 'id': 44, 'evaluate': False, 'trainId': 255},
{'name': 'trafficsignframe', 'id': 45, 'evaluate': True, 'trainId': 25},
{'name': 'utilitypole', 'id': 46, 'evaluate': True, 'trainId': 26},
{'name': 'motorcyclist', 'id': 47, 'evaluate': True, 'trainId': 14},
{'name': 'bicyclist', 'id': 48, 'evaluate': True, 'trainId': 14},
{'name': 'otherrider', 'id': 49, 'evaluate': True, 'trainId': 14},
{'name': 'bird', 'id': 50, 'evaluate': True, 'trainId': 30},
{'name': 'groundanimal', 'id': 51, 'evaluate': True, 'trainId': 31},
{'name': 'curb', 'id': 52, 'evaluate': True, 'trainId': 32},
{'name': 'trafficsignany', 'id': 53, 'evaluate': True, 'trainId': 33},
{'name': 'trafficsignback', 'id': 54, 'evaluate': True, 'trainId': 34},
{'name': 'trashcan', 'id': 55, 'evaluate': True, 'trainId': 35},
{'name': 'otherbarrier', 'id': 56, 'evaluate': True, 'trainId': 36},
{'name': 'othervehicle', 'id': 57, 'evaluate': True, 'trainId': 37},
{'name': 'autorickshaw', 'id': 58, 'evaluate': True, 'trainId': 38},
{'name': 'bench', 'id': 59, 'evaluate': True, 'trainId': 39},
{'name': 'mountain', 'id': 60, 'evaluate': True, 'trainId': 40},
{'name': 'tramtrack', 'id': 61, 'evaluate': True, 'trainId': 41},
{'name': 'wheeledslow', 'id': 62, 'evaluate': True, 'trainId': 42},
{'name': 'boat', 'id': 63, 'evaluate': True, 'trainId': 43},
{'name': 'bikelane', 'id': 64, 'evaluate': True, 'trainId': 44},
{'name': 'bikelanesidewalk', 'id': 65, 'evaluate': True, 'trainId': 45},
{'name': 'banner', 'id': 66, 'evaluate': True, 'trainId': 46},
{'name': 'dashcammount', 'id': 67, 'evaluate': True, 'trainId': 47},
{'name': 'water', 'id': 68, 'evaluate': False, 'trainId': 255},
{'name': 'sand', 'id': 69, 'evaluate': False, 'trainId': 255},
{'name': 'pedestrianarea', 'id': 70, 'evaluate': True, 'trainId': 48},
{'name': 'firehydrant', 'id': 71, 'evaluate': False, 'trainId': 255},
{'name': 'cctvcamera', 'id': 72, 'evaluate': False, 'trainId': 255},
{'name': 'snow', 'id': 73, 'evaluate': False, 'trainId': 255},
{'name': 'catchbasin', 'id': 74, 'evaluate': False, 'trainId': 255},
{'name': 'crosswalkplain', 'id': 75, 'evaluate': True, 'trainId': 49},
{'name': 'crosswalkzebra', 'id': 76, 'evaluate': True, 'trainId': 50},
{'name': 'manholesidewalk', 'id': 77, 'evaluate': False, 'trainId': 255},
{'name': 'curbterrain', 'id': 78, 'evaluate': False, 'trainId': 255},
{'name': 'servicelane', 'id': 79, 'evaluate': False, 'trainId': 255},
{'name': 'curbcut', 'id': 80, 'evaluate': False, 'trainId': 255}]

class wd2(BaseDataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(wd2, self).__init__(
                dataroot, annpath, trans_func, mode)
    
        
        self.n_cats = 26
        
        self.lb_ignore = -1
        # self.lb_ignore = 255
        self.lb_map = np.arange(256).astype(np.uint8)
        
        self.labels_info = labels_info
            
        for el in self.labels_info:
            if el['trainId'] > 24:
                self.lb_map[el['id']] = 25 #el['trainId']
            else:
                self.lb_map[el['id']] = el['trainId']
            # self.lb_map[el['id']] = el['trainId']

        self.to_tensor = ToTensor(
            mean=(0.3038, 0.3383, 0.3034), # city, rgb
            std=(0.2071, 0.2088, 0.2090),
        )

class TransformationVal(object):

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        return dict(im=im, lb=lb)

def get_data_loader(configer, aux_mode='eval', distributed=True, stage=None):
    mode = aux_mode
    n_datasets = configer.get('n_datasets')
    max_iter = configer.get('lr', 'max_iter')
    
    if mode == 'eval':
        trans_func = TransformationVal()
        batchsize = [configer.get('dataset'+str(i), 'eval_ims_per_gpu') for i in range(1, n_datasets+1)]
        annpath = [configer.get('dataset'+str(i), 'val_im_anns') for i in range(1, n_datasets+1)]
        imroot = [configer.get('dataset'+str(i), 'im_root') for i in range(1, n_datasets+1)]
        data_reader = [configer.get('dataset'+str(i), 'data_reader') for i in range(1, n_datasets+1)]
        
        shuffle = False
        drop_last = False
    elif mode == 'ret_path':
        trans_func = TransformationVal()
        batchsize = [1 for i in range(1, n_datasets+1)]
        if stage != None:
            annpath = [configer.get('dataset'+str(i), 'train_im_anns').replace('.txt', f'_{stage}.txt') for i in range(1, n_datasets+1)]
            print(annpath)
        else:
            annpath = [configer.get('dataset'+str(i), 'train_im_anns') for i in range(1, n_datasets+1)]
            
        imroot = [configer.get('dataset'+str(i), 'im_root') for i in range(1, n_datasets+1)]
        data_reader = [configer.get('dataset'+str(i), 'data_reader') for i in range(1, n_datasets+1)]
        
        shuffle = False
        drop_last = False
        
    
    ds = []
    for reader, root, path in zip(data_reader, imroot, annpath):

        ds.append(eval(reader)(root, path, trans_func=trans_func, mode=mode))
    # ds = [eval(reader)(root, path, trans_func=trans_func, mode=mode)
    #       for reader, root, path in zip(data_reader, imroot, annpath)]

    dl = []
    for idx in range(len(ds)):
        dataset = ds[idx]
        bs = batchsize[idx]
            

        dl.append(DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=4,
            pin_memory=True,
            # prefetch_factor=4
            ))
    return dl


class ToTensor(object):
    '''
    mean and std should be of the channel order 'bgr'
    '''
    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        im = im.transpose(2, 0, 1).astype(np.float32)
        im = torch.from_numpy(im).div_(255)
        dtype, device = im.dtype, im.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]
        im = im.sub_(mean).div_(std).clone()
        if not lb is None:
            lb = torch.from_numpy(lb.astype(np.int64))#.clone()
        return dict(im=im, lb=lb)

def get_img_for_everyclass(configer, dataset_id=None):
    n_datasets = configer.get("n_datasets")

    num_classes = []
    for i in range(1, n_datasets + 1):
        num_classes.append(configer.get("dataset" + str(i), "n_cats"))

    dls = get_data_loader(configer, aux_mode='ret_path', distributed=False)

    img_lists = []
    lb_lists  = []
    lb_info_list = []
    for i in range(0, n_datasets):
        this_img_lists = []
        this_lb_lists = []
        lb_info_list.append(dls[i].dataset.lb_map)
        
        if dataset_id != None and i != dataset_id:
            img_lists.append(this_img_lists)
            lb_lists.append(this_lb_lists)
            continue
        # print("cur dataset id： ", i)
        
        for label_id in range(0, num_classes[i]):
            this_img_lists.append([])
            this_lb_lists.append([])

        if dist.is_initialized() and dist.get_rank() != 0:
            diter = dls[i]
        else:
            diter = tqdm(dls[i])
        for im, lb, lbpth in diter:

            im = im[0]
            lb = lb.squeeze()
            for label_id in range(0, num_classes[i]):
                if len(this_img_lists[label_id]) < 100 and (lb == label_id).any():
                    this_img_lists[label_id].append(im)
                    this_lb_lists[label_id].append(lbpth)
                    

        for j, lb in enumerate(this_lb_lists):
            if len(lb) == 0:
                print("the number {} class has no image".format(j))
        
        
        img_lists.append(this_img_lists)
        lb_lists.append(this_lb_lists)
        
    
    return img_lists, lb_lists, lb_info_list

def sep_train_set_for_everyclass(configer, dataset_id=None):
    n_datasets = configer.get("n_datasets")

    num_classes = []
    for i in range(1, n_datasets + 1):
        num_classes.append(configer.get("dataset" + str(i), "n_cats"))

    dls = get_data_loader(configer, aux_mode='ret_path', distributed=False)

    # train_lists = []
    # test_lists  = []
    for i in range(0, n_datasets):
        this_train_lists = []
        this_test_lists = []
        this_class_len = []
        
        if dataset_id != None and i != dataset_id:
            # img_lists.append(this_img_lists)
            # lb_lists.append(this_lb_lists)
            continue
        # print("cur dataset id： ", i)
        
        for label_id in range(0, num_classes[i]):
            this_class_len.append(0)
            
            
        for im, lb, lbpth in dls[i]:
            im = im[0]
            lbpth = lbpth[0]
            lb = lb.squeeze()
            flag = False
            for label_id in range(0, num_classes[i]):
                if this_class_len[label_id] < 50 and (lb == label_id).any():
                    this_class_len[label_id] += 1 
                    if flag == False:
                        this_test_lists.append(im +', '+lbpth)
                        
                        
                    flag = True
            if flag == False:
                this_train_lists.append(im +', '+lbpth)
                    

        # for j, lb in enumerate(this_lb_lists):
        #     if len(lb) == 0:
        #         print("the number {} class has no image".format(j))
        
        
        strsplit = '\n'
        with open(f"{i}_train_1.txt","w") as f:
            f.write(strsplit.join(this_train_lists))
        
        with open(f"{i}_train_2.txt","w") as f:
            f.write(strsplit.join(this_test_lists))
        
    
    


def get_img_for_everyclass_single(configer, dls):
    n_datasets = configer.get("n_datasets")

    num_classes = []
    for i in range(1, n_datasets + 1):
        num_classes.append(configer.get("dataset" + str(i), "n_cats"))

    # dls = get_data_loader(configer, aux_mode='ret_path', distributed=False)
    dl_iters = [iter(dl) for dl in dls]

    img_lists = []
    lb_lists  = []
    for i in range(0, n_datasets):
        # print("cur dataset id： ", i)
        this_img_lists = []
        this_lb_lists = []
        for label_id in range(0, num_classes[i]):
            this_img_lists.append([])
            this_lb_lists.append([])
            
        cur_num = 0
        while True:
            try:
                im, lb, lbpth = next(dl_iters[i])
                while torch.min(lb) == 255:
                    im, lb, lbpth = next(dl_iters[i])

                if not len(im) == 1:
                    raise StopIteration
            except StopIteration:
                break
            

            im = im[0]
            lb = lb.squeeze()
            for label_id in range(0, num_classes[i]):
                if (lb == label_id).any():
                    this_img_lists[label_id].append(im)
                    this_lb_lists[label_id].append(lbpth)
                    
            # print('cur_num: ', cur_num)
            # print('num_classes: ', num_classes[i])
            # print(lbpth)
            # if cur_num == num_classes[i]:
            #     break

        for j, lb in enumerate(this_lb_lists):
            if len(lb) == 0:
                print("the number {} class has no image".format(j))
        
            
        img_lists.append(this_img_lists)
        lb_lists.append(this_lb_lists)

    return img_lists, lb_lists

def crop_image_by_label_value(img, label, label_value):
    # 将标签二值化
    binary = np.zeros_like(label, dtype=np.uint8)
    binary[label == label_value] = 255

    binary = cv2.convertScaleAbs(binary)
    
    # 执行闭运算操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 计算轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
#     # 找到覆盖所有轮廓的最小矩形
#     max_rect = cv2.minAreaRect(np.concatenate(contours))
    
#     # 获取最大包围盒的坐标
#     x, y, w, h = cv2.boundingRect(np.int0(cv2.boxPoints(max_rect)))
#     print(x, y, w, h)

    # 计算每个包围盒的面积并找到面积最大的包围盒
    max_area = 0
    max_bbox = None
    for contour in contours:
        bbox = cv2.boundingRect(contour)
        area = bbox[2] * bbox[3]
        if area > max_area:
            max_area = area
            max_bbox = bbox

    # 如果没有找到任何包围盒，返回空图像
    if max_bbox is None:
        return np.zeros_like(img)

    # 裁剪图像
    x, y, w, h = max_bbox

#     # 获取包围盒
#     x, y, w, h = cv2.boundingRect(contours[4])
#     print(x, y, w, h)
    
    # 裁剪图像
    # print(img.shape)
    cropped = img[y:y+h, x:x+w, :]
    
    # 将不属于该标签的像素点替换为指定的值
    label_roi = binary[y:y+h, x:x+w]
    
    k = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(label_roi, kernel=k)
    
    
    mask = np.ones_like(cropped, dtype=bool)
    mask[dilated != 255] = False
    cropped[~mask] = 128
    
    h, w, _ = cropped.shape
    if h < w:
        top_padding = (w - h) // 2
        bottom_padding = w - h - top_padding
        cropped = cv2.copyMakeBorder(cropped, top_padding, bottom_padding, 0, 0, cv2.BORDER_CONSTANT, value=[128, 128, 128])
    elif h > w:
        left_padding = (h - w) // 2
        right_padding = h - w - left_padding
        cropped = cv2.copyMakeBorder(cropped, 0, 0, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[128, 128, 128])
        
    # 返回裁剪后的图像
    return cropped


def gen_image_features(configer, dataset_id=None):
    img_lists, lb_lists, lb_info_list = get_img_for_everyclass(configer, dataset_id)
    
    n_datasets = configer.get('n_datasets')
    to_tensor = ToTensor(
                mean=(0.48145466, 0.4578275, 0.40821073), # clip , rgb
                std=(0.26862954, 0.26130258, 0.27577711),
            )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    with torch.no_grad():
        out_features = []
        for idx in range(0, n_datasets):
            this_label_info = lb_info_list[idx]
            if dataset_id != None and idx != dataset_id:
                continue
            print("dataset_id: ", idx)
            for i, im_lb_list in enumerate(zip(img_lists[idx], lb_lists[idx])):
                im_list, lb_list = im_lb_list
                if len(im_list) == 0:
                    print("why dataset_id: ", idx)
                    continue
                image_features_list = []
                for im_path, lb_path in zip(im_list, lb_list):
                    image = cv2.imread(im_path)
                    # print(lb_path[0])
                    lb = cv2.imread(lb_path[0], 0)
                    lb = this_label_info[lb]
                    if image is None:
                        print(im_path)
                        continue
                    # lb = lb.numpy()
                    cropped_img = crop_image_by_label_value(image, lb, i)
                        
                    im_lb = dict(im=cropped_img, lb=lb)
                    im_lb = to_tensor(im_lb)
                    img = im_lb['im'].cuda()
                    img = F.interpolate(img.unsqueeze(0), size=(224, 224))
                    image_features = model.encode_image(img).type(torch.float32)
                    image_features_list.append(image_features)

                # print("im_lb_list: ", im_lb_list)
                img_feat = torch.cat(image_features_list, dim=0)
                mean_feats = torch.mean(img_feat, dim=0, keepdim=True)
                # print(mean_feats.shape)
                out_features.append(mean_feats) 
    
    return out_features

def gen_image_features_single(configer, dls, gen_feature=False):
    if gen_feature is False:
        img_lists, lb_lists = get_img_for_everyclass_single(configer, dls)
    else:
        img_lists, lb_lists = dls
        
    n_datasets = configer.get('n_datasets')
    to_tensor = ToTensor(
                mean=(0.48145466, 0.4578275, 0.40821073), # clip , rgb
                std=(0.26862954, 0.26130258, 0.27577711),
            )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    with torch.no_grad():
        out_features = []
        for dataset_id in range(0, n_datasets):
            print("dataset_id: ", dataset_id)
            for i, im_lb_list in enumerate(zip(img_lists[dataset_id], lb_lists[dataset_id])):
                im_list, lb_list = im_lb_list
                im_lb_zip_list = list(zip(im_list, lb_list))
                if len(im_list) == 0:
                    print("why dataset_id: ", dataset_id)
                    continue
                image_features_list = []
                im_path, lb_path = random.choice(im_lb_zip_list)
                image = cv2.imread(im_path)
                # print(lb_path[0])
                lb = cv2.imread(lb_path[0], 0)
                if image is None:
                    print(im_path)
                    continue
                # lb = lb.numpy()
                cropped_img = crop_image_by_label_value(image, lb, i)
                    
                im_lb = dict(im=cropped_img, lb=lb)
                im_lb = to_tensor(im_lb)
                img = im_lb['im'].cuda()
                img = F.interpolate(img.unsqueeze(0), size=(224, 224))
                image_features = model.encode_image(img).type(torch.float32)
                image_features_list.append(image_features)

                # print("im_lb_list: ", im_lb_list)
                img_feat = torch.cat(image_features_list, dim=0)
                mean_feats = torch.mean(img_feat, dim=0, keepdim=True)
                # print(mean_feats.shape)
                out_features.append(mean_feats) 

    
    return out_features, [img_lists, lb_lists]

def gen_image_features_storage(configer, dataset_id):
    img_lists, lb_lists = get_img_for_everyclass(configer, dataset_id)
    
    n_datasets = configer.get('n_datasets')
    to_tensor = ToTensor(
                mean=(0.48145466, 0.4578275, 0.40821073), # clip , rgb
                std=(0.26862954, 0.26130258, 0.27577711),
            )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    with torch.no_grad():
        this_datasets_feats = []
        print("dataset_id: ", dataset_id)
        for i, im_lb_list in enumerate(zip(img_lists[dataset_id], lb_lists[dataset_id])):
            im_list, lb_list = im_lb_list
            if len(im_list) == 0:
                print("why dataset_id: ", dataset_id)
                continue
            image_features_list = []
            for im_path, lb_path in zip(im_list, lb_list):
                image = cv2.imread(im_path)
                # print(lb_path[0])
                lb = cv2.imread(lb_path[0], 0)
                if image is None:
                    print(im_path)
                    continue
                # lb = lb.numpy()
                cropped_img = crop_image_by_label_value(image, lb, i)
                    
                im_lb = dict(im=cropped_img, lb=lb)
                im_lb = to_tensor(im_lb)
                img = im_lb['im'].cuda()
                img = F.interpolate(img.unsqueeze(0), size=(224, 224))
                image_features = model.encode_image(img).type(torch.float32)
                image_features_list.append(image_features)

            # print("im_lb_list: ", im_lb_list)
            img_feat = torch.cat(image_features_list, dim=0)
            # mean_feats = torch.mean(img_feat, dim=0, keepdim=True)
            # print(mean_feats.shape)
            this_datasets_feats.append(img_feat) 
    # out_features = torch.cat(out_features, dim=0)    
    return this_datasets_feats


def get_encode_lb_vec(configer, datasets_id=None):
    ori_model_path = osp.join(os.getcwd(), 'llama-2-7b-hf')
    # param_json = 'llama-2-7b/params.json'
    config_kwargs = {
        "trust_remote_code": True,
        # "cache_dir": './llama-2-7b-hf',
        "revision": 'main',
        # "use_auth_token": None,
        "output_hidden_states": True
    }
    tokenizer = AutoTokenizer.from_pretrained(ori_model_path)
    config = AutoConfig.from_pretrained(ori_model_path, **config_kwargs)

    llama_model = AutoModelForCausalLM.from_pretrained(
        ori_model_path,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        revision='main',
    ).cuda()
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        llama_model.resize_token_embeddings(len(tokenizer))
    n_datasets = configer.get('n_datasets')
    text_feature_vecs_all = []
    with torch.no_grad():
        # clip_model, _ = clip.load("ViT-B/32", device="cuda")
        for i in range(0, n_datasets):
            text_feature_vecs = []
            if datasets_id != None and i != datasets_id:
                continue
            lb_name = configer.get("dataset"+str(i+1), "label_names")
            for lb in lb_name:
                # from_sta = lb.find('from')
                # from_end = lb.find('.', from_sta)
                # lb = lb[:from_sta-1] + lb[from_end:]
                tokens = tokenizer.encode_plus(lb, add_special_tokens=True, padding='max_length', truncation=True,
                                        max_length=128, return_tensors='pt')
                input_ids = tokens["input_ids"]
                attention_mask = tokens['attention_mask']
                outputs = llama_model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda())
                # embedding = list(outputs.hidden_states)
                
                hidden_states = outputs.hidden_states
                last_hidden_states = hidden_states[-1].squeeze()
                # fisrt_larst_avg_status = last_hidden_states.mean(dim=0)
                first_hidden_states = hidden_states[0].squeeze()
                # print(hidden_states.shape)
                # 计算平均状态
                fisrt_larst_avg_status = (first_hidden_states + last_hidden_states).mean(dim=0)
                text_feature_vecs.append(fisrt_larst_avg_status[None])
            text_feature_vecs = torch.cat(text_feature_vecs, dim=0)
            text_feature_vecs_all.append(text_feature_vecs)
            # text_feature_vecs.append(text_features)
            
    return text_feature_vecs_all
                
def get_encode_lb_vec_clip(configer, datasets_id=None):
    n_datasets = configer.get('n_datasets')
    text_feature_vecs = []
    with torch.no_grad():
        clip_model, _ = clip.load("ViT-B/32", device="cuda")
        for i in range(0, n_datasets):
            if datasets_id != None and i != datasets_id:
                continue
            lb_name = configer.get("dataset"+str(i+1), "label_names")
            lb_name = ["a photo of " + name + "." for name in lb_name]
            text = clip.tokenize(lb_name).cuda()
            text_features = clip_model.encode_text(text).type(torch.float32)
            text_feature_vecs.append(text_features)
            
    return text_feature_vecs
    
def gen_graph_node_feature(cfg):
    configer = Configer(configs=cfg.DATASETS.CONFIGER)
    n_datasets = configer.get("n_datasets")
    save_pth = 'output/'
    if not osp.exists(save_pth): os.makedirs(save_pth)
    
    file_name = save_pth + 'graph_node_features_llama'
    dataset_names = []
    for i in range(0, configer.get('n_datasets')):
        # file_name += '_'+str(configer.get('dataset'+str(i+1), 'data_reader'))
        dataset_names.append(str(configer.get('dataset'+str(i+1), 'data_reader')))
    
    # file_name += '.pt'
    out_features = []
    for i in range(0, n_datasets):
        this_file_name = file_name + f'_{dataset_names[i]}.pt' 
        if osp.exists(this_file_name):
            this_graph_node_features = torch.load(this_file_name, map_location='cpu')

            out_features.append(this_graph_node_features)
        else:
            print(f'gen_graph_node_featuer: {i}')
            # img_feature_vecs = gen_image_features(configer, i)
            # img_feat_tensor = torch.cat(img_feature_vecs, dim=0)
            
            text_feature_vecs = get_encode_lb_vec(configer, i)[0]
            # this_graph_node_features = torch.cat([text_feature_vecs, img_feat_tensor], dim=1)
            
            print("gen finished")
            torch.save(text_feature_vecs.clone(), this_file_name)

            out_features.append(text_feature_vecs.cpu())
    
    out_features = torch.cat(out_features, dim=0)
    print(out_features.shape)
    return out_features 
    
                
def gen_graph_node_feature_clip(cfg):
    configer = Configer(configs=cfg.DATASETS.CONFIGER)
    # configer = cfg
    n_datasets = configer.get("n_datasets")
    save_pth = 'output/'
    if not osp.exists(save_pth): os.makedirs(save_pth)
    
    file_name = save_pth + 'graph_node_features'
    dataset_names = []
    for i in range(0, configer.get('n_datasets')):
        # file_name += '_'+str(configer.get('dataset'+str(i+1), 'data_reader'))
        dataset_names.append(str(configer.get('dataset'+str(i+1), 'data_reader')))
    
    # file_name += '.pt'
    out_features = []
    for i in range(0, n_datasets):
        this_file_name = file_name + f'_{dataset_names[i]}.pt' 
        if osp.exists(this_file_name):
            this_graph_node_features = torch.load(this_file_name, map_location='cpu')

            out_features.append(this_graph_node_features)
        else:
            # raise Exception("Not Imp!")
            print(f'gen_graph_node_featuer: {i}')
            img_feature_vecs = gen_image_features(configer, i)
            img_feat_tensor = torch.cat(img_feature_vecs, dim=0)
            
            text_feature_vecs = get_encode_lb_vec_clip(configer, i)[0]
            this_graph_node_features = torch.cat([text_feature_vecs, img_feat_tensor], dim=1)
            
            print("gen finished")
            torch.save(this_graph_node_features.clone(), this_file_name)

            out_features.append(this_graph_node_features.cpu())
    
    out_features = torch.cat(out_features, dim=0)
    print(out_features.shape)
    return out_features 
    
    # if not osp.exists(save_pth): os.makedirs(save_pth)
    
    # file_name = save_pth + 'graph_node_features'+str(configer.get('n_datasets'))
    # for i in range(0, configer.get('n_datasets')):
    #     file_name += '_'+str(configer.get('dataset'+str(i+1), 'data_reader'))
    
    # file_name += '.pt'

    # if osp.exists(file_name):
    #     graph_node_features = torch.load(file_name)
    # else:
    #     print("gen_graph_node_feature")
    #     text_feature_vecs = get_encode_lb_vec(configer)
    #     text_feat_tensor = torch.cat(text_feature_vecs, dim=0)
    #     print(text_feat_tensor.shape)
    #     print("gen_text_feature_vecs")
    #     img_feature_vecs = gen_image_features(configer)
    #     img_feat_tensor = torch.cat(img_feature_vecs, dim=0)
    #     print(img_feat_tensor.shape)
    #     print("gen_img_features")
    #     graph_node_features = torch.cat([text_feat_tensor, img_feat_tensor], dim=1)
    #     # graph_node_features = (text_feat_tensor+img_feat_tensor)/2
    #     print(graph_node_features.shape)
    #     torch.save(graph_node_features.clone(), file_name)
    
    # return graph_node_features



if __name__ == "__main__":
    configer = Configer(configs="configs/ltbgnn_5_datasets.json")
    # img_feature_vecs = gen_graph_node_feature_storage(configer) 
    gen_graph_node_feature(configer)
    print('finished')
    # print(img_feature_vecs[0][0])
    # print(img_feature_vecs)

    # print(graph_node_features.shape)
    # norm_adj_feat = F.normalize(graph_node_features, p=2, dim=1)
    # similar_matrix = torch.einsum('nc, mc -> nm', norm_adj_feat, norm_adj_feat)
    # print("similar_matrix_max:", torch.max(similar_matrix))
    # print("similar_matrix_min:", torch.min(similar_matrix))
    # torch.set_printoptions(profile="full")
    # print(similar_matrix)
    
    