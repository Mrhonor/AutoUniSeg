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

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import threading
from nvidia.dali.plugin.pytorch import DALIClassificationIterator as PyTorchIterator

from nvidia.dali.plugin.pytorch import LastBatchPolicy
from detectron2.structures import BitMasks, Instances
import logging
from ..dataset_mappers.semantic_dataset_mapper import SemanticDatasetMapper
from ..dataset_mappers.mds_panoptic_dataset_mapper import MdsMaskFormerPanopticDatasetMapper
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader

class PanoLoaderAdapter:
    def __init__(self, cfg, aux_mode='train', dataset_id=None, datasets_name=None) -> None:
        Log = logging.getLogger(__name__)
        Log.info(f"LoaderAdapter:{aux_mode}, dataset_id:{dataset_id}")
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
        
        mapper = []
        for i in range(len(dataset)):
            should_lkt = True

            mapper.append(MdsMaskFormerPanopticDatasetMapper(cfg, True, i, should_lkt))

        if aux_mode == 'train':
            self.dls = [build_detection_train_loader(cfg, mapper=mp, dataset=ds) for ds, mp in zip(dataset, mapper)]
        elif aux_mode == 'eval':
            mapper = MdsMaskFormerPanopticDatasetMapper(cfg, False, dataset_id, True)
            Log.info(f"evaluate {self.datasets_name[dataset_id]}")
            self.dls = build_detection_test_loader(cfg, dataset_name=self.datasets_name[dataset_id], mapper=mapper)            
        else: # aux_mode == 'eval':
            
            # if dataset_id > 0:
            #     mapper = SemanticDatasetMapper(cfg, False, dataset_id-1, True)
            #     Log.info(f"evaluate {self.datasets_name[dataset_id-1]}")
            #     self.dls = build_detection_test_loader(cfg, dataset_name=self.datasets_name[dataset_id-1], mapper=mapper)
            # else:
            if dataset_id >= len(self.datasets_name):
                mapper = MdsMaskFormerPanopticDatasetMapper(cfg, False, dataset_id, True)
                Log.info(f"evaluate {self.datasets_name[0]}")
                self.dls = build_detection_test_loader(cfg, dataset_name=self.datasets_name[0], mapper=mapper)            
            else:
                mapper = MdsMaskFormerPanopticDatasetMapper(cfg, False, dataset_id, True)
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