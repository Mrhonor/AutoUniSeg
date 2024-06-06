import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

__all__ = ["SemanticDatasetMapper"]


class SemanticDatasetMapper:

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        should_lookup_table,
        lookup_table,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.should_lookup_table = should_lookup_table
        logger = logging.getLogger(__name__)
        if self.should_lookup_table:
            self.lb_map = np.arange(256).astype(np.uint8)
            for k, v in lookup_table.items():
                self.lb_map[k] = v
            # self.lb_map = torch.tensor(self.lb_map)

        
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True, dataset_id=0, should_lookup_table=False):
        # Build augmentation
        logger = logging.getLogger(__name__)
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        if is_train:
            dataset_names = cfg.DATASETS.TRAIN
        else:
            dataset_names = cfg.DATASETS.TEST
        meta = MetadataCatalog.get(dataset_names[dataset_id])
    
        ignore_label = meta.ignore_label
        thing_dataset_id_to_contiguous_id = None
        if should_lookup_table:
            thing_dataset_id_to_contiguous_id = meta.thing_dataset_id_to_contiguous_id

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "should_lookup_table": should_lookup_table,
            "lookup_table": thing_dataset_id_to_contiguous_id
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        logger = logging.getLogger(__name__)
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # logger.info(f"dataset_dict: {dataset_dict}")
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None and self.is_train:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        if self.is_train:
            aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
            aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
            image = aug_input.image
            sem_seg_gt = aug_input.sem_seg

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.is_train and self.size_divisibility > 0:
            image_size = [image.shape[-2], image.shape[-1]] 
            # image_size[0] = self.size_divisibility if image_size[0] == 0 else image_size[0]
            # image_size[1] = self.size_divisibility if image_size[1] == 0 else image_size[1]
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        if sem_seg_gt is not None:
            
            if self.should_lookup_table:
                sem_seg_gt = sem_seg_gt.long()
                sem_seg_gt = self.lb_map[sem_seg_gt]
                
                dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt).long()
            else:
                dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # # Prepare per-category binary masks
        # if sem_seg_gt is not None:
        #     sem_seg_gt = sem_seg_gt.numpy()
        #     instances = Instances(image_shape)
        #     classes = np.unique(sem_seg_gt)
        #     # remove ignored region
        #     classes = classes[classes != self.ignore_label]
        #     instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

        #     masks = []
        #     for class_id in classes:
        #         masks.append(sem_seg_gt == class_id)

        #     if len(masks) == 0:
        #         # Some image does not have annotation (all ignored)
        #         instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
        #     else:
        #         masks = BitMasks(
        #             torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
        #         )
        #         instances.gt_masks = masks.tensor

        #     dataset_dict["instances"] = instances

        # logger.info(logger.info(f"output dataset_dict: {dataset_dict}"))
        return dataset_dict