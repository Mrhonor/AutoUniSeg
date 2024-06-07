# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN



def add_hrnet_config(cfg):
    # NOTE: configs from original hrnet
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "SemanticDatasetMapper"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    cfg.MODEL.HRNET = CN()

    cfg.MODEL.HRNET.HRNET_CFG = "hrnet48"
    cfg.MODEL.HRNET.KEEP_IMAGENET_HEAD = False  # not used
    cfg.MODEL.HRNET.DROP_STAGE4 = False
    cfg.MODEL.HRNET.FULL_RES_STEM = False
    cfg.MODEL.HRNET.BN_TYPE = "torchbn"  # use syncbn for cityscapes dataset
    cfg.MODEL.AUX_MODE = "train"
    cfg.MODEL.WITH_DATASETS_AUX = False
    cfg.MODEL.PRETRAINING = False
    cfg.MODEL.SIZE_DIVISIBILITY = -1
    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    cfg.DATASETS.DATASETS_CATS = [19]
    cfg.DATASETS.EVAL = []
    cfg.DATASETS.IGNORE_LB = 255
    cfg.DATASETS.NUM_UNIFY_CLASS = 19
    cfg.DATASETS.CONFIGER = 'configs/ltbgnn_7_datasets_snp.json'
    cfg.DATASETS.SPECIFIC_DATASET_ID = -1

    
    cfg.LOSS = CN()
    cfg.LOSS.OHEM_THRESH = 0.7 
    
    

    cfg.MODEL.SEM_SEG_HEAD.OUTPUT_FEAT_DIM = 19
    cfg.MODEL.SEM_SEG_HEAD.WITH_DATASETS_AUX = False
    cfg.MODEL.SEM_SEG_HEAD.BN_TYPE = "torchbn"

def add_gnn_config(cfg):
    cfg.MODEL.GNN = CN()
    
    cfg.MODEL.GNN.GNN_MODEL_NAME = "Learnable_Topology_BGNN"
    cfg.MODEL.GNN.NFEAT = 1024
    cfg.MODEL.GNN.NFEAT_OUT = 512
    cfg.MODEL.GNN.nfeat_adj = 256
    cfg.MODEL.GNN.adj_feat_dim = 128
    cfg.MODEL.GNN.dropout_rate = 0.5
    cfg.MODEL.GNN.threshold_value = 0.95
    cfg.MODEL.GNN.calc_bipartite = False
    cfg.MODEL.GNN.output_max_adj = True
    cfg.MODEL.GNN.output_softmax_adj = True
    cfg.MODEL.GNN.uot_ratio = 1.01
    cfg.MODEL.GNN.mse_or_adv = "None"
    cfg.MODEL.GNN.GNN_type = "GSAGE"
    cfg.MODEL.GNN.with_datasets_aux = True
    cfg.MODEL.GNN.init_stage_iters = 10000
    cfg.MODEL.GNN.isGumbelSoftmax = False
    cfg.MODEL.GNN.FINETUNE_STAGE1_ITERS = 20000
    cfg.MODEL.GNN.GNN_ITERS = 20000
    cfg.MODEL.GNN.SEG_ITERS = 20000
    cfg.MODEL.GNN.FIRST_STAGE_GNN_ITERS = 15000
    cfg.MODEL.GNN.FINETUNE_STAGE1_ITERS = 20000
    cfg.MODEL.GNN.INIT_ADJ_PATH = 'output/init_adj_7_datasets.pt'
    cfg.MODEL.GNN.N_POINTS = 12455
    cfg.MODEL.GNN.dataset_embedding = False
    
    
    cfg.LOSS.WITH_SPA_LOSS = True
    cfg.LOSS.WITH_ORTH_LOSS = False
    cfg.LOSS.WITH_ADJ_LOSS = False

    
