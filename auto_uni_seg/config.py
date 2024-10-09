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
    cfg.LOSS.WITH_RELATION_LOSS = False
    cfg.LOSS.WITH_GAUSSIAN_LOSS = False

    cfg.DATASETS.RELATION_GRAPH = None

    

def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
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

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.OUT_FEAT_DIM = 512
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

def add_afformer_config(cfg):
    cfg.MODEL.AFFORMER_CONFIG = None