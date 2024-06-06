import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import os.path as osp

labels_info = [
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

dataroot = 'datasets/bdd100k'
annpath = f'auto_uni_seg/datasets/bdd100k/val.txt'
def bdd():
    # assert mode in ('train', 'eval', 'test')

    with open(annpath, 'r') as fr:
        pairs = fr.read().splitlines()
    img_paths, lb_paths = [], []
    for pair in pairs:
        imgpth, lbpth = pair.split(',')
        img_paths.append(osp.join(dataroot, imgpth))
        lb_paths.append(osp.join(dataroot, lbpth))

    assert len(img_paths) == len(lb_paths)
    dataset_dicts = []
    for (img_path, gt_path) in zip(img_paths, lb_paths):
        record = {}
        record["file_name"] = img_path
        record["sem_seg_file_name"] = gt_path
        dataset_dicts.append(record)

    return dataset_dicts


def register_bdd():
    
    
    # meta = _get_ade20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in labels_info:
        lb_map[el['id']] = el['trainId']

    name = f"bdd_sem_seg_val"
    DatasetCatalog.register(
        name, bdd
    )

    MetadataCatalog.get(name).set(
        stuff_classes=["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"],
        stuff_dataset_id_to_contiguous_id=lb_map,
        thing_dataset_id_to_contiguous_id=lb_map,
        stuff_colors=[[ 97, 187, 159],[236,   2, 131],     [154, 172, 132],     [174, 203, 111],     [145, 208,  42],     [ 70,  94, 120],     [153,  29, 134],     [218, 118, 115],     [128, 109,  69],     [155, 117, 111],     [225, 115,  77],     [198, 221, 182],     [ 24, 192, 253],     [ 14, 159,  57],     [ 81, 255, 160],     [ 79, 127, 244],     [252,   9, 152],     [ 12,  50, 106],     [103,  81, 234]],
        evaluator_type="sem_seg",
        ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
    )


# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_bdd()



train_annpath = f'auto_uni_seg/datasets/bdd100k/train.txt'

def bdd_train(anp):

    with open(anp, 'r') as fr:
        pairs = fr.read().splitlines()
    img_paths, lb_paths = [], []
    for pair in pairs:
        imgpth, lbpth = pair.split(',')
        img_paths.append(osp.join(dataroot, imgpth))
        lb_paths.append(osp.join(dataroot, lbpth))

    assert len(img_paths) == len(lb_paths)
    dataset_dicts = []
    for (img_path, gt_path) in zip(img_paths, lb_paths):
        record = {}
        record["file_name"] = img_path
        record["sem_seg_file_name"] = gt_path
        dataset_dicts.append(record)

    return dataset_dicts


def register_bdd_train():
    
    
    # meta = _get_bdd20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in labels_info:
        lb_map[el['id']] = el['trainId']
    for n, anp in [("train", "train"), ("train_1", "train_1"), ("train_2", "train_2")]:
        name = f"bdd_sem_seg_{n}"
        annpath = f'auto_uni_seg/datasets/bdd100k/{anp}.txt'
        DatasetCatalog.register(
            name, lambda x=annpath : bdd_train(x)
        )
        
        MetadataCatalog.get(name).set(
            stuff_classes=["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"],
            stuff_dataset_id_to_contiguous_id=lb_map,
            thing_dataset_id_to_contiguous_id=lb_map,
            stuff_colors=[[ 97, 187, 159],[236,   2, 131],     [154, 172, 132],     [174, 203, 111],     [145, 208,  42],     [ 70,  94, 120],     [153,  29, 134],     [218, 118, 115],     [128, 109,  69],     [155, 117, 111],     [225, 115,  77],     [198, 221, 182],     [ 24, 192, 253],     [ 14, 159,  57],     [ 81, 255, 160],     [ 79, 127, 244],     [252,   9, 152],     [ 12,  50, 106],     [103,  81, 234]],
            evaluator_type="sem_seg",
            ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
        )

register_bdd_train()