import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import os.path as osp

labels_info = [
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

dataroot = 'datasets'
annpath = f'auto_uni_seg/datasets/sunrgbd/test.txt'
def sunrgbd():
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


def register_sunrgbd():
    
    
    # meta = _get_ade20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in labels_info:
        lb_map[el['id']] = el['trainId']

    name = f"sunrgbd_sem_seg_val"
    DatasetCatalog.register(
        name, sunrgbd
    )
    
    MetadataCatalog.get(name).set(
        stuff_classes=["bag", "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter", "blinds", "desk", "shelves", "curtain", "dresser", "pillow", "mirror", "floor mat", "clothes", "ceiling", "books", "refridgerator", "television", "paper", "towel", "shower curtain", "box", "whiteboard", "person", "night stand", "toilet", "sink", "lamp", "bathtub"],
        stuff_dataset_id_to_contiguous_id=lb_map,
        thing_dataset_id_to_contiguous_id=lb_map,
        stuff_colors=[[164, 121, 219],[174, 203, 111],     [ 23,  69, 195],     [233, 198,  64],     [ 10,   7, 111],     [ 38, 181, 228],     [ 75, 211,  41],     [  7, 162,   5],     [ 77, 110,  58],     [ 35, 133,  44],     [163, 186, 153],     [141, 251,  82],     [138,  37,  98],     [177,  96,  58],     [ 83,  69,  70],     [126,  22, 239],     [113,  39, 205],     [224, 216, 151],     [ 54,   3, 109],     [146, 240, 187],     [132, 182, 126],     [112,  78, 139],     [ 19, 228,  89],     [240,  79,  15],     [177, 158, 120],     [ 61, 206, 227],     [230, 101, 208],     [235,  91,  37],     [109,  64,  73],     [ 58,  34,   9],     [178, 109,  61],     [198, 221, 182],     [160, 111,   0],     [109, 155, 209],     [236, 223, 130],     [168,  62, 253],     [ 12,  71,  15]],
        evaluator_type="sem_seg",
        ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
    )



# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_sunrgbd()


train_annpath = f'auto_uni_seg/datasets/sunrgbd/train.txt'
def sunrgbd_train(anp):

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


def register_sunrgbd_train():
    
    
    # meta = _get_cs20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in labels_info:
        lb_map[el['id']] = el['trainId']
    for n, anp in [("train", "train"), ("train_1", "train_1"), ("train_2", "train_2")]:
        name = f"sunrgbd_sem_seg_{n}"
        annpath = f'auto_uni_seg/datasets/sunrgbd/{anp}.txt'
        DatasetCatalog.register(
            name, lambda x=annpath : sunrgbd_train(x)
        )
            
        MetadataCatalog.get(name).set(
            stuff_classes=["bag", "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter", "blinds", "desk", "shelves", "curtain", "dresser", "pillow", "mirror", "floor mat", "clothes", "ceiling", "books", "refridgerator", "television", "paper", "towel", "shower curtain", "box", "whiteboard", "person", "night stand", "toilet", "sink", "lamp", "bathtub"],
            stuff_dataset_id_to_contiguous_id=lb_map,
            thing_dataset_id_to_contiguous_id=lb_map,
            stuff_colors=[[164, 121, 219],[174, 203, 111],     [ 23,  69, 195],     [233, 198,  64],     [ 10,   7, 111],     [ 38, 181, 228],     [ 75, 211,  41],     [  7, 162,   5],     [ 77, 110,  58],     [ 35, 133,  44],     [163, 186, 153],     [141, 251,  82],     [138,  37,  98],     [177,  96,  58],     [ 83,  69,  70],     [126,  22, 239],     [113,  39, 205],     [224, 216, 151],     [ 54,   3, 109],     [146, 240, 187],     [132, 182, 126],     [112,  78, 139],     [ 19, 228,  89],     [240,  79,  15],     [177, 158, 120],     [ 61, 206, 227],     [230, 101, 208],     [235,  91,  37],     [109,  64,  73],     [ 58,  34,   9],     [178, 109,  61],     [198, 221, 182],     [160, 111,   0],     [109, 155, 209],     [236, 223, 130],     [168,  62, 253],     [ 12,  71,  15]],
            evaluator_type="sem_seg",
            ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
        )

# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_sunrgbd_train()