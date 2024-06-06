import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import os.path as osp

labels_info = [
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

dataroot = 'datasets/idd'
annpath = f'auto_uni_seg/datasets/IDD/val.txt'
def idd():
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


def register_idd():
    
    
    # meta = _get_ade20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in labels_info:
        lb_map[el['id']] = el['trainId']

    name = f"idd_sem_seg_val"
    DatasetCatalog.register(
        name, idd
    )
    
    MetadataCatalog.get(name).set(
        stuff_classes=["road", "drivable fallback or parking", "sidewalk", "non-drivable fallback or rail track", "person or animal", "out of roi or rider", "motorcycle", "bicycle", "autorickshaw", "car", "truck", "bus", "trailer or caravan or vehicle fallback", "curb", "wall", "fence", "guard rail", "billboard", "traffic sign", "traffic light", "polegroup or pole", "obs-str-bar-fallback", "building", "tunnel or bridge", "vegetation", "sky or fallback background"],
        stuff_dataset_id_to_contiguous_id=lb_map,
        thing_dataset_id_to_contiguous_id=lb_map,
        stuff_colors=[[ 97, 187, 159],[109, 220, 239],     [236,   2, 131],     [  9,  60, 225],     [198, 221, 182],     [ 24, 192, 253],     [ 12,  50, 106],     [103,  81, 234],     [ 77, 252, 183],     [ 14, 159,  57],     [ 81, 255, 160],     [ 79, 127, 244],     [224, 216, 151],     [ 66,  69, 176],     [174, 203, 111],     [145, 208,  42],     [180, 241, 180],     [215,  48,  75],     [218, 118, 115],     [153,  29, 134],     [ 70,  94, 120],     [160, 111,   0],     [154, 172, 132],     [ 18,  58,  98],     [128, 109,  69],     [225, 115,  77]],
        evaluator_type="sem_seg",
        ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
    )


# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_idd()

train_annpath = f'auto_uni_seg/datasets/IDD/train.txt'
def IDD_train(anp):

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


def register_IDD_train():
    
    
    # meta = _get_cs20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in labels_info:
        lb_map[el['id']] = el['trainId']
    for n, anp in [("train", "train"), ("train_1", "train_1"), ("train_2", "train_2")]:
        name = f"idd_sem_seg_{n}"
        annpath = f'auto_uni_seg/datasets/IDD/{anp}.txt'
        DatasetCatalog.register(
            name, lambda x=annpath : IDD_train(x)
        )
        
        MetadataCatalog.get(name).set(
            stuff_classes=["road", "drivable fallback or parking", "sidewalk", "non-drivable fallback or rail track", "person or animal", "out of roi or rider", "motorcycle", "bicycle", "autorickshaw", "car", "truck", "bus", "trailer or caravan or vehicle fallback", "curb", "wall", "fence", "guard rail", "billboard", "traffic sign", "traffic light", "polegroup or pole", "obs-str-bar-fallback", "building", "tunnel or bridge", "vegetation", "sky or fallback background"],
            stuff_dataset_id_to_contiguous_id=lb_map,
            thing_dataset_id_to_contiguous_id=lb_map,
            stuff_colors=[[ 97, 187, 159],[109, 220, 239],     [236,   2, 131],     [  9,  60, 225],     [198, 221, 182],     [ 24, 192, 253],     [ 12,  50, 106],     [103,  81, 234],     [ 77, 252, 183],     [ 14, 159,  57],     [ 81, 255, 160],     [ 79, 127, 244],     [224, 216, 151],     [ 66,  69, 176],     [174, 203, 111],     [145, 208,  42],     [180, 241, 180],     [215,  48,  75],     [218, 118, 115],     [153,  29, 134],     [ 70,  94, 120],     [160, 111,   0],     [154, 172, 132],     [ 18,  58,  98],     [128, 109,  69],     [225, 115,  77]],
            evaluator_type="sem_seg",
            ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
        )

register_IDD_train()


# mseg_labels_info = [
#     {"name": "person", "id": 0, "color": [0, 0, 0], "trainId": 4},
#     {"name": "truck", "id": 1, "color": [0, 0, 0], "trainId": 10},
#     {"name": "fence", "id": 2, "color": [0, 0, 0], "trainId": 15},
#     {"name": "billboard", "id": 3, "color": [0, 0, 0], "trainId": 17},
#     {"name": "bus", "id": 4, "color": [0, 0, 0], "trainId": 11},
#     {"name": "out of roi", "id": 5, "color": [0, 0, 0], "trainId": 255},
#     {"name": "curb", "id": 6, "color": [0, 0, 0], "trainId": 13},
#     {"name": "obs-str-bar-fallback", "id": 7, "color": [0, 0, 0], "trainId": 21},
#     {"name": "tunnel", "id": 8, "color": [0, 0, 0], "trainId": 23},
#     {"name": "non-drivable fallback", "id": 9, "color": [0, 0, 0], "trainId": 3},
#     {"name": "bridge", "id": 10, "color": [0, 0, 0], "trainId": 23},
#     {"name": "road", "id": 11, "color": [0, 0, 0], "trainId": 0},
#     {"name": "wall", "id": 12, "color": [0, 0, 0], "trainId": 14},
#     {"name": "traffic sign", "id": 13, "color": [0, 0, 0], "trainId": 18},
#     {"name": "trailer", "id": 14, "color": [0, 0, 0], "trainId": 12},
#     {"name": "animal", "id": 15, "color": [0, 0, 0], "trainId": 4},
#     {"name": "building", "id": 16, "color": [0, 0, 0], "trainId": 22},
#     {"name": "sky", "id": 17, "color": [0, 0, 0], "trainId": 25},
#     {"name": "drivable fallback", "id": 18, "color": [0, 0, 0], "trainId": 1},
#     {"name": "guard rail", "id": 19, "color": [0, 0, 0], "trainId": 16},
#     {"name": "bicycle", "id": 20, "color": [0, 0, 0], "trainId": 7},
#     {"name": "traffic light", "id": 21, "color": [0, 0, 0], "trainId": 19},
#     {"name": "polegroup", "id": 22, "color": [0, 0, 0], "trainId": 20},
#     {"name": "motorcycle", "id": 23, "color": [0, 0, 0], "trainId": 6},
#     {"name": "car", "id": 24, "color": [0, 0, 0], "trainId": 9},
#     {"name": "parking", "id": 25, "color": [0, 0, 0], "trainId": 1},
#     {"name": "fallback background", "id": 26, "color": [0, 0, 0], "trainId": 25},
#     {"name": "license plate", "id": 27, "color": [0, 0, 0], "trainId": 255},
#     {"name": "rectification border", "id": 28, "color": [0, 0, 0], "trainId": 255},
#     {"name": "train", "id": 29, "color": [0, 0, 0], "trainId": 255},
#     {"name": "rider", "id": 30, "color": [0, 0, 0], "trainId": 5},
#     {"name": "rail track", "id": 31, "color": [0, 0, 0], "trainId": 3},
#     {"name": "sidewalk", "id": 32, "color": [0, 0, 0], "trainId": 2},
#     {"name": "caravan", "id": 33, "color": [0, 0, 0], "trainId": 12},
#     {"name": "pole", "id": 34, "color": [0, 0, 0], "trainId": 20},
#     {"name": "vegetation", "id": 35, "color": [0, 0, 0], "trainId": 24},
#     {"name": "autorickshaw", "id": 36, "color": [0, 0, 0], "trainId": 8},
#     {"name": "vehicle fallback", "id": 37, "color": [0, 0, 0], "trainId": 12},
#     {"name": "unlabel", "id":255, "color":[0,0,0], "trainId": 255},
# ]


# def register_IDD_new():
    
    
#     # meta = _get_cs20k_full_meta()
#     # for name, dirname in [("train", "train"), ("val", "val")]:
#     # dirname = 'train'
#     lb_map = {}
#     for el in mseg_labels_info:
#         lb_map[el['id']] = el['trainId']
#     for n, anp in [("train", "train"), ("train_1", "train_1"), ("train_2", "train_2"), ("val", "val")]:
#         name = f"idd_sem_seg_{n}"
#         annpath = f'auto_uni_seg/datasets/IDD/{anp}.txt'
#         DatasetCatalog.register(
#             name, lambda x=annpath : IDD_train(x)
#         )
        
#         MetadataCatalog.get(name).set(
#             stuff_classes=["road", "drivable fallback or parking", "sidewalk", "non-drivable fallback or rail track", "person or animal", "out of roi or rider", "motorcycle", "bicycle", "autorickshaw", "car", "truck", "bus", "trailer or caravan or vehicle fallback", "curb", "wall", "fence", "guard rail", "billboard", "traffic sign", "traffic light", "polegroup or pole", "obs-str-bar-fallback", "building", "tunnel or bridge", "vegetation", "sky or fallback background"],
#             stuff_dataset_id_to_contiguous_id=lb_map,
#             thing_dataset_id_to_contiguous_id=lb_map,
#             stuff_colors=[[ 97, 187, 159],[109, 220, 239],     [236,   2, 131],     [  9,  60, 225],     [198, 221, 182],     [ 24, 192, 253],     [ 12,  50, 106],     [103,  81, 234],     [ 77, 252, 183],     [ 14, 159,  57],     [ 81, 255, 160],     [ 79, 127, 244],     [224, 216, 151],     [ 66,  69, 176],     [174, 203, 111],     [145, 208,  42],     [180, 241, 180],     [215,  48,  75],     [218, 118, 115],     [153,  29, 134],     [ 70,  94, 120],     [160, 111,   0],     [154, 172, 132],     [ 18,  58,  98],     [128, 109,  69],     [225, 115,  77]],
#             evaluator_type="sem_seg",
#             ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
#         )

# register_IDD_new()