import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import os.path as osp

# labels_info = [
# {"id": 0, "trainId": 255, "name": "unlabeled"},
# {"id": 1, "trainId": 0, "name": "ego vehicle"},
# {"id": 2, "trainId": 255, "name": "rectification border"},
# {"id": 3, "trainId": 255, "name": "out of roi"},
# {"id": 4, "trainId": 255, "name": "static"},
# {"id": 5, "trainId": 255, "name": "dynamic"},
# {"id": 6, "trainId": 255, "name": "ground"},
# {"id": 7, "trainId": 1, "name": "road"},
# {"id": 8, "trainId": 2, "name": "sidewalk"},
# {"id": 9, "trainId": 255, "name": "parking"},
# {"id": 10, "trainId": 255, "name": "rail track"},
# {"id": 11, "trainId": 3, "name": "building"},
# {"id": 12, "trainId": 4, "name": "wall"},
# {"id": 13, "trainId": 5, "name": "fence"},
# {"id": 14, "trainId": 6, "name": "guard rail"},
# {"id": 15, "trainId": 255, "name": "bridge"},
# {"id": 16, "trainId": 255, "name": "tunnel"},
# {"id": 17, "trainId": 7, "name": "pole"},
# {"id": 18, "trainId": 255, "name": "polegroup"},
# {"id": 19, "trainId": 8, "name": "traffic light"},
# {"id": 20, "trainId": 9, "name": "traffic sign"},
# {"id": 21, "trainId": 10, "name": "vegetation"},
# {"id": 22, "trainId": 11, "name": "terrain"},
# {"id": 23, "trainId": 12, "name": "sky"},
# {"id": 24, "trainId": 13, "name": "person"},
# {"id": 25, "trainId": 14, "name": "rider"},
# {"id": 26, "trainId": 15, "name": "car"},
# {"id": 27, "trainId": 16, "name": "truck"},
# {"id": 28, "trainId": 17, "name": "bus"},
# {"id": 29, "trainId": 255, "name": "caravan"},
# {"id": 30, "trainId": 255, "name": "trailer"},
# {"id": 31, "trainId": 255, "name": "train"},
# {"id": 32, "trainId": 18, "name": "motorcycle"},
# {"id": 33, "trainId": 19, "name": "bicycle"},
# {"id": 34, "trainId": 20, "name": "pickup"},
# {"id": 35, "trainId": 21, "name": "van"},
# {"id": 36, "trainId": 22, "name": "billboard"},
# {"id": 37, "trainId": 23, "name": "street-light"},
# {"id": 38, "trainId": 24, "name": "road-marking"},
# ]

# annpath = f'auto_uni_seg/datasets/WD2/validation.txt'
# def wilddash2():
#     # assert mode in ('train', 'eval', 'test')

#     with open(annpath, 'r') as fr:
#         pairs = fr.read().splitlines()
#     img_paths, lb_paths = [], []
#     for pair in pairs:
#         imgpth, lbpth = pair.split(',')
#         img_paths.append(osp.join(dataroot, imgpth))
#         lb_paths.append(osp.join(dataroot, lbpth))

#     assert len(img_paths) == len(lb_paths)
#     dataset_dicts = []
#     for (img_path, gt_path) in zip(img_paths, lb_paths):
#         record = {}
#         record["file_name"] = img_path
#         record["sem_seg_file_name"] = gt_path
#         dataset_dicts.append(record)

#     return dataset_dicts


# def register_wilddash2():
    
    
#     # meta = _get_ade20k_full_meta()
#     # for name, dirname in [("train", "train"), ("val", "val")]:
#     # dirname = 'train'
#     lb_map = {}
#     for el in labels_info:
#         lb_map[el['id']] = el['trainId']

#     name = f"wilddash2_sem_seg_val"
#     DatasetCatalog.register(
#         name, wilddash2
#     )
    
#     MetadataCatalog.get(name).set(
#         stuff_classes=["ego vehicle", "road", "sidewalk", "building", "wall", "fence", "guard rail", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "motorcycle", "bicycle", "pickup", "van", "billboard", "street-light", "road-marking"],
#         stuff_dataset_id_to_contiguous_id=lb_map,
#         thing_dataset_id_to_contiguous_id=lb_map,
#         evaluator_type="sem_seg",
#         ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
#     )


# # _root = os.getenv("DETECTRON2_DATASETS", "datasets")
# register_wilddash2()

# train_annpath = f'auto_uni_seg/datasets/WD2/train.txt'
# def wilddash2_train(anp):

#     with open(anp, 'r') as fr:
#         pairs = fr.read().splitlines()
#     img_paths, lb_paths = [], []
#     for pair in pairs:
#         imgpth, lbpth = pair.split(',')
#         img_paths.append(osp.join(dataroot, imgpth))
#         lb_paths.append(osp.join(dataroot, lbpth))

#     assert len(img_paths) == len(lb_paths)
#     dataset_dicts = []
#     for (img_path, gt_path) in zip(img_paths, lb_paths):
#         record = {}
#         record["file_name"] = img_path
#         record["sem_seg_file_name"] = gt_path
#         dataset_dicts.append(record)

#     return dataset_dicts


# def register_wilddash2_train():
    
    
#     # meta = _get_wilddash220k_full_meta()
#     # for name, dirname in [("train", "train"), ("val", "val")]:
#     # dirname = 'train'
#     lb_map = {}
#     for el in labels_info:
#         lb_map[el['id']] = el['trainId']
#     for n, anp in [("train", "train"), ("train_1", "train_1"), ("train_2", "train_2")]:
#         name = f"wilddash2_sem_seg_{n}"
#         annpath = f'auto_uni_seg/datasets/WD2/{anp}.txt'
#         DatasetCatalog.register(
#             name, lambda x=annpath : wilddash2_train(x)
#         )
        
#         MetadataCatalog.get(name).set(
#             stuff_classes=["ego vehicle", "road", "sidewalk", "building", "wall", "fence", "guard rail", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "motorcycle", "bicycle", "pickup", "van", "billboard", "street-light", "road-marking"],
#             stuff_dataset_id_to_contiguous_id=lb_map,
#             thing_dataset_id_to_contiguous_id=lb_map,
#             evaluator_type="sem_seg",
#             ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
#         )

# register_wilddash2_train()


all_lb_names = ['egovehicle', 'road', 'sidewalk', 'building', 'wall', 'fence', 'guardrail', 'pole', 'trafficlight', 'trafficsignfront', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pickup', 'van', 'billboard', 'streetlight', 'roadmarking', 'trafficsignframe', 'utilitypole', 'motorcyclist', 'bicyclist', 'otherrider', 'bird', 'groundanimal', 'curb', 'trafficsignany', 'trafficsignback', 'trashcan', 'otherbarrier', 'othervehicle', 'autorickshaw', 'bench', 'mountain', 'tramtrack', 'wheeledslow', 'boat', 'bikelane', 'bikelanesidewalk', 'banner', 'dashcammount', 'pedestrianarea', 'crosswalkplain', 'crosswalkzebra']

all_lb_infos = [{'name': 'unlabeled', 'id': 0, 'evaluate': False, 'trainId': 255},
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
{'name': 'polegroup', 'id': 18, 'evaluate': False, 'trainId': 7},
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
{'name': 'manhole', 'id': 41, 'evaluate': False, 'trainId': 1},
{'name': 'phonebooth', 'id': 42, 'evaluate': False, 'trainId': 255},
{'name': 'pothole', 'id': 43, 'evaluate': False, 'trainId': 255},
{'name': 'bikerack', 'id': 44, 'evaluate': False, 'trainId': 255},
{'name': 'trafficsignframe', 'id': 45, 'evaluate': True, 'trainId': 7},
{'name': 'utilitypole', 'id': 46, 'evaluate': True, 'trainId': 7},
{'name': 'motorcyclist', 'id': 47, 'evaluate': True, 'trainId': 14},
{'name': 'bicyclist', 'id': 48, 'evaluate': True, 'trainId': 14},
{'name': 'otherrider', 'id': 49, 'evaluate': True, 'trainId': 14},
{'name': 'bird', 'id': 50, 'evaluate': True, 'trainId': 255},
{'name': 'groundanimal', 'id': 51, 'evaluate': True, 'trainId': 255},
{'name': 'curb', 'id': 52, 'evaluate': True, 'trainId': 2},
{'name': 'trafficsignany', 'id': 53, 'evaluate': True, 'trainId': 9},
{'name': 'trafficsignback', 'id': 54, 'evaluate': True, 'trainId': 9},
{'name': 'trashcan', 'id': 55, 'evaluate': True, 'trainId': 255},
{'name': 'otherbarrier', 'id': 56, 'evaluate': True, 'trainId': 4},
{'name': 'othervehicle', 'id': 57, 'evaluate': True, 'trainId': 255},
{'name': 'autorickshaw', 'id': 58, 'evaluate': True, 'trainId': 18},
{'name': 'bench', 'id': 59, 'evaluate': True, 'trainId': 255},
{'name': 'mountain', 'id': 60, 'evaluate': True, 'trainId': 255},
{'name': 'tramtrack', 'id': 61, 'evaluate': True, 'trainId': 1},
{'name': 'wheeledslow', 'id': 62, 'evaluate': True, 'trainId': 255},
{'name': 'boat', 'id': 63, 'evaluate': True, 'trainId': 255},
{'name': 'bikelane', 'id': 64, 'evaluate': True, 'trainId': 1},
{'name': 'bikelanesidewalk', 'id': 65, 'evaluate': True, 'trainId': 2},
{'name': 'banner', 'id': 66, 'evaluate': True, 'trainId': 255},
{'name': 'dashcammount', 'id': 67, 'evaluate': True, 'trainId': 0},
{'name': 'water', 'id': 68, 'evaluate': False, 'trainId': 255},
{'name': 'sand', 'id': 69, 'evaluate': False, 'trainId': 255},
{'name': 'pedestrianarea', 'id': 70, 'evaluate': True, 'trainId': 1},
{'name': 'firehydrant', 'id': 71, 'evaluate': False, 'trainId': 255},
{'name': 'cctvcamera', 'id': 72, 'evaluate': False, 'trainId': 255},
{'name': 'snow', 'id': 73, 'evaluate': False, 'trainId': 255},
{'name': 'catchbasin', 'id': 74, 'evaluate': False, 'trainId': 1},
{'name': 'crosswalkplain', 'id': 75, 'evaluate': True, 'trainId': 1},
{'name': 'crosswalkzebra', 'id': 76, 'evaluate': True, 'trainId': 1},
{'name': 'manholesidewalk', 'id': 77, 'evaluate': False, 'trainId': 2},
{'name': 'curbterrain', 'id': 78, 'evaluate': False, 'trainId': 11},
{'name': 'servicelane', 'id': 79, 'evaluate': False, 'trainId': 1},
{'name': 'curbcut', 'id': 80, 'evaluate': False, 'trainId': 2}] 

dataroot = 'datasets/wd2'
annpath = f'auto_uni_seg/datasets/WD2/validation.txt'
def wilddash2_new():
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


def register_wilddash2_new():
    
    
    # meta = _get_ade20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in all_lb_infos:
        if el['trainId'] > 24:
            lb_map[el['id']] = 25 #el['trainId']
        else:
            lb_map[el['id']] = el['trainId']
            

    name = f"wilddash2_new_sem_seg_val"
    DatasetCatalog.register(
        name, wilddash2_new
    )
    
    MetadataCatalog.get(name).set(
        stuff_classes= ['ego vehicle', 'road', 'sidewalk', 'building', 'wall', 'fence', 'guard rail', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pickup', 'van', 'billboard', 'street-light', 'road-marking', 'void'],
        stuff_dataset_id_to_contiguous_id=lb_map,
        thing_dataset_id_to_contiguous_id=lb_map,
        stuff_colors = [[214, 92, 19], [97, 187, 159], [236, 2, 131], [154, 172, 132], [174, 203, 111], [145, 208, 42], [180, 241, 180], [126, 101, 165], [153, 29, 134], [169, 102, 174], [128, 109, 69], [155, 117, 111], [225, 115, 77], [198, 221, 182], [24, 192, 253], [14, 159, 57], [81, 255, 160], [79, 127, 244], [12, 50, 106], [103, 81, 234], [224, 216, 151], [109, 64, 73], [215, 48, 75], [26, 184, 83], [226, 27, 43], [160, 111, 0]],
        evaluator_type="sem_seg",
        ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
    )


# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_wilddash2_new()

test_dataroot = 'datasets/wilddash2'

test_annpath = f'auto_uni_seg/datasets/WD2/test.txt'
def wilddash2_new_test():
    # assert mode in ('train', 'eval', 'test')

    with open(test_annpath, 'r') as fr:
        pairs = fr.read().splitlines()
    img_paths, lb_paths = [], []
    for pair in pairs:
        imgpth = pair
        img_paths.append(osp.join(test_dataroot, imgpth))
        # lb_paths.append(osp.join(dataroot, lbpth))

    # assert len(img_paths) == len(lb_paths)
    dataset_dicts = []
    for img_path in img_paths:
        record = {}
        record["file_name"] = img_path
        # record["sem_seg_file_name"] = gt_path
        dataset_dicts.append(record)

    return dataset_dicts


def register_wilddash2_new_test():
    
    
    # meta = _get_ade20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in all_lb_infos:
        if el['trainId'] > 24:
            lb_map[el['id']] = 25 #el['trainId']
        else:
            lb_map[el['id']] = el['trainId']
            

    name = f"wilddash2_new_sem_seg_test"
    DatasetCatalog.register(
        name, wilddash2_new_test
    )
    
    MetadataCatalog.get(name).set(
        stuff_classes= ['ego vehicle', 'road', 'sidewalk', 'building', 'wall', 'fence', 'guard rail', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pickup', 'van', 'billboard', 'street-light', 'road-marking', 'void'],
        stuff_dataset_id_to_contiguous_id=lb_map,
        thing_dataset_id_to_contiguous_id=lb_map,
        stuff_colors = [[214, 92, 19], [97, 187, 159], [236, 2, 131], [154, 172, 132], [174, 203, 111], [145, 208, 42], [180, 241, 180], [126, 101, 165], [153, 29, 134], [169, 102, 174], [128, 109, 69], [155, 117, 111], [225, 115, 77], [198, 221, 182], [24, 192, 253], [14, 159, 57], [81, 255, 160], [79, 127, 244], [12, 50, 106], [103, 81, 234], [224, 216, 151], [109, 64, 73], [215, 48, 75], [26, 184, 83], [226, 27, 43], [0, 0, 0]],
        evaluator_type="sem_seg",
        ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
    )


# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_wilddash2_new_test()

# partial_label_name = ['ego vehicle', 'road', 'sidewalk', 'building', 'wall', 'fence', 'guard rail', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pickup', 'van', 'billboard', 'street-light', 'road-marking']
# partial_label_info = [{'name': 'unlabeled', 'id': 0, 'evaluate': False, 'trainId': 255},
# {'name': 'ego vehicle', 'id': 1, 'evaluate': True, 'trainId': 0},
# {'name': 'rectification border', 'id': 2, 'evaluate': False, 'trainId': 255},
# {'name': 'out of roi', 'id': 3, 'evaluate': False, 'trainId': 255},
# {'name': 'static', 'id': 4, 'evaluate': False, 'trainId': 255},
# {'name': 'dynamic', 'id': 5, 'evaluate': False, 'trainId': 255},
# {'name': 'ground', 'id': 6, 'evaluate': False, 'trainId': 255},
# {'name': 'road', 'id': 7, 'evaluate': True, 'trainId': 1},
# {'name': 'sidewalk', 'id': 8, 'evaluate': True, 'trainId': 2},
# {'name': 'parking', 'id': 9, 'evaluate': False, 'trainId': 255},
# {'name': 'rail track', 'id': 10, 'evaluate': False, 'trainId': 255},
# {'name': 'building', 'id': 11, 'evaluate': True, 'trainId': 3},
# {'name': 'wall', 'id': 12, 'evaluate': True, 'trainId': 4},
# {'name': 'fence', 'id': 13, 'evaluate': True, 'trainId': 5},
# {'name': 'guard rail', 'id': 14, 'evaluate': True, 'trainId': 6},
# {'name': 'bridge', 'id': 15, 'evaluate': False, 'trainId': 255},
# {'name': 'tunnel', 'id': 16, 'evaluate': False, 'trainId': 255},
# {'name': 'pole', 'id': 17, 'evaluate': True, 'trainId': 7},
# {'name': 'polegroup', 'id': 18, 'evaluate': False, 'trainId': 255},
# {'name': 'traffic light', 'id': 19, 'evaluate': True, 'trainId': 8},
# {'name': 'traffic sign', 'id': 20, 'evaluate': True, 'trainId': 9},
# {'name': 'vegetation', 'id': 21, 'evaluate': True, 'trainId': 10},
# {'name': 'terrain', 'id': 22, 'evaluate': True, 'trainId': 11},
# {'name': 'sky', 'id': 23, 'evaluate': True, 'trainId': 12},
# {'name': 'person', 'id': 24, 'evaluate': True, 'trainId': 13},
# {'name': 'rider', 'id': 25, 'evaluate': True, 'trainId': 14},
# {'name': 'car', 'id': 26, 'evaluate': True, 'trainId': 15},
# {'name': 'truck', 'id': 27, 'evaluate': True, 'trainId': 16},
# {'name': 'bus', 'id': 28, 'evaluate': True, 'trainId': 17},
# {'name': 'caravan', 'id': 29, 'evaluate': False, 'trainId': 255},
# {'name': 'trailer', 'id': 30, 'evaluate': False, 'trainId': 255},
# {'name': 'train', 'id': 31, 'evaluate': False, 'trainId': 255},
# {'name': 'motorcycle', 'id': 32, 'evaluate': True, 'trainId': 18},
# {'name': 'bicycle', 'id': 33, 'evaluate': True, 'trainId': 19},
# {'name': 'pickup', 'id': 34, 'evaluate': True, 'trainId': 20},
# {'name': 'van', 'id': 35, 'evaluate': True, 'trainId': 21},
# {'name': 'billboard', 'id': 36, 'evaluate': True, 'trainId': 22},
# {'name': 'street-light', 'id': 37, 'evaluate': True, 'trainId': 23},
# {'name': 'road-marking', 'id': 38, 'evaluate': True, 'trainId': 24}]