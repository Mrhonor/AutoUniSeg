import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import os.path as osp

labels_info = [
{'name': 'Bird', 'id': 0, 'color': [165, 42, 42], 'trainId': 0},
{'name': 'Ground Animal', 'id': 1, 'color': [0, 192, 0], 'trainId': 1},
{'name': 'Curb', 'id': 2, 'color': [196, 196, 196], 'trainId': 2},
{'name': 'Fence', 'id': 3, 'color': [190, 153, 153], 'trainId': 3},
{'name': 'Guard Rail', 'id': 4, 'color': [180, 165, 180], 'trainId': 4},
{'name': 'Barrier', 'id': 5, 'color': [90, 120, 150], 'trainId': 5},
{'name': 'Wall', 'id': 6, 'color': [102, 102, 156], 'trainId': 6},
{'name': 'Bike Lane', 'id': 7, 'color': [128, 64, 255], 'trainId': 7},
{'name': 'Crosswalk - Plain', 'id': 8, 'color': [140, 140, 200], 'trainId': 8},
{'name': 'Curb Cut', 'id': 9, 'color': [170, 170, 170], 'trainId': 9},
{'name': 'Parking', 'id': 10, 'color': [250, 170, 160], 'trainId': 10},
{'name': 'Pedestrian Area', 'id': 11, 'color': [96, 96, 96], 'trainId': 11},
{'name': 'Rail Track', 'id': 12, 'color': [230, 150, 140], 'trainId': 12},
{'name': 'Road', 'id': 13, 'color': [128, 64, 128], 'trainId': 13},
{'name': 'Service Lane', 'id': 14, 'color': [110, 110, 110], 'trainId': 14},
{'name': 'Sidewalk', 'id': 15, 'color': [244, 35, 232], 'trainId': 15},
{'name': 'Bridge', 'id': 16, 'color': [150, 100, 100], 'trainId': 16}, 
{'name': 'Building', 'id': 17, 'color': [70, 70, 70], 'trainId': 17}, 
{'name': 'Tunnel', 'id': 18, 'color': [150, 120, 90], 'trainId': 18}, 
{'name': 'Person', 'id': 19, 'color': [220, 20, 60], 'trainId': 19}, 
{'name': 'Bicyclist', 'id': 20, 'color': [255, 0, 0], 'trainId': 20}, 
{'name': 'Motorcyclist', 'id': 21, 'color': [255, 0, 100], 'trainId': 21}, 
{'name': 'Other Rider', 'id': 22, 'color': [255, 0, 200], 'trainId': 22}, 
{'name': 'Lane Marking - Crosswalk', 'id': 23, 'color': [200, 128, 128], 'trainId': 23}, 
{'name': 'Lane Marking - General', 'id': 24, 'color': [255, 255, 255], 'trainId': 24}, 
{'name': 'Mountain', 'id': 25, 'color': [64, 170, 64], 'trainId': 25}, 
{'name': 'Sand', 'id': 26, 'color': [230, 160, 50], 'trainId': 26}, 
{'name': 'Sky', 'id': 27, 'color': [70, 130, 180], 'trainId': 27},
{'name': 'Snow', 'id': 28, 'color': [190, 255, 255], 'trainId': 28}, 
{'name': 'Terrain', 'id': 29, 'color': [152, 251, 152], 'trainId': 29},
{'name': 'Vegetation', 'id': 30, 'color': [107, 142, 35], 'trainId': 30},
{'name': 'Water', 'id': 31, 'color': [0, 170, 30], 'trainId': 31}, 
{'name': 'Banner', 'id': 32, 'color': [255, 255, 128], 'trainId': 32},
{'name': 'Bench', 'id': 33, 'color': [250, 0, 30], 'trainId': 33}, 
{'name': 'Bike Rack', 'id': 34, 'color': [100, 140, 180], 'trainId': 34}, 
{'name': 'Billboard', 'id': 35, 'color': [220, 220, 220], 'trainId': 35}, 
{'name': 'Catch Basin', 'id': 36, 'color': [220, 128, 128], 'trainId': 36}, 
{'name': 'CCTV Camera', 'id': 37, 'color': [222, 40, 40], 'trainId': 37},
{'name': 'Fire Hydrant', 'id': 38, 'color': [100, 170, 30], 'trainId': 38},
{'name': 'Junction Box', 'id': 39, 'color': [40, 40, 40], 'trainId': 39}, 
{'name': 'Mailbox', 'id': 40, 'color': [33, 33, 33], 'trainId': 255},
{'name': 'Manhole', 'id': 41, 'color': [100, 128, 160], 'trainId': 40},
{'name': 'Phone Booth', 'id': 42, 'color': [142, 0, 0], 'trainId': 41},
{'name': 'Pothole', 'id': 43, 'color': [70, 100, 150], 'trainId': 42}, 
{'name': 'Street Light', 'id': 44, 'color': [210, 170, 100], 'trainId': 43},
{'name': 'Pole', 'id': 45, 'color': [153, 153, 153], 'trainId': 44}, 
{'name': 'Traffic Sign Frame', 'id': 46, 'color': [128, 128, 128], 'trainId': 45},
{'name': 'Utility Pole', 'id': 47, 'color': [0, 0, 80], 'trainId': 46}, 
{'name': 'Traffic Light', 'id': 48, 'color': [250, 170, 30], 'trainId': 47},
{'name': 'Traffic Sign (Back)', 'id': 49, 'color': [192, 192, 192], 'trainId': 48},
{'name': 'Traffic Sign (Front)', 'id': 50, 'color': [220, 220, 0], 'trainId': 49},
{'name': 'Trash Can', 'id': 51, 'color': [140, 140, 20], 'trainId': 50},
{'name': 'Bicycle', 'id': 52, 'color': [119, 11, 32], 'trainId': 51}, 
{'name': 'Boat', 'id': 53, 'color': [150, 0, 255], 'trainId': 52}, 
{'name': 'Bus', 'id': 54, 'color': [0, 60, 100], 'trainId': 53}, 
{'name': 'Car', 'id': 55, 'color': [0, 0, 142], 'trainId': 54}, 
{'name': 'Caravan', 'id': 56, 'color': [0, 0, 90], 'trainId': 55},
{'name': 'Motorcycle', 'id': 57, 'color': [0, 0, 230], 'trainId': 56}, 
{'name': 'On Rails', 'id': 58, 'color': [0, 80, 100], 'trainId': 57},
{'name': 'Other Vehicle', 'id': 59, 'color': [128, 64, 64], 'trainId': 58},
{'name': 'Trailer', 'id': 60, 'color': [0, 0, 110], 'trainId': 59},
{'name': 'Truck', 'id': 61, 'color': [0, 0, 70], 'trainId': 60},
{'name': 'Wheeled Slow', 'id': 62, 'color': [0, 0, 192], 'trainId': 61},
{'name': 'Car Mount', 'id': 63, 'color': [32, 32, 32], 'trainId': 62}, 
{'name': 'Ego Vehicle', 'id': 64, 'color': [120, 10, 10], 'trainId': 63},
{'name': 'Unlabeled', 'id': 65, 'color': [0, 0, 0], 'trainId': 255},
]

dataroot = 'datasets/mapi'
annpath = f'auto_uni_seg/datasets/mapi/validation.txt'
def mapi():
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


def register_mapi():
    
    
    # meta = _get_ade20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in labels_info:
        lb_map[el['id']] = el['trainId']

    name = f"mapi_sem_seg_val"
    DatasetCatalog.register(
        name, mapi
    )
    
    MetadataCatalog.get(name).set(
        stuff_classes=["Bird", "Ground Animal", "Curb", "Fence", "Guard Rail", "Barrier", "Wall", "Bike Lane", "Crosswalk - Plain", "Curb Cut", "Parking", "Pedestrian Area", "Rail Track", "Road", "Service Lane", "Sidewalk", "Bridge", "Building", "Tunnel", "Person", "Bicyclist", "Motorcyclist", "Other Rider", "Lane Marking - Crosswalk", "Lane Marking - General", "Mountain", "Sand", "Sky", "Snow", "Terrain", "Vegetation", "Water", "Banner", "Bench", "Bike Rack", "Billboard", "Catch Basin", "CCTV Camera", "Fire Hydrant", "Junction Box", "Manhole", "Phone Booth", "Pothole", "Street Light", "Pole", "Traffic Sign Frame", "Utility Pole", "Traffic Light", "Traffic Sign (Back)", "Traffic Sign (Front)", "Trash Can", "Bicycle", "Boat", "Bus", "Car", "Caravan", "Motorcycle", "On Rails", "Other Vehicle", "Trailer", "Truck", "Wheeled Slow", "Car Mount", "Ego Vehicle"],
        stuff_dataset_id_to_contiguous_id=lb_map,
        thing_dataset_id_to_contiguous_id=lb_map,
        stuff_colors=[[251, 225,  86],[ 50, 105,  96],      [ 66,  69, 176],      [145, 208,  42],      [180, 241, 180],      [212, 221, 176],      [174, 203, 111],      [230, 246,  34],      [  0,  81, 175],      [113, 216, 102],      [109, 220, 239],      [178, 109,  61],      [  9,  60, 225],      [ 97, 187, 159],      [195, 172, 234],      [236,   2, 131],      [ 76,  66, 237],      [154, 172, 132],      [ 18,  58,  98],      [198, 221, 182],      [199, 215,  29],      [ 64, 129, 196],      [ 24, 192, 253],      [232, 182, 126],      [226,  27,  43],      [252, 168, 166],      [135, 161, 131],      [225, 115,  77],      [148, 126,  90],      [155, 117, 111],      [128, 109,  69],      [ 79,  24, 196],      [ 30, 123,  47],      [ 38, 181, 228],      [141,  17,  60],      [215,  48,  75],      [236, 223, 130],      [208, 211,  35],      [ 73, 208,  97],      [ 58,  34,   9],      [ 34, 168, 123],      [177, 158, 120],      [100,  32,  22],      [ 26, 184,  83],      [ 70,  94, 120],      [186, 140, 216],      [126, 101, 165],      [153,  29, 134],      [169, 102, 174],      [218, 118, 115],      [ 30,  47, 184],      [103,  81, 234],      [106,  48, 148],      [ 79, 127, 244],      [ 14, 159,  57],      [109,  64,  73],      [ 12,  50, 106],      [106, 248, 185],      [125, 123, 180],      [ 80, 108,  57],      [ 81, 255, 160],      [230, 175, 199],      [131, 218,  87],      [214,  92,  19]],
        evaluator_type="sem_seg",
        ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
    )



# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_mapi()

train_annpath = f'auto_uni_seg/datasets/mapi/training.txt'
def mapi_train(anp):
    # assert mode in ('train', 'eval', 'test')

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


def register_mapi_train():
    
    
    # meta = _get_ade20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in labels_info:
        lb_map[el['id']] = el['trainId']
    for n, anp in [("train", "training"), ("train_1", "training_1"), ("train_2", "training_2")]:
        name = f"mapi_sem_seg_{n}"
        annpath = f'auto_uni_seg/datasets/mapi/{anp}.txt'
        DatasetCatalog.register(
            name, lambda x=annpath : mapi_train(x)
        )
        
        MetadataCatalog.get(name).set(
            stuff_classes=["Bird", "Ground Animal", "Curb", "Fence", "Guard Rail", "Barrier", "Wall", "Bike Lane", "Crosswalk - Plain", "Curb Cut", "Parking", "Pedestrian Area", "Rail Track", "Road", "Service Lane", "Sidewalk", "Bridge", "Building", "Tunnel", "Person", "Bicyclist", "Motorcyclist", "Other Rider", "Lane Marking - Crosswalk", "Lane Marking - General", "Mountain", "Sand", "Sky", "Snow", "Terrain", "Vegetation", "Water", "Banner", "Bench", "Bike Rack", "Billboard", "Catch Basin", "CCTV Camera", "Fire Hydrant", "Junction Box", "Manhole", "Phone Booth", "Pothole", "Street Light", "Pole", "Traffic Sign Frame", "Utility Pole", "Traffic Light", "Traffic Sign (Back)", "Traffic Sign (Front)", "Trash Can", "Bicycle", "Boat", "Bus", "Car", "Caravan", "Motorcycle", "On Rails", "Other Vehicle", "Trailer", "Truck", "Wheeled Slow", "Car Mount", "Ego Vehicle"],
            stuff_dataset_id_to_contiguous_id=lb_map,
            thing_dataset_id_to_contiguous_id=lb_map,
            stuff_colors=[[251, 225,  86],[ 50, 105,  96],      [ 66,  69, 176],      [145, 208,  42],      [180, 241, 180],      [212, 221, 176],      [174, 203, 111],      [230, 246,  34],      [  0,  81, 175],      [113, 216, 102],      [109, 220, 239],      [178, 109,  61],      [  9,  60, 225],      [ 97, 187, 159],      [195, 172, 234],      [236,   2, 131],      [ 76,  66, 237],      [154, 172, 132],      [ 18,  58,  98],      [198, 221, 182],      [199, 215,  29],      [ 64, 129, 196],      [ 24, 192, 253],      [232, 182, 126],      [226,  27,  43],      [252, 168, 166],      [135, 161, 131],      [225, 115,  77],      [148, 126,  90],      [155, 117, 111],      [128, 109,  69],      [ 79,  24, 196],      [ 30, 123,  47],      [ 38, 181, 228],      [141,  17,  60],      [215,  48,  75],      [236, 223, 130],      [208, 211,  35],      [ 73, 208,  97],      [ 58,  34,   9],      [ 34, 168, 123],      [177, 158, 120],      [100,  32,  22],      [ 26, 184,  83],      [ 70,  94, 120],      [186, 140, 216],      [126, 101, 165],      [153,  29, 134],      [169, 102, 174],      [218, 118, 115],      [ 30,  47, 184],      [103,  81, 234],      [106,  48, 148],      [ 79, 127, 244],      [ 14, 159,  57],      [109,  64,  73],      [ 12,  50, 106],      [106, 248, 185],      [125, 123, 180],      [ 80, 108,  57],      [ 81, 255, 160],      [230, 175, 199],      [131, 218,  87],      [214,  92,  19]],
            evaluator_type="sem_seg",
            ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
        )




register_mapi_train()

Mseg_label_info = [{'name': 'Bird', 'id': 0, 'color': [165, 42, 42], 'trainId': 0},
{'name': 'Ground Animal', 'id': 1, 'color': [0, 192, 0], 'trainId': 1},
{'name': 'Curb', 'id': 2, 'color': [196, 196, 196], 'trainId': 9},
{'name': 'Fence', 'id': 3, 'color': [190, 153, 153], 'trainId': 28},
{'name': 'Guard Rail', 'id': 4, 'color': [180, 165, 180], 'trainId': 29},
{'name': 'Barrier', 'id': 5, 'color': [90, 120, 150], 'trainId': 16},
{'name': 'Wall', 'id': 6, 'color': [102, 102, 156], 'trainId': 17},
{'name': 'Bike Lane', 'id': 7, 'color': [128, 64, 255], 'trainId': 7},
{'name': 'Crosswalk - Plain', 'id': 8, 'color': [140, 140, 200], 'trainId': 7},
{'name': 'Curb Cut', 'id': 9, 'color': [170, 170, 170], 'trainId': 9},
{'name': 'Parking', 'id': 10, 'color': [250, 170, 160], 'trainId': 7},
{'name': 'Pedestrian Area', 'id': 11, 'color': [96, 96, 96], 'trainId': 9},
{'name': 'Rail Track', 'id': 12, 'color': [230, 150, 140], 'trainId': 6},
{'name': 'Road', 'id': 13, 'color': [128, 64, 128], 'trainId': 7},
{'name': 'Service Lane', 'id': 14, 'color': [110, 110, 110], 'trainId': 7},
{'name': 'Sidewalk', 'id': 15, 'color': [244, 35, 232], 'trainId': 9},
{'name': 'Bridge', 'id': 16, 'color': [150, 100, 100], 'trainId': 3},
{'name': 'Building', 'id': 17, 'color': [70, 70, 70], 'trainId': 4},
{'name': 'Tunnel', 'id': 18, 'color': [150, 120, 90], 'trainId': 2},
{'name': 'Person', 'id': 19, 'color': [220, 20, 60], 'trainId': 11},
{'name': 'Bicyclist', 'id': 20, 'color': [255, 0, 0], 'trainId': 13},
{'name': 'Motorcyclist', 'id': 21, 'color': [255, 0, 100], 'trainId': 14},
{'name': 'Other Rider', 'id': 22, 'color': [255, 0, 200], 'trainId': 12},
{'name': 'Lane Marking - Crosswalk', 'id': 23, 'color': [200, 128, 128], 'trainId': 7},
{'name': 'Lane Marking - General', 'id': 24, 'color': [255, 255, 255], 'trainId': 7},
{'name': 'Mountain', 'id': 25, 'color': [64, 170, 64], 'trainId': 30},
{'name': 'Sand', 'id': 26, 'color': [230, 160, 50], 'trainId': 10},
{'name': 'Sky', 'id': 27, 'color': [70, 130, 180], 'trainId': 26},
{'name': 'Snow', 'id': 28, 'color': [190, 255, 255], 'trainId': 8},
{'name': 'Terrain', 'id': 29, 'color': [152, 251, 152], 'trainId': 10},
{'name': 'Vegetation', 'id': 30, 'color': [107, 142, 35], 'trainId': 32},
{'name': 'Water', 'id': 31, 'color': [0, 170, 30], 'trainId': 42},
{'name': 'Banner', 'id': 32, 'color': [255, 255, 128], 'trainId': 31},
{'name': 'Bench', 'id': 33, 'color': [250, 0, 30], 'trainId': 23},
{'name': 'Bike Rack', 'id': 34, 'color': [100, 140, 180], 'trainId': 24},
{'name': 'Billboard', 'id': 35, 'color': [220, 220, 220], 'trainId': 25},
{'name': 'Catch Basin', 'id': 36, 'color': [220, 128, 128], 'trainId': 255},
{'name': 'CCTV Camera', 'id': 37, 'color': [222, 40, 40], 'trainId': 18},
{'name': 'Fire Hydrant', 'id': 38, 'color': [100, 170, 30], 'trainId': 22},
{'name': 'Junction Box', 'id': 39, 'color': [40, 40, 40], 'trainId': 19},
{'name': 'Mailbox', 'id': 40, 'color': [33, 33, 33], 'trainId': 255},
{'name': 'Manhole', 'id': 41, 'color': [100, 128, 160], 'trainId': 255},
{'name': 'Phone Booth', 'id': 42, 'color': [142, 0, 0], 'trainId': 4},
{'name': 'Pothole', 'id': 43, 'color': [70, 100, 150], 'trainId': 7},
{'name': 'Street Light', 'id': 44, 'color': [210, 170, 100], 'trainId': 15},
{'name': 'Pole', 'id': 45, 'color': [153, 153, 153], 'trainId': 27},
{'name': 'Traffic Sign Frame', 'id': 46, 'color': [128, 128, 128], 'trainId': 20},
{'name': 'Utility Pole', 'id': 47, 'color': [0, 0, 80], 'trainId': 27},
{'name': 'Traffic Light', 'id': 48, 'color': [250, 170, 30], 'trainId': 21},
{'name': 'Traffic Sign (Back)', 'id': 49, 'color': [192, 192, 192], 'trainId': 20},
{'name': 'Traffic Sign (Front)', 'id': 50, 'color': [220, 220, 0], 'trainId': 20},
{'name': 'Trash Can', 'id': 51, 'color': [140, 140, 20], 'trainId': 5},
{'name': 'Bicycle', 'id': 52, 'color': [119, 11, 32], 'trainId': 33},
{'name': 'Boat', 'id': 53, 'color': [150, 0, 255], 'trainId': 40},
{'name': 'Bus', 'id': 54, 'color': [0, 60, 100], 'trainId': 36},
{'name': 'Car', 'id': 55, 'color': [0, 0, 142], 'trainId': 34},
{'name': 'Caravan', 'id': 56, 'color': [0, 0, 90], 'trainId': 34},
{'name': 'Motorcycle', 'id': 57, 'color': [0, 0, 230], 'trainId': 35},
{'name': 'On Rails', 'id': 58, 'color': [0, 80, 100], 'trainId': 37},
{'name': 'Other Vehicle', 'id': 59, 'color': [128, 64, 64], 'trainId': 255},
{'name': 'Trailer', 'id': 60, 'color': [0, 0, 110], 'trainId': 39},
{'name': 'Truck', 'id': 61, 'color': [0, 0, 70], 'trainId': 38},
{'name': 'Wheeled Slow', 'id': 62, 'color': [0, 0, 192], 'trainId': 41},
{'name': 'Car Mount', 'id': 63, 'color': [32, 32, 32], 'trainId': 255},
{'name': 'Ego Vehicle', 'id': 64, 'color': [120, 10, 10], 'trainId': 255},
{'name': 'Unlabeled', 'id': 65, 'color': [0, 0, 0], 'trainId': 255}]

num = 43
def register_mapi_mseg():
    
    # meta = _get_mapi20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in Mseg_label_info:
        lb_map[el['id']] = el['trainId']
    train_to_mseg_map = {}
    for train, mseg in zip(labels_info, Mseg_label_info):
        train_to_mseg_map[train['trainId']] = mseg['trainId']

    for n, anp in [("train", "training"), ("train_1", "training_1"), ("train_2", "training_2"), ("val", "validation")]:
        name = f"mapi_mseg_sem_seg_{n}"
        annpath = f'auto_uni_seg/datasets/mapi/{anp}.txt'
        DatasetCatalog.register(
            name, lambda x=annpath : mapi_train(x)
        )
        
        MetadataCatalog.get(name).set(
            stuff_classes=["bird", "dog", "tunnel", "bridge", "building", "trash_can", "railroad", "road", "snow", "sidewalk_pavement", "terrain", "person", "rider_other", "bicyclist", "motorcyclist", "streetlight", "road_barrier", "wall", "cctv_camera", "junction_box", "traffic_sign", "traffic_light", "fire_hydrant", "bench", "bike_rack", "billboard", "sky", "pole", "fence", "guard_rail", "mountain_hill", "banner", "vegetation", "bicycle", "car", "motorcycle", "bus", "train", "truck", "trailer", "boat_ship", "slow_wheeled_object", "water_other"],
            stuff_dataset_id_to_contiguous_id=lb_map,
            thing_dataset_id_to_contiguous_id=lb_map,
            stuff_colors=[[251, 225,  86],[ 50, 105,  96],      [ 66,  69, 176],      [145, 208,  42],      [180, 241, 180],      [212, 221, 176],      [174, 203, 111],      [230, 246,  34],      [  0,  81, 175],      [113, 216, 102],      [109, 220, 239],      [178, 109,  61],      [  9,  60, 225],      [ 97, 187, 159],      [195, 172, 234],      [236,   2, 131],      [ 76,  66, 237],      [154, 172, 132],      [ 18,  58,  98],      [198, 221, 182],      [199, 215,  29],      [ 64, 129, 196],      [ 24, 192, 253],      [232, 182, 126],      [226,  27,  43],      [252, 168, 166],      [135, 161, 131],      [225, 115,  77],      [148, 126,  90],      [155, 117, 111],      [128, 109,  69],      [ 79,  24, 196],      [ 30, 123,  47],      [ 38, 181, 228],      [141,  17,  60],      [215,  48,  75],      [236, 223, 130],      [208, 211,  35],      [ 73, 208,  97],      [ 58,  34,   9],      [ 34, 168, 123],      [177, 158, 120],      [100,  32,  22],      [ 26, 184,  83],      [ 70,  94, 120],      [186, 140, 216],      [126, 101, 165],      [153,  29, 134],      [169, 102, 174],      [218, 118, 115],      [ 30,  47, 184],      [103,  81, 234],      [106,  48, 148],      [ 79, 127, 244],      [ 14, 159,  57],      [109,  64,  73],      [ 12,  50, 106],      [106, 248, 185],      [125, 123, 180],      [ 80, 108,  57],      [ 81, 255, 160],      [230, 175, 199],      [131, 218,  87],      [214,  92,  19]],
            evaluator_type="sem_seg",
            ignore_label=255,  
            trainId_to_msegId=train_to_mseg_map
        )

register_mapi_mseg()
