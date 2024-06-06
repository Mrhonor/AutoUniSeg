import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import os.path as osp

labels_info = [
    {"name": "Sky", "id": 0, "color": [128, 128, 128], "trainId": 0},
    {"name": "Bridge", "id": 1, "color": [0, 128, 64], "trainId": 1},
    {"name": "Building", "id": 2, "color": [128, 0, 0], "trainId": 1},
    {"name": "Wall", "id": 3, "color": [64, 192, 0], "trainId": 1},
    {"name": "Tunnel", "id": 4, "color": [64, 0, 64], "trainId": 1},
    {"name": "Archway", "id": 5, "color": [192, 0, 128], "trainId": 1},
    {"name": "Column_Pole", "id": 6, "color": [192, 192, 128], "trainId": 2},
    {"name": "TrafficCone", "id": 7, "color": [0, 0, 64], "trainId": 2},
    {"name": "Road", "id": 8, "color": [128, 64, 128], "trainId": 3},
    {"name": "LaneMkgsDriv", "id": 9, "color": [128, 0, 192], "trainId": 3},
    {"name": "LaneMkgsNonDriv", "id": 10, "color": [192, 0, 64], "trainId": 3},
    {"name": "Sidewalk", "id": 11, "color": [0, 0, 192], "trainId": 4},
    {"name": "ParkingBlock", "id": 12, "color": [64, 192, 128], "trainId": 4},
    {"name": "RoadShoulder", "id": 13, "color": [128, 128, 192], "trainId": 4},
    {"name": "Tree", "id": 14, "color": [128, 128, 0], "trainId": 5},
    {"name": "VegetationMisc", "id": 15, "color": [192, 192, 0], "trainId": 5},
    {"name": "SignSymbol", "id": 16, "color": [192, 128, 128], "trainId": 6},
    {"name": "Misc_Text", "id": 17, "color": [128, 128, 64], "trainId": 6},
    {"name": "TrafficLight", "id": 18, "color": [0, 64, 64], "trainId": 6},
    {"name": "Fence", "id": 19, "color": [64, 64, 128], "trainId": 7},
    {"name": "Car", "id": 20, "color": [64, 0, 128], "trainId": 8},
    {"name": "SUVPickupTruck", "id": 21, "color": [64, 128, 192], "trainId": 8},
    {"name": "Truck_Bus", "id": 22, "color": [192, 128, 192], "trainId": 8},
    {"name": "Train", "id": 23, "color": [192, 64, 128], "trainId": 8},
    {"name": "OtherMoving", "id": 24, "color": [128, 64, 64], "trainId": 8},
    {"name": "Pedestrian", "id": 25, "color": [64, 64, 0], "trainId":9},
    {"name": "Child", "id": 26, "color": [192, 128, 64], "trainId":9},
    {"name": "CartLuggagePram", "id": 27, "color": [64, 0, 192], "trainId": 9},
    {"name": "Animal", "id": 28, "color": [64, 128, 64], "trainId": 9},
    {"name": "Bicyclist", "id": 29, "color": [0, 128, 192], "trainId": 10},
    {"name": "MotorcycleScooter", "id": 30, "color": [192, 0, 192], "trainId": 10},
    {"name": "Void", "id": 31, "color": [0, 0, 0], "trainId": 255}
]

dataroot = 'datasets/CamVid'
annpath = f'auto_uni_seg/datasets/CamVid/test.txt'
def camvid():
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


def register_camvid():
    
    
    # meta = _get_ade20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in labels_info:
        lb_map[el['id']] = el['trainId']

    name = f"camvid_sem_seg_test"
    DatasetCatalog.register(
        name, camvid
    )
    
    MetadataCatalog.get(name).set(
        stuff_classes=["Sky", "Building", "Column_Pole", "Road", "Sidewalk", "Tree", "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist"],
        stuff_dataset_id_to_contiguous_id=lb_map,
        thing_dataset_id_to_contiguous_id=lb_map,
        stuff_colors=[[225, 115,  77], [154, 172, 132], [ 70,  94, 120],[ 97, 187, 159],[236,   2, 131], [128, 109,  69],[218, 118, 115],[145, 208,  42],[ 14, 159,  57],[198, 221, 182], [ 24, 192, 253]],
        evaluator_type="sem_seg",
        ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
    )


# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_camvid()

train_annpath = f'auto_uni_seg/datasets/CamVid/train.txt'
def camvid_train(anp):

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


def register_camvid_train():
    
    
    # meta = _get_cs20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in labels_info:
        lb_map[el['id']] = el['trainId']
    for n, anp in [("train", "train"),]:
        name = f"camvid_sem_seg_{n}"
        annpath = f'auto_uni_seg/datasets/CamVid/{anp}.txt'
        DatasetCatalog.register(
            name, lambda x=annpath : camvid_train(x)
        )
        
        MetadataCatalog.get(name).set(
            stuff_classes=["Sky", "Building", "Column_Pole", "Road", "Sidewalk", "Tree", "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist"],
            stuff_dataset_id_to_contiguous_id=lb_map,
            thing_dataset_id_to_contiguous_id=lb_map,
            stuff_colors=[[225, 115,  77], [154, 172, 132], [ 70,  94, 120],[ 97, 187, 159],[236,   2, 131], [128, 109,  69],[218, 118, 115],[145, 208,  42],[ 14, 159,  57],[198, 221, 182], [ 24, 192, 253]],
            evaluator_type="sem_seg",
            ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
        )

register_camvid_train()

