import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import os.path as osp

dataroot = 'datasets/ScanNet/data/tasks/scannet-41/scannet_frames_25k'
annpath = f'auto_uni_seg/datasets/scannet/val.txt'
def scannet():
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


def register_scannet():
    
    
    # meta = _get_ade20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in range(20):
        lb_map[el] = el

    name = f"scannet_sem_seg_val"
    DatasetCatalog.register(
        name, scannet
    )
    
    MetadataCatalog.get(name).set(
        stuff_classes=["bathtub","bed","bookshelf","cabinet","chair","counter","curtain","desk","door","floor","otherfurniture","picture","refridgerator","shower curtain","sink","sofa","table","toilet","wall","window"],
        stuff_dataset_id_to_contiguous_id=lb_map,
        thing_dataset_id_to_contiguous_id=lb_map,
        evaluator_type="sem_seg",
        ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
    )


# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_scannet()

train_annpath = f'auto_uni_seg/datasets/scannet/train.txt'
def scannet_train(anp):

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


def register_scannet_train():
    
    
    # meta = _get_cs20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in range(20):
        lb_map[el] = el
    for n, anp in [("train", "train"),]:
        name = f"scannet_sem_seg_{n}"
        annpath = f'auto_uni_seg/datasets/scannet/{anp}.txt'
        DatasetCatalog.register(
            name, lambda x=annpath : scannet_train(x)
        )
        
        MetadataCatalog.get(name).set(
            stuff_classes=["bathtub","bed","bookshelf","cabinet","chair","counter","curtain","desk","door","floor","otherfurniture","picture","refridgerator","shower curtain","sink","sofa","table","toilet","wall","window"],
            stuff_dataset_id_to_contiguous_id=lb_map,
            thing_dataset_id_to_contiguous_id=lb_map,
            evaluator_type="sem_seg",
            ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
        )

register_scannet_train()

