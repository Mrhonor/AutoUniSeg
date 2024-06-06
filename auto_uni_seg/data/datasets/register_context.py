import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import os.path as osp

dataroot = 'datasets/context'
annpath = f'auto_uni_seg/datasets/context/val.txt'
def context():
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


def register_context():
    
    
    # meta = _get_ade20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in range(60):
        lb_map[el] = el

    name = f"context_sem_seg_val"
    DatasetCatalog.register(
        name, context
    )
    
    MetadataCatalog.get(name).set(
        stuff_classes=["aeroplane","background","bag","bed","bedclothes","bench","bicycle","bird","boat","book","bottle","building","bus","cabinet","car","cat","ceiling","chair","cloth","computer","cow","cup","curtain","dog","door","fence","floor","flower","food","grass","ground","horse","keyboard","light","motorbike","mountain","mouse","person","plate","platform","pottedplant","road","rock","sheep","shelves","sidewalk","sign","sky","snow","sofa","table","track","train","tree","truck","tvmonitor","wall","water","window","wood"],
        stuff_dataset_id_to_contiguous_id=lb_map,
        thing_dataset_id_to_contiguous_id=lb_map,
        evaluator_type="sem_seg",
        ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
    )


# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_context()

train_annpath = f'auto_uni_seg/datasets/context/train.txt'
def context_train(anp):

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


def register_context_train():
    
    
    # meta = _get_cs20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in range(60):
        lb_map[el] = el
    for n, anp in [("train", "train"),]:
        name = f"context_sem_seg_{n}"
        annpath = f'auto_uni_seg/datasets/context/{anp}.txt'
        DatasetCatalog.register(
            name, lambda x=annpath : context_train(x)
        )
        
        MetadataCatalog.get(name).set(
            stuff_classes=["aeroplane","background","bag","bed","bedclothes","bench","bicycle","bird","boat","book","bottle","building","bus","cabinet","car","cat","ceiling","chair","cloth","computer","cow","cup","curtain","dog","door","fence","floor","flower","food","grass","ground","horse","keyboard","light","motorbike","mountain","mouse","person","plate","platform","pottedplant","road","rock","sheep","shelves","sidewalk","sign","sky","snow","sofa","table","track","train","tree","truck","tvmonitor","wall","water","window","wood"],
            stuff_dataset_id_to_contiguous_id=lb_map,
            thing_dataset_id_to_contiguous_id=lb_map,
            evaluator_type="sem_seg",
            ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
        )

register_context_train()

