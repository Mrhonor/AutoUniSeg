import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import os.path as osp

dataroot = 'datasets/VOC'
annpath = f'auto_uni_seg/datasets/voc/val.txt'
def voc():
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


def register_voc():
    
    
    # meta = _get_ade20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in range(256):
        if el == 0:
            lb_map[0] = 255
        elif el > 20:
            lb_map[el] = 255
        else:
            lb_map[el] = el-1

    name = f"voc_sem_seg_val"
    DatasetCatalog.register(
        name, voc
    )
    
    MetadataCatalog.get(name).set(
        stuff_classes=["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"],
        stuff_dataset_id_to_contiguous_id=lb_map,
        thing_dataset_id_to_contiguous_id=lb_map,
        stuff_colors=[[87, 59, 76], [103, 81, 234], [251, 225,  86],[106, 48, 148],[217, 135, 68],[79, 127, 244],[14, 159, 57],[0,0,0], [38, 181, 228],[76, 21, 14], [7, 162, 5], [76, 21, 14],[76, 21, 14],[12, 50, 106],[198, 221, 182],[206, 106, 97],[76, 21, 14],[75, 211, 41],[254, 205,  75],[61, 206, 227]],
        evaluator_type="sem_seg",
        ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
    )


# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_voc()

train_annpath = f'auto_uni_seg/datasets/voc/train.txt'
def voc_train(anp):

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


def register_voc_train():
    
    
    # meta = _get_cs20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in range(256):
        if el == 0:
            lb_map[0] = 255
        elif el > 20:
            lb_map[el] = 255
        else:
            lb_map[el] = el-1
    for n, anp in [("train", "train"),]:
        name = f"voc_sem_seg_{n}"
        annpath = f'auto_uni_seg/datasets/voc/{anp}.txt'
        DatasetCatalog.register(
            name, lambda x=annpath : voc_train(x)
        )
        
        MetadataCatalog.get(name).set(
            stuff_classes=["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"],
            stuff_dataset_id_to_contiguous_id=lb_map,
            thing_dataset_id_to_contiguous_id=lb_map,
            evaluator_type="sem_seg",
            stuff_colors=[[87, 59, 76], [103, 81, 234], [251, 225,  86],[106, 48, 148],[217, 135, 68],[79, 127, 244],[14, 159, 57],[0,0,0], [38, 181, 228],[76, 21, 14], [7, 162, 5], [76, 21, 14],[76, 21, 14],[12, 50, 106],[198, 221, 182],[206, 106, 97],[76, 21, 14],[75, 211, 41],[254, 205,  75],[61, 206, 227]],
            ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
        )

register_voc_train()

