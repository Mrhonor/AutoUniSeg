# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.data.datasets.builtin_meta import CITYSCAPES_CATEGORIES
from detectron2.utils.file_io import PathManager


_PREDEFINED_SPLITS_CITYSCAPES_PANOPTIC = {
    "cityscapes_train_panoptic": (
        # This is the original panoptic annotation directory
        "cityscapes/panoptic_train2017",
        "cityscapes/annotations/panoptic_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "cityscapes/panoptic_semseg_train2017",
    ),
    "cityscapes_val_panoptic": (
        "cityscapes/panoptic_val2017",
        "cityscapes/annotations/panoptic_val2017.json",
        "cityscapes/panoptic_semseg_val2017",
    ),
}

CITYSCAPES_PANOPTIC_DIR = {
    "cityscapes_fine_panoptic_train_mds": {
        # This is the original panoptic annotation directory
        "im_dir": "/home1/marong/datasets/cityscapes/leftImg8bit/train",
        "json_dir": "/home1/marong/datasets/cityscapes/gtFine/cityscapes_panoptic_train.json",
        "lb_dir": "/home1/marong/datasets/cityscapes/gtFine/cityscapes_panoptic_train",
    },
    "cityscapes_fine_panoptic_val_mds": {
        "im_dir": "/home1/marong/datasets/cityscapes/leftImg8bit/val",
        "json_dir": "/home1/marong/datasets/cityscapes/gtFine/cityscapes_panoptic_val.json",
        "lb_dir": "/home1/marong/datasets/cityscapes/gtFine/cityscapes_panoptic_val",
    },
}


def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in CITYSCAPES_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in CITYSCAPES_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in CITYSCAPES_CATEGORIES]
    stuff_colors = [k["color"] for k in CITYSCAPES_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors
    # print(stuff_colors)
    

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(CITYSCAPES_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def load_cityscapes_panoptic_json(json_file, image_dir, gt_dir, semseg_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/cityscapes/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = int(ann["image_id"])
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for CITYSCAPES. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other CITYSCAPES-like datasets.
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(gt_dir, ann["file_name"])
        sem_label_file = os.path.join(semseg_dir, ann["file_name"])
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "sem_seg_file_name": sem_label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    return ret

def load_cityscapes_panoptic_json_2(json_file, image_dir, gt_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/cityscapes/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        file_root = ann["file_name"].split("_")[0]
        image_id = ann["image_id"]
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for CITYSCAPES. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other CITYSCAPES-like datasets.
        image_file = os.path.join(image_dir, file_root, image_id+'_leftImg8bit.png')
        label_file = os.path.join(gt_dir, ann["file_name"])
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    # assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    return ret

def register_cityscapes_panoptic_annos_sem_seg(
    name, metadata, image_root, panoptic_root, panoptic_json, sem_seg_root, instances_json
):
    panoptic_name = name
    delattr(MetadataCatalog.get(panoptic_name), "thing_classes")
    delattr(MetadataCatalog.get(panoptic_name), "thing_colors")
    MetadataCatalog.get(panoptic_name).set(
        thing_classes=metadata["thing_classes"],
        thing_colors=metadata["thing_colors"],
        # thing_dataset_id_to_contiguous_id=metadata["thing_dataset_id_to_contiguous_id"],
    )

    # the name is "cityscapes_2017_train_panoptic_with_sem_seg" and "cityscapes_2017_val_panoptic_with_sem_seg"
    semantic_name = name + "_with_sem_seg"
    DatasetCatalog.register(
        semantic_name,
        lambda: load_cityscapes_panoptic_json(panoptic_json, image_root, panoptic_root, sem_seg_root, metadata),
    )
    MetadataCatalog.get(semantic_name).set(
        sem_seg_root=sem_seg_root,
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )

def register_cityscapes_panoptic_annos_sem_seg_2(
    name, metadata, image_root, panoptic_root, panoptic_json
):
    # panoptic_name = name
    # delattr(MetadataCatalog.get(panoptic_name), "thing_classes")
    # delattr(MetadataCatalog.get(panoptic_name), "thing_colors")
    # MetadataCatalog.get(panoptic_name).set(
    #     thing_classes=metadata["thing_classes"],
    #     thing_colors=metadata["thing_colors"],
    #     thing_dataset_id_to_contiguous_id=metadata["thing_dataset_id_to_contiguous_id"],
    # )

    # the name is "cityscapes_2017_train_panoptic_with_sem_seg" and "cityscapes_2017_val_panoptic_with_sem_seg"
    semantic_name = name #+ "_with_sem_seg"
    DatasetCatalog.register(
        semantic_name,
        lambda: load_cityscapes_panoptic_json_2(panoptic_json, image_root, panoptic_root, metadata),
    )
    MetadataCatalog.get(semantic_name).set(
        # sem_seg_root=sem_seg_root,
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        # json_file=instances_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )

def register_all_cityscapes_panoptic_annos_sem_seg_2(root):
    for (
        dataset_name,
        dir_infos,
    ) in CITYSCAPES_PANOPTIC_DIR.items():
        # prefix_instances = prefix[: -len("_panoptic")]
        # instances_meta = MetadataCatalog.get(prefix_instances)
        # image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        image_root = dir_infos["im_dir"]
        panoptic_root, panoptic_json = dir_infos["lb_dir"], dir_infos["json_dir"]

        register_cityscapes_panoptic_annos_sem_seg_2(
            dataset_name,
            get_metadata(),
            image_root,
            panoptic_root,
            panoptic_json,
        )

def register_all_cityscapes_panoptic_annos_sem_seg(root):
    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_CITYSCAPES_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file

        register_cityscapes_panoptic_annos_sem_seg(
            prefix,
            get_metadata(),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_cityscapes_panoptic_annos_sem_seg_2(_root)