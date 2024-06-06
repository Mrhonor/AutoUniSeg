import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import os.path as osp

labels_info = [
{"name": "person", "id": 1, "trainId": 0},
{"name": "bicycle", "id": 2, "trainId": 1},
{"name": "car", "id": 3, "trainId": 2},
{"name": "motorcycle", "id": 4, "trainId": 3},
{"name": "airplane", "id": 5, "trainId": 4},
{"name": "bus", "id": 6, "trainId": 5},
{"name": "train", "id": 7, "trainId": 6},
{"name": "truck", "id": 8, "trainId": 7},
{"name": "boat", "id": 9, "trainId": 8},
{"name": "traffic light", "id": 10, "trainId": 9},
{"name": "fire hydrant", "id": 11, "trainId": 10},
{"name": "stop sign", "id": 13, "trainId": 11},
{"name": "parking meter", "id": 14, "trainId": 12},
{"name": "bench", "id": 15, "trainId": 13},
{"name": "bird", "id": 16, "trainId": 14},
{"name": "cat", "id": 17, "trainId": 15},
{"name": "dog", "id": 18, "trainId": 16},
{"name": "horse", "id": 19, "trainId": 17},
{"name": "sheep", "id": 20, "trainId": 18},
{"name": "cow", "id": 21, "trainId": 19},
{"name": "elephant", "id": 22, "trainId": 20},
{"name": "bear", "id": 23, "trainId": 21},
{"name": "zebra", "id": 24, "trainId": 22},
{"name": "giraffe", "id": 25, "trainId": 23},
{"name": "backpack", "id": 27, "trainId": 24},
{"name": "umbrella", "id": 28, "trainId": 25},
{"name": "handbag", "id": 31, "trainId": 26},
{"name": "tie", "id": 32, "trainId": 27},
{"name": "suitcase", "id": 33, "trainId": 28},
{"name": "frisbee", "id": 34, "trainId": 29},
{"name": "skis", "id": 35, "trainId": 30},
{"name": "snowboard", "id": 36, "trainId": 31},
{"name": "sports ball", "id": 37, "trainId": 32},
{"name": "kite", "id": 38, "trainId": 33},
{"name": "baseball bat", "id": 39, "trainId": 34},
{"name": "baseball glove", "id": 40, "trainId": 35},
{"name": "skateboard", "id": 41, "trainId": 36},
{"name": "surfboard", "id": 42, "trainId": 37},
{"name": "tennis racket", "id": 43, "trainId": 38},
{"name": "bottle", "id": 44, "trainId": 39},
{"name": "wine glass", "id": 46, "trainId": 40},
{"name": "cup", "id": 47, "trainId": 41},
{"name": "fork", "id": 48, "trainId": 42},
{"name": "knife", "id": 49, "trainId": 43},
{"name": "spoon", "id": 50, "trainId": 44},
{"name": "bowl", "id": 51, "trainId": 45},
{"name": "banana", "id": 52, "trainId": 46},
{"name": "apple", "id": 53, "trainId": 47},
{"name": "sandwich", "id": 54, "trainId": 48},
{"name": "orange", "id": 55, "trainId": 49},
{"name": "broccoli", "id": 56, "trainId": 50},
{"name": "carrot", "id": 57, "trainId": 51},
{"name": "hot dog", "id": 58, "trainId": 52},
{"name": "pizza", "id": 59, "trainId": 53},
{"name": "donut", "id": 60, "trainId": 54},
{"name": "cake", "id": 61, "trainId": 55},
{"name": "chair", "id": 62, "trainId": 56},
{"name": "couch", "id": 63, "trainId": 57},
{"name": "potted plant", "id": 64, "trainId": 58},
{"name": "bed", "id": 65, "trainId": 59},
{"name": "dining table", "id": 67, "trainId": 60},
{"name": "toilet", "id": 70, "trainId": 61},
{"name": "tv", "id": 72, "trainId": 62},
{"name": "laptop", "id": 73, "trainId": 63},
{"name": "mouse", "id": 74, "trainId": 64},
{"name": "remote", "id": 75, "trainId": 65},
{"name": "keyboard", "id": 76, "trainId": 66},
{"name": "cell phone", "id": 77, "trainId": 67},
{"name": "microwave", "id": 78, "trainId": 68},
{"name": "oven", "id": 79, "trainId": 69},
{"name": "toaster", "id": 80, "trainId": 70},
{"name": "sink", "id": 81, "trainId": 71},
{"name": "refrigerator", "id": 82, "trainId": 72},
{"name": "book", "id": 84, "trainId": 73},
{"name": "clock", "id": 85, "trainId": 74},
{"name": "vase", "id": 86, "trainId": 75},
{"name": "scissors", "id": 87, "trainId": 76},
{"name": "teddy bear", "id": 88, "trainId": 77},
{"name": "hair drier", "id": 89, "trainId": 78},
{"name": "toothbrush", "id": 90, "trainId": 79},
{"name": "banner", "id": 92, "trainId": 80},
{"name": "blanket", "id": 93, "trainId": 81},
{"name": "bridge", "id": 95, "trainId": 82},
{"name": "cardboard", "id": 100, "trainId": 83},
{"name": "counter", "id": 107, "trainId": 84},
{"name": "curtain", "id": 109, "trainId": 85},
{"name": "door-stuff", "id": 112, "trainId": 86},
{"name": "floor-wood", "id": 118, "trainId": 87},
{"name": "flower", "id": 119, "trainId": 88},
{"name": "fruit", "id": 122, "trainId": 89},
{"name": "gravel", "id": 125, "trainId": 90},
{"name": "house", "id": 128, "trainId": 91},
{"name": "light", "id": 130, "trainId": 92},
{"name": "mirror-stuff", "id": 133, "trainId": 93},
{"name": "net", "id": 138, "trainId": 94},
{"name": "pillow", "id": 141, "trainId": 95},
{"name": "platform", "id": 144, "trainId": 96},
{"name": "playingfield", "id": 145, "trainId": 97},
{"name": "railroad", "id": 147, "trainId": 98},
{"name": "river", "id": 148, "trainId": 99},
{"name": "road", "id": 149, "trainId": 100},
{"name": "roof", "id": 151, "trainId": 101},
{"name": "sand", "id": 154, "trainId": 102},
{"name": "sea", "id": 155, "trainId": 103},
{"name": "shelf", "id": 156, "trainId": 104},
{"name": "snow", "id": 159, "trainId": 105},
{"name": "stairs", "id": 161, "trainId": 106},
{"name": "tent", "id": 166, "trainId": 107},
{"name": "towel", "id": 168, "trainId": 108},
{"name": "wall-brick", "id": 171, "trainId": 109},
{"name": "wall-stone", "id": 175, "trainId": 110},
{"name": "wall-tile", "id": 176, "trainId": 111},
{"name": "wall-wood", "id": 177, "trainId": 112},
{"name": "water-other", "id": 178, "trainId": 113},
{"name": "window-blind", "id": 180, "trainId": 114},
{"name": "window-other", "id": 181, "trainId": 115},
{"name": "tree-merged", "id": 184, "trainId": 116},
{"name": "fence-merged", "id": 185, "trainId": 117},
{"name": "ceiling-merged", "id": 186, "trainId": 118},
{"name": "sky-other-merged", "id": 187, "trainId": 119},
{"name": "cabinet-merged", "id": 188, "trainId": 120},
{"name": "table-merged", "id": 189, "trainId": 121},
{"name": "floor-other-merged", "id": 190, "trainId": 122},
{"name": "pavement-merged", "id": 191, "trainId": 123},
{"name": "mountain-merged", "id": 192, "trainId": 124},
{"name": "grass-merged", "id": 193, "trainId": 125},
{"name": "dirt-merged", "id": 194, "trainId": 126},
{"name": "paper-merged", "id": 195, "trainId": 127},
{"name": "food-other-merged", "id": 196, "trainId": 128},
{"name": "building-other-merged", "id": 197, "trainId": 129},
{"name": "rock-merged", "id": 198, "trainId": 130},
{"name": "wall-other-merged", "id": 199, "trainId": 131},
{"name": "rug-merged", "id": 200, "trainId": 132},
{"name": "unlabeled", "id":0, "trainId": 255},
]
dataroot = 'datasets/coco'
annpath = f'auto_uni_seg/datasets/coco/val.txt'
def coco():
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


def register_coco():
    
    
    # meta = _get_ade20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in labels_info:
        lb_map[el['id']] = el['trainId']

    name = f"coco_sem_seg_val"
    DatasetCatalog.register(
        name, coco
    )
    
    MetadataCatalog.get(name).set(
        stuff_classes=["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "banner", "blanket", "bridge", "cardboard", "counter", "curtain", "door-stuff", "floor-wood", "flower", "fruit", "gravel", "house", "light", "mirror-stuff", "net", "pillow", "platform", "playingfield", "railroad", "river", "road", "roof", "sand", "sea", "shelf", "snow", "stairs", "tent", "towel", "wall-brick", "wall-stone", "wall-tile", "wall-wood", "water", "window-blind", "window", "tree", "fence", "ceiling", "sky", "cabinet", "table", "floor", "pavement", "mountain", "grass", "dirt", "paper", "food", "building", "rock", "wall", "rug"],
        # stuff_classes=["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "banner", "blanket", "bridge", "cardboard", "counter", "curtain", "door-stuff", "floor-wood", "flower", "fruit", "gravel", "house", "light", "mirror-stuff", "net", "pillow", "platform", "playingfield", "railroad", "river", "road", "roof", "sand", "sea", "shelf", "snow", "stairs", "tent", "towel", "wall-brick", "wall-stone", "wall-tile", "wall-wood", "water-other", "window-blind", "window-other", "tree-merged", "fence-merged", "ceiling-merged", "sky-other-merged", "cabinet-merged", "table-merged", "floor-other-merged", "pavement-merged", "mountain-merged", "grass-merged", "dirt-merged", "paper-merged", "food-other-merged", "building-other-merged", "rock-merged", "wall-other-merged", "rug-merged"],
        stuff_dataset_id_to_contiguous_id=lb_map,
        thing_dataset_id_to_contiguous_id=lb_map,
        stuff_colors=[[198, 221, 182],      [103,  81, 234],      [ 14, 159,  57],      [ 12,  50, 106],      [ 87,  59,  76],      [ 79, 127, 244],      [254, 205,  75],      [ 81, 255, 160],      [106,  48, 148],      [153,  29, 134],      [ 73, 208,  97],      [218, 118, 115],      [157, 128,  24],      [ 47, 118, 136],      [189,  47,  88],      [133, 110, 199],      [ 50, 105,  96],      [164, 226,  74],      [ 79, 252,  40],      [ 99,  99, 147],      [160, 116,  78],      [ 71, 235,  99],      [208,  49, 185],      [193, 200, 137],      [108, 156,  92],      [254,  47,  11],      [164, 121, 219],      [113, 216, 102],      [ 25, 111,  21],      [213, 208, 155],      [  0,  81, 175],      [ 33,   4, 237],      [ 31, 148,  91],      [201,  31, 122],      [ 16, 176,  44],      [152, 133, 134],      [ 84,  79, 104],      [121, 137, 142],      [109,  36, 216],      [217, 135,  68],      [158,  92,  94],      [ 99,  16, 132],      [208, 211,  35],      [129,  86, 148],      [177,  40,  46],      [ 16, 226,  71],      [ 15, 161,  47],      [ 39,  59, 205],      [138, 195,  40],      [ 57,  14, 178],      [ 30, 123,  47],      [153, 109, 201],      [ 27,  45, 173],      [154,  81, 177],      [205, 251, 130],      [  0, 247,  36],      [ 38, 181, 228],      [ 75, 211,  41],      [206, 106,  97],      [ 10,   7, 111],      [ 91, 182,  84],      [109, 155, 209],      [ 61, 206, 227],      [167, 118,  46],      [177, 163, 165],      [230, 232, 157],      [ 15, 236, 141],      [249, 125, 105],      [184, 102,  69],      [109, 248,  25],      [129, 125, 231],      [236, 223, 130],      [177, 158, 120],      [240,  79,  15],      [115, 177,  95],      [111, 173,  82],      [ 55, 211,  67],      [146,  99,  68],      [131, 218,  87],      [ 99, 244,  92],      [215,  48,  75],      [158, 123, 213],      [ 76,  66, 237],      [169, 102, 174],      [138,  37,  98],      [113,  39, 205],      [ 77, 110,  58],      [236, 135, 127],      [218,  37, 136],      [251, 225,  86],      [100,  32,  22],      [189, 161,  55],      [104, 162, 189],      [146, 240, 187],      [231, 117, 185],      [ 54,   3, 109],      [223, 218,  93],      [112,   2, 104],      [  9,  60, 225],      [136,  66,  69],      [ 97, 187, 159],      [175, 119, 215],      [135, 161, 131],      [199, 183,  46],      [126,  22, 239],      [148, 126,  90],      [180, 241, 180],      [189, 124,   9],      [235,  91,  37],      [176,  92, 116],      [234,  44,  84],      [230, 197,  29],      [252,   9, 152],      [ 79,  24, 196],      [177,  96,  58],      [ 35, 133,  44],      [ 68,  68, 237],      [145, 208,  42],      [ 19, 228,  89],      [225, 115,  77],      [233, 198,  64],      [  7, 162,   5],      [ 23,  69, 195],      [182,  60, 104],      [252, 168, 166],      [188,  49,  31],      [ 39, 192,  19],      [230, 101, 208],      [180,  12, 242],      [154, 172, 132],      [126,  91,  65],      [174, 203, 111], [195, 177, 180]],
        evaluator_type="sem_seg",
        ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
    )


# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_coco()

train_annpath = f'auto_uni_seg/datasets/coco/train.txt'
def coco_train(anp):

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


def register_coco_train():
    
    
    # meta = _get_cs20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in labels_info:
        lb_map[el['id']] = el['trainId']
    for n, anp in [("train", "train"), ("train_1", "train_1"), ("train_2", "train_2"), ("val_temp", "val_temp")]:
        name = f"coco_sem_seg_{n}"
        annpath = f'auto_uni_seg/datasets/coco/{anp}.txt'
        DatasetCatalog.register(
            name, lambda x=annpath : coco_train(x)
        )
            
        MetadataCatalog.get(name).set(
            stuff_classes=["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "banner", "blanket", "bridge", "cardboard", "counter", "curtain", "door-stuff", "floor-wood", "flower", "fruit", "gravel", "house", "light", "mirror-stuff", "net", "pillow", "platform", "playingfield", "railroad", "river", "road", "roof", "sand", "sea", "shelf", "snow", "stairs", "tent", "towel", "wall-brick", "wall-stone", "wall-tile", "wall-wood", "water", "window-blind", "window", "tree", "fence", "ceiling", "sky", "cabinet", "table", "floor", "pavement", "mountain", "grass", "dirt", "paper", "food", "building", "rock", "wall", "rug"],
            # stuff_classes=["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "banner", "blanket", "bridge", "cardboard", "counter", "curtain", "door-stuff", "floor-wood", "flower", "fruit", "gravel", "house", "light", "mirror-stuff", "net", "pillow", "platform", "playingfield", "railroad", "river", "road", "roof", "sand", "sea", "shelf", "snow", "stairs", "tent", "towel", "wall-brick", "wall-stone", "wall-tile", "wall-wood", "water-other", "window-blind", "window-other", "tree-merged", "fence-merged", "ceiling-merged", "sky-other-merged", "cabinet-merged", "table-merged", "floor-other-merged", "pavement-merged", "mountain-merged", "grass-merged", "dirt-merged", "paper-merged", "food-other-merged", "building-other-merged", "rock-merged", "wall-other-merged", "rug-merged"],
            stuff_dataset_id_to_contiguous_id=lb_map,
            thing_dataset_id_to_contiguous_id=lb_map,
            stuff_colors=[[198, 221, 182],      [103,  81, 234],      [ 14, 159,  57],      [ 12,  50, 106],      [ 87,  59,  76],      [ 79, 127, 244],      [254, 205,  75],      [ 81, 255, 160],      [106,  48, 148],      [153,  29, 134],      [ 73, 208,  97],      [218, 118, 115],      [157, 128,  24],      [ 47, 118, 136],      [189,  47,  88],      [133, 110, 199],      [ 50, 105,  96],      [164, 226,  74],      [ 79, 252,  40],      [ 99,  99, 147],      [160, 116,  78],      [ 71, 235,  99],      [208,  49, 185],      [193, 200, 137],      [108, 156,  92],      [254,  47,  11],      [164, 121, 219],      [113, 216, 102],      [ 25, 111,  21],      [213, 208, 155],      [  0,  81, 175],      [ 33,   4, 237],      [ 31, 148,  91],      [201,  31, 122],      [ 16, 176,  44],      [152, 133, 134],      [ 84,  79, 104],      [121, 137, 142],      [109,  36, 216],      [217, 135,  68],      [158,  92,  94],      [ 99,  16, 132],      [208, 211,  35],      [129,  86, 148],      [177,  40,  46],      [ 16, 226,  71],      [ 15, 161,  47],      [ 39,  59, 205],      [138, 195,  40],      [ 57,  14, 178],      [ 30, 123,  47],      [153, 109, 201],      [ 27,  45, 173],      [154,  81, 177],      [205, 251, 130],      [  0, 247,  36],      [ 38, 181, 228],      [ 75, 211,  41],      [206, 106,  97],      [ 10,   7, 111],      [ 91, 182,  84],      [109, 155, 209],      [ 61, 206, 227],      [167, 118,  46],      [177, 163, 165],      [230, 232, 157],      [ 15, 236, 141],      [249, 125, 105],      [184, 102,  69],      [109, 248,  25],      [129, 125, 231],      [236, 223, 130],      [177, 158, 120],      [240,  79,  15],      [115, 177,  95],      [111, 173,  82],      [ 55, 211,  67],      [146,  99,  68],      [131, 218,  87],      [ 99, 244,  92],      [215,  48,  75],      [158, 123, 213],      [ 76,  66, 237],      [169, 102, 174],      [138,  37,  98],      [113,  39, 205],      [ 77, 110,  58],      [236, 135, 127],      [218,  37, 136],      [251, 225,  86],      [100,  32,  22],      [189, 161,  55],      [104, 162, 189],      [146, 240, 187],      [231, 117, 185],      [ 54,   3, 109],      [223, 218,  93],      [112,   2, 104],      [  9,  60, 225],      [136,  66,  69],      [ 97, 187, 159],      [175, 119, 215],      [135, 161, 131],      [199, 183,  46],      [126,  22, 239],      [148, 126,  90],      [180, 241, 180],      [189, 124,   9],      [235,  91,  37],      [176,  92, 116],      [234,  44,  84],      [230, 197,  29],      [252,   9, 152],      [ 79,  24, 196],      [177,  96,  58],      [ 35, 133,  44],      [ 68,  68, 237],      [145, 208,  42],      [ 19, 228,  89],      [225, 115,  77],      [233, 198,  64],      [  7, 162,   5],      [ 23,  69, 195],      [182,  60, 104],      [252, 168, 166],      [188,  49,  31],      [ 39, 192,  19],      [230, 101, 208],      [180,  12, 242],      [154, 172, 132],      [126,  91,  65],      [174, 203, 111], [195, 177, 180]],
            evaluator_type="sem_seg",
            ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
        )


register_coco_train()

Mseg_label_info = [{'name': 'person', 'id': 1, 'trainId': 79},
{'name': 'bicycle', 'id': 2, 'trainId': 108},
{'name': 'car', 'id': 3, 'trainId': 109},
{'name': 'motorcycle', 'id': 4, 'trainId': 110},
{'name': 'airplane', 'id': 5, 'trainId': 111},
{'name': 'bus', 'id': 6, 'trainId': 112},
{'name': 'train', 'id': 7, 'trainId': 113},
{'name': 'truck', 'id': 8, 'trainId': 114},
{'name': 'boat', 'id': 9, 'trainId': 115},
{'name': 'traffic light', 'id': 10, 'trainId': 82},
{'name': 'fire hydrant', 'id': 11, 'trainId': 83},
{'name': 'stop sign', 'id': 13, 'trainId': 81},
{'name': 'parking meter', 'id': 14, 'trainId': 84},
{'name': 'bench', 'id': 15, 'trainId': 85},
{'name': 'bird', 'id': 16, 'trainId': 5},
{'name': 'cat', 'id': 17, 'trainId': 6},
{'name': 'dog', 'id': 18, 'trainId': 7},
{'name': 'horse', 'id': 19, 'trainId': 8},
{'name': 'sheep', 'id': 20, 'trainId': 9},
{'name': 'cow', 'id': 21, 'trainId': 10},
{'name': 'elephant', 'id': 22, 'trainId': 11},
{'name': 'bear', 'id': 23, 'trainId': 12},
{'name': 'zebra', 'id': 24, 'trainId': 13},
{'name': 'giraffe', 'id': 25, 'trainId': 14},
{'name': 'backpack', 'id': 27, 'trainId': 0},
{'name': 'umbrella', 'id': 28, 'trainId': 1},
{'name': 'handbag', 'id': 31, 'trainId': 2},
{'name': 'tie', 'id': 32, 'trainId': 3},
{'name': 'suitcase', 'id': 33, 'trainId': 4},
{'name': 'frisbee', 'id': 34, 'trainId': 90},
{'name': 'skis', 'id': 35, 'trainId': 91},
{'name': 'snowboard', 'id': 36, 'trainId': 92},
{'name': 'sports ball', 'id': 37, 'trainId': 93},
{'name': 'kite', 'id': 38, 'trainId': 94},
{'name': 'baseball bat', 'id': 39, 'trainId': 95},
{'name': 'baseball glove', 'id': 40, 'trainId': 96},
{'name': 'skateboard', 'id': 41, 'trainId': 97},
{'name': 'surfboard', 'id': 42, 'trainId': 98},
{'name': 'tennis racket', 'id': 43, 'trainId': 99},
{'name': 'bottle', 'id': 44, 'trainId': 72},
{'name': 'wine glass', 'id': 46, 'trainId': 74},
{'name': 'cup', 'id': 47, 'trainId': 73},
{'name': 'fork', 'id': 48, 'trainId': 76},
{'name': 'knife', 'id': 49, 'trainId': 75},
{'name': 'spoon', 'id': 50, 'trainId': 77},
{'name': 'bowl', 'id': 51, 'trainId': 78},
{'name': 'banana', 'id': 52, 'trainId': 32},
{'name': 'apple', 'id': 53, 'trainId': 33},
{'name': 'sandwich', 'id': 54, 'trainId': 34},
{'name': 'orange', 'id': 55, 'trainId': 35},
{'name': 'broccoli', 'id': 56, 'trainId': 36},
{'name': 'carrot', 'id': 57, 'trainId': 37},
{'name': 'hot dog', 'id': 58, 'trainId': 38},
{'name': 'pizza', 'id': 59, 'trainId': 39},
{'name': 'donut', 'id': 60, 'trainId': 40},
{'name': 'cake', 'id': 61, 'trainId': 41},
{'name': 'chair', 'id': 62, 'trainId': 44},
{'name': 'couch', 'id': 63, 'trainId': 45},
{'name': 'potted plant', 'id': 64, 'trainId': 46},
{'name': 'bed', 'id': 65, 'trainId': 47},
{'name': 'dining table', 'id': 67, 'trainId': 48},
{'name': 'toilet', 'id': 70, 'trainId': 20},
{'name': 'tv', 'id': 72, 'trainId': 30},
{'name': 'laptop', 'id': 73, 'trainId': 25},
{'name': 'mouse', 'id': 74, 'trainId': 27},
{'name': 'remote', 'id': 75, 'trainId': 28},
{'name': 'keyboard', 'id': 76, 'trainId': 26},
{'name': 'cell phone', 'id': 77, 'trainId': 29},
{'name': 'microwave', 'id': 78, 'trainId': 15},
{'name': 'oven', 'id': 79, 'trainId': 16},
{'name': 'toaster', 'id': 80, 'trainId': 17},
{'name': 'sink', 'id': 81, 'trainId': 18},
{'name': 'refrigerator', 'id': 82, 'trainId': 19},
{'name': 'book', 'id': 84, 'trainId': 64},
{'name': 'clock', 'id': 85, 'trainId': 66},
{'name': 'vase', 'id': 86, 'trainId': 67},
{'name': 'scissors', 'id': 87, 'trainId': 68},
{'name': 'teddy bear', 'id': 88, 'trainId': 69},
{'name': 'hair drier', 'id': 89, 'trainId': 70},
{'name': 'toothbrush', 'id': 90, 'trainId': 71},
{'name': 'banner', 'id': 92, 'trainId': 101},
{'name': 'blanket', 'id': 93, 'trainId': 102},
{'name': 'bridge', 'id': 95, 'trainId': 21},
{'name': 'cardboard', 'id': 100, 'trainId': 65},
{'name': 'counter', 'id': 107, 'trainId': 49},
{'name': 'curtain', 'id': 109, 'trainId': 103},
{'name': 'door-stuff', 'id': 112, 'trainId': 50},
{'name': 'floor-wood', 'id': 118, 'trainId': 31},
{'name': 'flower', 'id': 119, 'trainId': 107},
{'name': 'fruit', 'id': 122, 'trainId': 42},
{'name': 'gravel', 'id': 125, 'trainId': 56},
{'name': 'house', 'id': 128, 'trainId': 23},
{'name': 'light', 'id': 130, 'trainId': 51},
{'name': 'mirror-stuff', 'id': 133, 'trainId': 52},
{'name': 'net', 'id': 138, 'trainId': 100},
{'name': 'pillow', 'id': 141, 'trainId': 104},
{'name': 'platform', 'id': 144, 'trainId': 57},
{'name': 'playingfield', 'id': 145, 'trainId': 58},
{'name': 'railroad', 'id': 147, 'trainId': 59},
{'name': 'river', 'id': 148, 'trainId': 116},
{'name': 'road', 'id': 149, 'trainId': 60},
{'name': 'roof', 'id': 151, 'trainId': 23},
{'name': 'sand', 'id': 154, 'trainId': 63},
{'name': 'sea', 'id': 155, 'trainId': 117},
{'name': 'shelf', 'id': 156, 'trainId': 53},
{'name': 'snow', 'id': 159, 'trainId': 61},
{'name': 'stairs', 'id': 161, 'trainId': 54},
{'name': 'tent', 'id': 166, 'trainId': 22},
{'name': 'towel', 'id': 168, 'trainId': 105},
{'name': 'wall-brick', 'id': 171, 'trainId': 119},
{'name': 'wall-stone', 'id': 175, 'trainId': 119},
{'name': 'wall-tile', 'id': 176, 'trainId': 119},
{'name': 'wall-wood', 'id': 177, 'trainId': 119},
{'name': 'water-other', 'id': 178, 'trainId': 118},
{'name': 'window-blind', 'id': 180, 'trainId': 121},
{'name': 'window-other', 'id': 181, 'trainId': 120},
{'name': 'tree-merged', 'id': 184, 'trainId': 107},
{'name': 'fence-merged', 'id': 185, 'trainId': 87},
{'name': 'ceiling-merged', 'id': 186, 'trainId': 24},
{'name': 'sky-other-merged', 'id': 187, 'trainId': 86},
{'name': 'cabinet-merged', 'id': 188, 'trainId': 55},
{'name': 'table-merged', 'id': 189, 'trainId': 48},
{'name': 'floor-other-merged', 'id': 190, 'trainId': 31},
{'name': 'pavement-merged', 'id': 191, 'trainId': 62},
{'name': 'mountain-merged', 'id': 192, 'trainId': 88},
{'name': 'grass-merged', 'id': 193, 'trainId': 63},
{'name': 'dirt-merged', 'id': 194, 'trainId': 63},
{'name': 'paper-merged', 'id': 195, 'trainId': 80},
{'name': 'food-other-merged', 'id': 196, 'trainId': 43},
{'name': 'building-other-merged', 'id': 197, 'trainId': 23},
{'name': 'rock-merged', 'id': 198, 'trainId': 89},
{'name': 'wall-other-merged', 'id': 199, 'trainId': 119},
{'name': 'rug-merged', 'id': 200, 'trainId': 106},
{'name': 'unlabeled', 'id': 0, 'trainId': 255}]

num = 122
def register_coco_mseg():
    
    lb_map = {}
    for el in Mseg_label_info:
        lb_map[el['id']] = el['trainId']
    train_to_mseg_map = {}
    for train, mseg in zip(labels_info, Mseg_label_info):
        train_to_mseg_map[train['trainId']] = mseg['trainId']
    
    
    for n, anp in [("train", "train"), ("train_1", "train_1"), ("train_2", "train_2"), ("val", "val"), ("val_temp", "val_temp")]:
        name = f"coco_mseg_sem_seg_{n}"
        annpath = f'auto_uni_seg/datasets/coco/{anp}.txt'
        DatasetCatalog.register(
            name, lambda x=annpath : coco_train(x)
        )
        
        MetadataCatalog.get(name).set(
            stuff_classes=["backpack", "umbrella", "bag", "tie", "suitcase", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "microwave", "oven", "toaster", "sink", "refrigerator", "toilet", "bridge", "tent", "building", "ceiling", "laptop", "keyboard", "mouse", "remote", "cell phone", "television", "floor", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot_dog", "pizza", "donut", "cake", "fruit_other", "food_other", "chair_other", "couch", "potted_plant", "bed", "table", "counter_other", "door", "light_other", "mirror", "shelf", "stairs", "cabinet", "gravel", "platform", "playingfield", "railroad", "road", "snow", "sidewalk_pavement", "terrain", "book", "box", "clock", "vase", "scissors", "teddy_bear", "hair_dryer", "toothbrush", "bottle", "cup", "wine_glass", "knife", "fork", "spoon", "bowl", "person", "paper", "traffic_sign", "traffic_light", "fire_hydrant", "parking_meter", "bench", "sky", "fence", "mountain_hill", "rock", "frisbee", "skis", "snowboard", "sports_ball", "kite", "baseball_bat", "baseball_glove", "skateboard", "surfboard", "tennis_racket", "net", "banner", "blanket", "curtain_other", "pillow", "towel", "rug_floormat", "vegetation", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat_ship", "river_lake", "sea", "water_other", "wall", "window", "window_blind"],
            stuff_dataset_id_to_contiguous_id=lb_map,
            thing_dataset_id_to_contiguous_id=lb_map,
            evaluator_type="sem_seg",
            ignore_label=255,  
            trainId_to_msegId=train_to_mseg_map
            
        )

register_coco_mseg()
