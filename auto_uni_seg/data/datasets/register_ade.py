import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import os.path as osp

labels_info = [
    {"name": "unlabel", "id": 0, "trainId": 255},
    {"name": "flag", "id": 150, "trainId": 0},
    {"name": "wall", "id": 1, "trainId": 1},
    {"name": "building", "id": 2, "trainId": 2},
    {"name": "sky", "id": 3, "trainId": 3},
    {"name": "floor", "id": 4, "trainId": 4},
    {"name": "tree", "id": 5, "trainId": 5},
    {"name": "ceiling", "id": 6, "trainId": 6},
    {"name": "road", "id": 7, "trainId": 7},
    {"name": "bed", "id": 8, "trainId": 8},
    {"name": "windowpane", "id": 9, "trainId": 9},
    {"name": "grass", "id": 10, "trainId": 10},
    {"name": "cabinet", "id": 11, "trainId": 11},
    {"name": "sidewalk", "id": 12, "trainId": 12},
    {"name": "person", "id": 13, "trainId": 13},
    {"name": "earth", "id": 14, "trainId": 14},
    {"name": "door", "id": 15, "trainId": 15},
    {"name": "table", "id": 16, "trainId": 16},
    {"name": "mountain", "id": 17, "trainId": 17},
    {"name": "plant", "id": 18, "trainId": 18},
    {"name": "curtain", "id": 19, "trainId": 19},
    {"name": "chair", "id": 20, "trainId": 20},
    {"name": "car", "id": 21, "trainId": 21},
    {"name": "water", "id": 22 , "trainId": 22 },
    {"name": "painting", "id": 23, "trainId": 23},
    {"name": "sofa", "id": 24 , "trainId": 24 },
    {"name": "shelf", "id": 25 , "trainId": 25 },
    {"name": "house", "id": 26 , "trainId": 26 },
    {"name": "sea", "id": 27 , "trainId": 27 },
    {"name": "mirror", "id": 28, "trainId": 28},
    {"name": "rug", "id": 29, "trainId": 29},
    {"name": "field", "id": 30, "trainId": 30},
    {"name": "armchair", "id": 31, "trainId": 31},
    {"name": "seat", "id": 32, "trainId": 32},
    {"name": "fence", "id": 33, "trainId": 33},
    {"name": "desk", "id": 34, "trainId": 34},
    {"name": "rock", "id": 35, "trainId": 35},
    {"name": "wardrobe, closet, press", "id": 36, "trainId": 36},
    {"name": "lamp", "id": 37, "trainId": 37},
    {"name": "bathtub, bathing tub, bath, tub", "id": 38, "trainId": 38},
    {"name": "railing, rail", "id": 39, "trainId": 39},
    {"name": "cushion", "id": 40, "trainId": 40},
    {"name": "base, pedestal, stand", "id": 41, "trainId": 41},
    {"name": "box", "id": 42, "trainId": 42},
    {"name": "column, pillar", "id": 43, "trainId": 43},
    {"name": "signboard, sign", "id": 44, "trainId": 44},
    {"name": "chest of drawers, chest, bureau, dresser", "id": 45, "trainId": 45},
    {"name": "counter", "id": 46, "trainId": 46},
    {"name": "sand", "id": 47, "trainId": 47},
    {"name": "sink", "id": 48, "trainId": 48},
    {"name": "skyscraper", "id": 49, "trainId": 49},
    {"name": "fireplace, hearth, open fireplace", "id": 50, "trainId": 50},
    {"name": "refrigerator, icebox", "id": 51, "trainId": 51},
    {"name": "grandstand, covered stand", "id": 52, "trainId": 52},
    {"name": "path", "id": 53, "trainId": 53},
    {"name": "stairs, steps", "id": 54, "trainId": 54},
    {"name": "runway", "id": 55, "trainId": 55},
    {"name": "case, display case, showcase, vitrine", "id": 56, "trainId": 56},
    {"name": "pool table, billiard table, snooker table", "id": 57, "trainId": 57},
    {"name": "pillow", "id": 58, "trainId": 58},
    {"name": "screen door, screen", "id": 59, "trainId": 59},
    {"name": "stairway, staircase", "id": 60, "trainId": 60},
    {"name": "river", "id": 61, "trainId": 61},
    {"name": "bridge, span", "id": 62, "trainId": 62},
    {"name": "bookcase", "id": 63, "trainId": 63},
    {"name": "blind, screen", "id": 64, "trainId": 64},
    {"name": "coffee table, cocktail table", "id": 65, "trainId": 65},
    {"name": "toilet, can, commode, crapper, pot, potty, stool, throne", "id": 66, "trainId": 66},
    {"name": "flower", "id": 67, "trainId": 67},
    {"name": "book", "id": 68, "trainId": 68},
    {"name": "hill", "id": 69, "trainId": 69},
    {"name": "bench", "id": 70, "trainId": 70},
    {"name": "countertop", "id": 71, "trainId": 71},
    {"name": "stove, kitchen stove, range, kitchen range, cooking stove", "id": 72, "trainId": 72},
    {"name": "palm, palm tree", "id": 73, "trainId": 73},
    {"name": "kitchen island", "id": 74, "trainId": 74},
    {"name": "computer, computing machine, computing device, data processor, electronic computer, information processing system", "id": 75, "trainId": 75},
    {"name": "swivel chair", "id": 76, "trainId": 76},
    {"name": "boat", "id": 77, "trainId": 77},
    {"name": "bar", "id": 78, "trainId": 78},
    {"name": "arcade machine", "id": 79, "trainId": 79},
    {"name": "hovel, hut, hutch, shack, shanty", "id": 80, "trainId": 80},
    {"name": "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle", "id": 81, "trainId": 81},
    {"name": "towel", "id": 82, "trainId": 82},
    {"name": "light, light source", "id": 83, "trainId": 83},
    {"name": "truck, motortruck", "id": 84, "trainId": 84},
    {"name": "tower", "id": 85, "trainId": 85},
    {"name": "chandelier, pendant, pendent", "id": 86, "trainId": 86},
    {"name": "awning, sunshade, sunblind", "id": 87, "trainId": 87},
    {"name": "streetlight, street lamp", "id": 88, "trainId": 88},
    {"name": "booth, cubicle, stall, kiosk", "id": 89, "trainId": 89},
    {"name": "television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box", "id": 90, "trainId": 90},
    {"name": "airplane, aeroplane, plane", "id": 91, "trainId": 91},
    {"name": "dirt track", "id": 92, "trainId": 92},
    {"name": "apparel, wearing apparel, dress, clothes", "id": 93, "trainId": 93},
    {"name": "pole", "id": 94, "trainId": 94},
    {"name": "land, ground, soil", "id": 95, "trainId": 95},
    {"name": "bannister, banister, balustrade, balusters, handrail", "id": 96, "trainId": 96},
    {"name": "escalator, moving staircase, moving stairway", "id": 97, "trainId": 97},
    {"name": "ottoman, pouf, pouffe, puff, hassock", "id": 98, "trainId": 98},
    {"name": "bottle", "id": 99, "trainId": 99},
    {"name": "buffet, counter, sideboard", "id": 100, "trainId": 100},
    {"name": "poster, posting, placard, notice, bill, card", "id": 101, "trainId": 101},
    {"name": "stage", "id": 102, "trainId": 102},
    {"name": "van", "id": 103, "trainId": 103},
    {"name": "ship", "id": 104, "trainId": 104},
    {"name": "fountain", "id": 105, "trainId": 105},
    {"name": "conveyer belt, conveyor belt, conveyer, conveyor, transporter", "id": 106, "trainId": 106},
    {"name": "canopy", "id": 107, "trainId": 107},
    {"name": "washer, automatic washer, washing machine", "id": 108, "trainId": 108},
    {"name": "plaything, toy", "id": 109, "trainId": 109},
    {"name": "swimming pool, swimming bath, natatorium", "id": 110, "trainId": 110},
    {"name": "stool", "id": 111, "trainId": 111},
    {"name": "barrel, cask", "id": 112, "trainId": 112},
    {"name": "basket, handbasket", "id": 113, "trainId": 113},
    {"name": "waterfall, falls", "id": 114, "trainId": 114},
    {"name": "tent, collapsible shelter", "id": 115, "trainId": 115},
    {"name": "bag", "id": 116, "trainId": 116},
    {"name": "minibike, motorbike", "id": 117, "trainId": 117},
    {"name": "cradle", "id": 118, "trainId": 118},
    {"name": "oven", "id": 119, "trainId": 119},
    {"name": "ball", "id": 120, "trainId": 120},
    {"name": "food, solid food", "id": 121, "trainId": 121},
    {"name": "step, stair", "id": 122, "trainId": 122},
    {"name": "tank, storage tank", "id": 123, "trainId": 123},
    {"name": "trade name, brand name, brand, marque", "id": 124, "trainId": 124},
    {"name": "microwave, microwave oven", "id": 125, "trainId": 125},
    {"name": "pot, flowerpot", "id": 126, "trainId": 126},
    {"name": "animal, animate being, beast, brute, creature, fauna", "id": 127, "trainId": 127},
    {"name": "bicycle, bike, wheel, cycle ", "id": 128, "trainId": 128},
    {"name": "lake", "id": 129, "trainId": 129},
    {"name": "dishwasher, dish washer, dishwashing machine", "id": 130, "trainId": 130},
    {"name": "screen, silver screen, projection screen", "id": 131, "trainId": 131},
    {"name": "blanket, cover", "id": 132, "trainId": 132},
    {"name": "sculpture", "id": 133, "trainId": 133},
    {"name": "hood, exhaust hood", "id": 134, "trainId": 134},
    {"name": "sconce", "id": 135, "trainId": 135},
    {"name": "vase", "id": 136, "trainId": 136},
    {"name": "traffic light, traffic signal, stoplight", "id": 137, "trainId": 137},
    {"name": "tray", "id": 138, "trainId": 138},
    {"name": "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin", "id": 139, "trainId": 139},
    {"name": "fan", "id": 140, "trainId": 140},
    {"name": "pier, wharf, wharfage, dock", "id": 141, "trainId": 141},
    {"name": "crt screen", "id": 142, "trainId": 142},
    {"name": "plate", "id": 143, "trainId": 143},
    {"name": "monitor, monitoring device", "id": 144, "trainId": 144},
    {"name": "bulletin board, notice board", "id": 145, "trainId": 145},
    {"name": "shower", "id": 146, "trainId": 146},
    {"name": "radiator", "id": 147, "trainId": 147},
    {"name": "glass, drinking glass", "id": 148, "trainId": 148},
    {"name": "clock", "id": 149, "trainId": 149},
]
dataroot = 'datasets/ADEChallengeData2016'
annpath = f'auto_uni_seg/datasets/ADE/validation.txt'
def ade():
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


def register_ade():
    
    
    # meta = _get_ade20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in labels_info:
        lb_map[el['id']] = el['trainId']
        # if el['id'] == 0:
        #     lb_map[el['id']] = 255
        # else:
        #     lb_map[el['id']] = el['id'] - 1

    name = f"ade_sem_seg_val"
    DatasetCatalog.register(
        name, ade
    )
    
    
    MetadataCatalog.get(name).set(
        stuff_classes=['flag', 'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock'],
        stuff_dataset_id_to_contiguous_id=lb_map,
        thing_dataset_id_to_contiguous_id=lb_map,
        stuff_colors=[[ 34, 155,  59],[174, 203, 111],     [154, 172, 132],     [225, 115,  77],     [ 23,  69, 195],     [ 68,  68, 237],     [ 19, 228,  89],     [ 97, 187, 159],     [ 10,   7, 111],     [ 35, 133,  44],     [188,  49,  31],     [233, 198,  64],     [236,   2, 131],     [198, 221, 182],     [ 25, 250, 252],     [ 77, 110,  58],     [  7, 162,   5],     [252, 168, 166],     [128, 109,  69],     [113,  39, 205],     [ 38, 181, 228],     [ 14, 159,  57],     [ 79,  24, 196],     [141, 251,  82],     [ 75, 211,  41],     [126,  22, 239],     [189, 161,  55],     [199, 183,  46],     [146, 240, 187],     [195, 177, 180],     [112,   2, 104],     [132, 178,  25],     [214, 111, 212],     [145, 208,  42],     [ 83,  69,  70],     [234,  44,  84],     [167, 230, 156],     [168,  62, 253],     [ 12,  71,  15],     [180, 241, 180],     [ 56,  31, 101],     [ 25, 111,  21],     [ 58,  34,   9],     [192,  49, 133],     [225,  30,  12],     [191,  27, 123],     [138,  37,  98],     [135, 161, 131],     [236, 223, 130],     [ 18,  18, 139],     [211, 209, 237],     [177, 158, 120],     [ 55, 204,   1],     [ 39, 192,  19],     [230, 246,  34],     [ 38,  98, 154],     [ 51,  56,  95],     [ 72, 172, 147],     [ 54,   3, 109],     [137, 139, 117],     [127, 244, 110],     [136,  66,  69],     [ 76,  66, 237],     [163, 186, 153],     [233,  94, 191],     [ 91, 182,  84],     [109, 155, 209],     [218,  37, 136],     [240,  79,  15],     [ 87,  48, 117],     [ 47, 118, 136],     [127,   2,  56],     [109, 248,  25],     [220, 218,  77],     [219, 103,  62],     [214,  92,  19],     [230, 175, 199],     [106,  48, 148],     [212, 221, 176],     [ 87, 177, 247],     [193, 106,  86],     [ 79, 127, 244],     [235,  91,  37],     [104, 162, 189],     [ 81, 255, 160],     [ 33, 116, 176],     [  3, 186, 105],     [174,  67, 252],     [ 26, 184,  83],     [142, 222, 139],     [ 61, 206, 227],     [ 87,  59,  76],     [240,  29, 224],     [112,  78, 139],     [ 70,  94, 120],     [149, 202, 142],     [141,  17,  60],     [176,  92, 116],     [ 68, 217,   9],     [217, 135,  68],     [171, 130, 103],     [110, 237, 116],     [233,  37,  88],     [249, 125, 105],     [171,  98,  21],     [174,  68,   3],     [240, 255, 149],     [139,  23,  73],     [214,   3, 173],     [231, 117, 185],     [249, 159, 243],     [114,  64,  69],     [248, 107, 184],     [ 99,  16, 132],     [172, 222,  36],     [189, 124,   9],     [108, 156,  92],     [ 12,  50, 106],     [178, 135,   6],     [193, 163, 215],     [ 31, 148,  91],     [180,  12, 242],     [125, 123, 180],     [  9,  57, 229],     [ 44,  87,  68],     [184, 102,  69],     [206, 106,  97],     [ 76,  21,  14],     [103,  81, 234],     [164, 161,  80],     [230, 183, 128],     [195, 211, 185],     [ 31, 228, 121],     [235,  96,  62],     [198, 132,  42],     [ 46, 140, 152],     [111, 173,  82],     [153,  29, 134],     [106, 248, 185],     [ 30,  47, 184],     [184,  49, 182],     [129, 139, 126],     [ 80, 108,  57],     [213, 208, 155],     [167, 118,  46],     [ 27, 138, 140],     [146,  38,  43],     [ 89,  54, 145],     [158,  92,  94],     [115, 177,  95]],
        evaluator_type="sem_seg",
        ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
    )



# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_ade()

def ade_train(anp):
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


def register_ade_train():
    
    
    # meta = _get_ade20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in labels_info:
        lb_map[el['id']] = el['trainId']
    for n, anp in [("train", "training"), ("train_1", "training_1"), ("train_2", "training_2")]:
        name = f"ade_sem_seg_{n}"
        annpath = f'auto_uni_seg/datasets/ADE/{anp}.txt'
        DatasetCatalog.register(
            name, lambda x=annpath : ade_train(x)
        )
        
    
        MetadataCatalog.get(name).set(
            stuff_classes=['flag', 'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock'],
            stuff_dataset_id_to_contiguous_id=lb_map,
            thing_dataset_id_to_contiguous_id=lb_map,
            stuff_colors=[[ 34, 155,  59],[174, 203, 111],     [154, 172, 132],     [225, 115,  77],     [ 23,  69, 195],     [ 68,  68, 237],     [ 19, 228,  89],     [ 97, 187, 159],     [ 10,   7, 111],     [ 35, 133,  44],     [188,  49,  31],     [233, 198,  64],     [236,   2, 131],     [198, 221, 182],     [ 25, 250, 252],     [ 77, 110,  58],     [  7, 162,   5],     [252, 168, 166],     [128, 109,  69],     [113,  39, 205],     [ 38, 181, 228],     [ 14, 159,  57],     [ 79,  24, 196],     [141, 251,  82],     [ 75, 211,  41],     [126,  22, 239],     [189, 161,  55],     [199, 183,  46],     [146, 240, 187],     [195, 177, 180],     [112,   2, 104],     [132, 178,  25],     [214, 111, 212],     [145, 208,  42],     [ 83,  69,  70],     [234,  44,  84],     [167, 230, 156],     [168,  62, 253],     [ 12,  71,  15],     [180, 241, 180],     [ 56,  31, 101],     [ 25, 111,  21],     [ 58,  34,   9],     [192,  49, 133],     [225,  30,  12],     [191,  27, 123],     [138,  37,  98],     [135, 161, 131],     [236, 223, 130],     [ 18,  18, 139],     [211, 209, 237],     [177, 158, 120],     [ 55, 204,   1],     [ 39, 192,  19],     [230, 246,  34],     [ 38,  98, 154],     [ 51,  56,  95],     [ 72, 172, 147],     [ 54,   3, 109],     [137, 139, 117],     [127, 244, 110],     [136,  66,  69],     [ 76,  66, 237],     [163, 186, 153],     [233,  94, 191],     [ 91, 182,  84],     [109, 155, 209],     [218,  37, 136],     [240,  79,  15],     [ 87,  48, 117],     [ 47, 118, 136],     [127,   2,  56],     [109, 248,  25],     [220, 218,  77],     [219, 103,  62],     [214,  92,  19],     [230, 175, 199],     [106,  48, 148],     [212, 221, 176],     [ 87, 177, 247],     [193, 106,  86],     [ 79, 127, 244],     [235,  91,  37],     [104, 162, 189],     [ 81, 255, 160],     [ 33, 116, 176],     [  3, 186, 105],     [174,  67, 252],     [ 26, 184,  83],     [142, 222, 139],     [ 61, 206, 227],     [ 87,  59,  76],     [240,  29, 224],     [112,  78, 139],     [ 70,  94, 120],     [149, 202, 142],     [141,  17,  60],     [176,  92, 116],     [ 68, 217,   9],     [217, 135,  68],     [171, 130, 103],     [110, 237, 116],     [233,  37,  88],     [249, 125, 105],     [171,  98,  21],     [174,  68,   3],     [240, 255, 149],     [139,  23,  73],     [214,   3, 173],     [231, 117, 185],     [249, 159, 243],     [114,  64,  69],     [248, 107, 184],     [ 99,  16, 132],     [172, 222,  36],     [189, 124,   9],     [108, 156,  92],     [ 12,  50, 106],     [178, 135,   6],     [193, 163, 215],     [ 31, 148,  91],     [180,  12, 242],     [125, 123, 180],     [  9,  57, 229],     [ 44,  87,  68],     [184, 102,  69],     [206, 106,  97],     [ 76,  21,  14],     [103,  81, 234],     [164, 161,  80],     [230, 183, 128],     [195, 211, 185],     [ 31, 228, 121],     [235,  96,  62],     [198, 132,  42],     [ 46, 140, 152],     [111, 173,  82],     [153,  29, 134],     [106, 248, 185],     [ 30,  47, 184],     [184,  49, 182],     [129, 139, 126],     [ 80, 108,  57],     [213, 208, 155],     [167, 118,  46],     [ 27, 138, 140],     [146,  38,  43],     [ 89,  54, 145],     [158,  92,  94],     [115, 177,  95]],
            evaluator_type="sem_seg",
            ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
        )

register_ade_train()


Mseg_label_info = [{'name': 'unlabel', 'id': 0, 'trainId': 255},
{'name': 'flag', 'id': 150, 'trainId': 95},
{'name': 'wall', 'id': 1, 'trainId': 114},
{'name': 'building', 'id': 2, 'trainId': 19},
{'name': 'sky', 'id': 3, 'trainId': 82},
{'name': 'floor', 'id': 4, 'trainId': 22},
{'name': 'tree', 'id': 5, 'trainId': 101},
{'name': 'ceiling', 'id': 6, 'trainId': 20},
{'name': 'road', 'id': 7, 'trainId': 60},
{'name': 'bed', 'id': 8, 'trainId': 33},
{'name': 'windowpane', 'id': 9, 'trainId': 115},
{'name': 'grass', 'id': 10, 'trainId': 63},
{'name': 'cabinet', 'id': 11, 'trainId': 56},
{'name': 'sidewalk', 'id': 12, 'trainId': 61},
{'name': 'person', 'id': 13, 'trainId': 77},
{'name': 'earth', 'id': 14, 'trainId': 63},
{'name': 'door', 'id': 15, 'trainId': 47},
{'name': 'table', 'id': 16, 'trainId': 34},
{'name': 'mountain', 'id': 17, 'trainId': 86},
{'name': 'plant', 'id': 18, 'trainId': 255},
{'name': 'curtain', 'id': 19, 'trainId': 97},
{'name': 'chair', 'id': 20, 'trainId': 25},
{'name': 'car', 'id': 21, 'trainId': 103},
{'name': 'water', 'id': 22, 'trainId': 111},
{'name': 'painting', 'id': 23, 'trainId': 69},
{'name': 'sofa', 'id': 24, 'trainId': 30},
{'name': 'shelf', 'id': 25, 'trainId': 53},
{'name': 'house', 'id': 26, 'trainId': 19},
{'name': 'sea', 'id': 27, 'trainId': 110},
{'name': 'mirror', 'id': 28, 'trainId': 52},
{'name': 'rug', 'id': 29, 'trainId': 100},
{'name': 'field', 'id': 30, 'trainId': 63},
{'name': 'armchair', 'id': 31, 'trainId': 26},
{'name': 'seat', 'id': 32, 'trainId': 29},
{'name': 'fence', 'id': 33, 'trainId': 84},
{'name': 'desk', 'id': 34, 'trainId': 37},
{'name': 'rock', 'id': 35, 'trainId': 87},
{'name': 'wardrobe', 'id': 36, 'trainId': 39},
{'name': 'lamp', 'id': 37, 'trainId': 49},
{'name': 'bathtub', 'id': 38, 'trainId': 14},
{'name': 'railing', 'id': 39, 'trainId': 85},
{'name': 'cushion', 'id': 40, 'trainId': 98},
{'name': 'base', 'id': 41, 'trainId': 89},
{'name': 'box', 'id': 42, 'trainId': 65},
{'name': 'column', 'id': 43, 'trainId': 91},
{'name': 'signboard', 'id': 44, 'trainId': 255},
{'name': 'chest of drawers', 'id': 45, 'trainId': 42},
{'name': 'counter', 'id': 46, 'trainId': 44},
{'name': 'sand', 'id': 47, 'trainId': 63},
{'name': 'sink', 'id': 48, 'trainId': 8},
{'name': 'skyscraper', 'id': 49, 'trainId': 19},
{'name': 'fireplace', 'id': 50, 'trainId': 57},
{'name': 'refrigerator', 'id': 51, 'trainId': 9},
{'name': 'grandstand', 'id': 52, 'trainId': 19},
{'name': 'path', 'id': 53, 'trainId': 61},
{'name': 'stairs', 'id': 54, 'trainId': 54},
{'name': 'runway', 'id': 55, 'trainId': 62},
{'name': 'case', 'id': 56, 'trainId': 1},
{'name': 'pool table', 'id': 57, 'trainId': 35},
{'name': 'pillow', 'id': 58, 'trainId': 98},
{'name': 'screen door', 'id': 59, 'trainId': 47},
{'name': 'stairway', 'id': 60, 'trainId': 54},
{'name': 'river', 'id': 61, 'trainId': 109},
{'name': 'bridge', 'id': 62, 'trainId': 16},
{'name': 'bookcase', 'id': 63, 'trainId': 43},
{'name': 'blind', 'id': 64, 'trainId': 116},
{'name': 'coffee table', 'id': 65, 'trainId': 34},
{'name': 'toilet', 'id': 66, 'trainId': 13},
{'name': 'flower', 'id': 67, 'trainId': 101},
{'name': 'book', 'id': 68, 'trainId': 64},
{'name': 'hill', 'id': 69, 'trainId': 86},
{'name': 'bench', 'id': 70, 'trainId': 80},
{'name': 'countertop', 'id': 71, 'trainId': 45},
{'name': 'stove', 'id': 72, 'trainId': 58},
{'name': 'palm', 'id': 73, 'trainId': 101},
{'name': 'kitchen island', 'id': 74, 'trainId': 46},
{'name': 'computer', 'id': 75, 'trainId': 255},
{'name': 'swivel chair', 'id': 76, 'trainId': 27},
{'name': 'boat', 'id': 77, 'trainId': 108},
{'name': 'bar', 'id': 78, 'trainId': 255},
{'name': 'arcade machine', 'id': 79, 'trainId': 59},
{'name': 'hovel', 'id': 80, 'trainId': 19},
{'name': 'bus', 'id': 81, 'trainId': 106},
{'name': 'towel', 'id': 82, 'trainId': 99},
{'name': 'light', 'id': 83, 'trainId': 48},
{'name': 'truck', 'id': 84, 'trainId': 107},
{'name': 'tower', 'id': 85, 'trainId': 19},
{'name': 'chandelier', 'id': 86, 'trainId': 51},
{'name': 'awning', 'id': 87, 'trainId': 93},
{'name': 'streetlight', 'id': 88, 'trainId': 78},
{'name': 'booth', 'id': 89, 'trainId': 19},
{'name': 'television receiver', 'id': 90, 'trainId': 21},
{'name': 'airplane', 'id': 91, 'trainId': 105},
{'name': 'dirt track', 'id': 92, 'trainId': 63},
{'name': 'apparel', 'id': 93, 'trainId': 94},
{'name': 'pole', 'id': 94, 'trainId': 83},
{'name': 'land', 'id': 95, 'trainId': 63},
{'name': 'bannister', 'id': 96, 'trainId': 85},
{'name': 'escalator', 'id': 97, 'trainId': 55},
{'name': 'ottoman', 'id': 98, 'trainId': 38},
{'name': 'bottle', 'id': 99, 'trainId': 72},
{'name': 'buffet', 'id': 100, 'trainId': 56},
{'name': 'poster', 'id': 101, 'trainId': 70},
{'name': 'stage', 'id': 102, 'trainId': 23},
{'name': 'van', 'id': 103, 'trainId': 103},
{'name': 'ship', 'id': 104, 'trainId': 108},
{'name': 'fountain', 'id': 105, 'trainId': 92},
{'name': 'conveyer belt', 'id': 106, 'trainId': 7},
{'name': 'canopy', 'id': 107, 'trainId': 255},
{'name': 'washer', 'id': 108, 'trainId': 10},
{'name': 'plaything', 'id': 109, 'trainId': 68},
{'name': 'swimming pool', 'id': 110, 'trainId': 112},
{'name': 'stool', 'id': 111, 'trainId': 28},
{'name': 'barrel', 'id': 112, 'trainId': 36},
{'name': 'basket', 'id': 113, 'trainId': 41},
{'name': 'waterfall', 'id': 114, 'trainId': 113},
{'name': 'tent', 'id': 115, 'trainId': 18},
{'name': 'bag', 'id': 116, 'trainId': 0},
{'name': 'minibike', 'id': 117, 'trainId': 104},
{'name': 'cradle', 'id': 118, 'trainId': 40},
{'name': 'oven', 'id': 119, 'trainId': 5},
{'name': 'ball', 'id': 120, 'trainId': 88},
{'name': 'food', 'id': 121, 'trainId': 24},
{'name': 'step', 'id': 122, 'trainId': 255},
{'name': 'tank', 'id': 123, 'trainId': 6},
{'name': 'trade name', 'id': 124, 'trainId': 81},
{'name': 'microwave', 'id': 125, 'trainId': 3},
{'name': 'pot', 'id': 126, 'trainId': 32},
{'name': 'animal', 'id': 127, 'trainId': 2},
{'name': 'bicycle', 'id': 128, 'trainId': 102},
{'name': 'lake', 'id': 129, 'trainId': 109},
{'name': 'dishwasher', 'id': 130, 'trainId': 12},
{'name': 'screen', 'id': 131, 'trainId': 255},
{'name': 'blanket', 'id': 132, 'trainId': 96},
{'name': 'sculpture', 'id': 133, 'trainId': 90},
{'name': 'hood', 'id': 134, 'trainId': 75},
{'name': 'sconce', 'id': 135, 'trainId': 50},
{'name': 'vase', 'id': 136, 'trainId': 67},
{'name': 'traffic light', 'id': 137, 'trainId': 79},
{'name': 'tray', 'id': 138, 'trainId': 74},
{'name': 'ashcan', 'id': 139, 'trainId': 31},
{'name': 'fan', 'id': 140, 'trainId': 11},
{'name': 'pier', 'id': 141, 'trainId': 17},
{'name': 'crt screen', 'id': 142, 'trainId': 255},
{'name': 'plate', 'id': 143, 'trainId': 76},
{'name': 'monitor', 'id': 144, 'trainId': 255},
{'name': 'bulletin board', 'id': 145, 'trainId': 71},
{'name': 'shower', 'id': 146, 'trainId': 15},
{'name': 'radiator', 'id': 147, 'trainId': 4},
{'name': 'glass', 'id': 148, 'trainId': 73},
{'name': 'clock', 'id': 149, 'trainId': 66}]

num = 117
def register_ade_mseg():
    
    
    # meta = _get_ade20k_full_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    # dirname = 'train'
    lb_map = {}
    for el in Mseg_label_info:
        lb_map[el['id']] = el['trainId']
    
    train_to_mseg_map = {}
    for train, mseg in zip(labels_info, Mseg_label_info):
        train_to_mseg_map[train['trainId']] = mseg['trainId']
    for n, anp in [("train", "training"), ("train_1", "training_1"), ("train_2", "training_2"), ("val", "validation")]:
        name = f"ade_mseg_sem_seg_{n}"
        annpath = f'auto_uni_seg/datasets/ADE/{anp}.txt'
        DatasetCatalog.register(
            name, lambda x=annpath : ade_train(x)
        )
        
        MetadataCatalog.get(name).set(
            stuff_classes=['flag', 'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock'],
            stuff_dataset_id_to_contiguous_id=lb_map,
            thing_dataset_id_to_contiguous_id=lb_map,
            evaluator_type="sem_seg",
            ignore_label=255,  
            trainId_to_msegId=train_to_mseg_map
        )

register_ade_mseg()
