# libraries 导入模块
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from detectron2.data import MetadataCatalog
from auto_uni_seg import *
np.random.seed(123)
palette = np.random.randint(0, 256, (512, 3), dtype=np.uint8)
colors = {}

datasets = ['city', 'mapi', 'sun', 'bdd', 'idd', 'ade', 'coco']
# datasets = ['city', 'sun', 'cv']
# datasets = ['city', 'mapi', 'bdd', 'idd', 'wd']
metalist = [MetadataCatalog.get('cs_sem_seg_train'),
            MetadataCatalog.get('mapi_sem_seg_train'),
            MetadataCatalog.get('bdd_sem_seg_train'),
            MetadataCatalog.get('idd_sem_seg_train')]
stuff_colors = [meta.stuff_colors for meta in metalist]
city_lb = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
mapi_lb = ["Bird", "Ground Animal", "Curb", "Fence", "Guard Rail", "Barrier", "Wall", "Bike Lane", "Crosswalk - Plain", "Curb Cut", "Parking", "Pedestrian Area", "Rail Track", "Road", "Service Lane", "Sidewalk", "Bridge", "Building", "Tunnel", "Person", "Bicyclist", "Motorcyclist", "Other Rider", "Lane Marking - Crosswalk", "Lane Marking - General", "Mountain", "Sand", "Sky", "Snow", "Terrain", "Vegetation", "Water", "Banner", "Bench", "Bike Rack", "Billboard", "Catch Basin", "CCTV Camera", "Fire Hydrant", "Junction Box", "Manhole", "Phone Booth", "Pothole", "Street Light", "Pole", "Traffic Sign Frame", "Utility Pole", "Traffic Light", "Traffic Sign (Back)", "Traffic Sign (Front)", "Trash Can", "Bicycle", "Boat", "Bus", "Car", "Caravan", "Motorcycle", "On Rails", "Other Vehicle", "Trailer", "Truck", "Wheeled Slow", "Car Mount", "Ego Vehicle"]
sun_lb = [ "bag", "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter", "blinds", "desk", "shelves", "curtain", "dresser", "pillow", "mirror", "floor mat", "clothes", "ceiling", "books", "refridgerator", "television", "paper", "towel", "shower curtain", "box", "whiteboard", "person", "night stand", "toilet", "sink", "lamp", "bathtub"]
bdd_lb = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
idd_lb = ["road", "drivable fallback or parking", "sidewalk", "non-drivable fallback or rail track", "person or animal", "out of roi or rider", "motorcycle", "bicycle", "autorickshaw", "car", "truck", "bus", "trailer or caravan or vehicle fallback", "curb", "wall", "fence", "guard rail", "billboard", "traffic sign", "traffic light", "polegroup or pole", "obs-str-bar-fallback", "building", "tunnel or bridge", "vegetation", "sky or fallback background"]
ade_lb = ['flag', 'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock']
coco_lb = ["rug-merged", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "banner", "blanket", "bridge", "cardboard", "counter", "curtain", "door-stuff", "floor-wood", "flower", "fruit", "gravel", "house", "light", "mirror-stuff", "net", "pillow", "platform", "playingfield", "railroad", "river", "road", "roof", "sand", "sea", "shelf", "snow", "stairs", "tent", "towel", "wall-brick", "wall-stone", "wall-tile", "wall-wood", "water-other", "window-blind", "window-other", "tree-merged", "fence-merged", "ceiling-merged", "sky-other-merged", "cabinet-merged", "table-merged", "floor-other-merged", "pavement-merged", "mountain-merged", "grass-merged", "dirt-merged", "paper-merged", "food-other-merged", "building-other-merged", "rock-merged", "wall-other-merged"]
wd_lb = ["ego vehicle", "road", "sidewalk", "building", "wall", "fence", "guard rail", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "motorcycle", "bicycle", "pickup", "van", "billboard", "street-light", "road-marking", "void"]
cv_lb = ["Sky", "Building", "Column_Pole", "Road", "Sidewalk", "Tree", "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist"]
# focus_list = [3, 23, 31, 62, 67, 94, 107, 127, 144, 152, 160, 164, 205, 223]
id_to_lb = []
id_to_ori_id = []
id_to_datasetid = []
for name in datasets:
    id_to_lb.extend([name+':'+lb_name for lb_name in eval(name+"_lb")])
    id_to_datasetid.extend([datasets.index(name)]*len(eval(name+"_lb")))
    id_to_ori_id.extend([i for i in range(len(eval(name+"_lb")))])
# print(id_to_ori_id)
# state = torch.load('output/find_unuse_7ds_model_hrnet.pth')
state = torch.load('output/model_0054999.pth')
# state = torch.load('output/model_0004999.pth')
# bigraphs = [state()[f'proj_head.bipartite_graphs.{i}'] for i in range(len(datasets))]
# bigraphs = [state[f'proj_head.bipartite_graphs.{i}'] for i in range(len(datasets))]
bigraphs = [state['model'][f'proj_head.bipartite_graphs.{i}'] for i in range(len(datasets))]
bigraph = torch.cat(bigraphs, dim=0)

graph = torch.load('output/init_adj_7_datasets_2.pt')

for ds in datasets:
    num = len(eval(ds+'_lb'))
    colors[ds] = [[0,0,0] for _ in range(num)]

from_list = []
to_list = []
ID_list = []
myvalue_list = []
# for i in range(graph.shape[0]):
#     for j in range(graph.shape[1]):
#         if graph[i][j] == 1:
#             from_list.append(str(j))
#             to_list.append(id_to_lb[i])
#             ID_list.append(id_to_lb[i])
#             myvalue_list.append(datasets[id_to_datasetid[i]])
#             # print(len(colors[ds]))
            
#             colors[datasets[id_to_datasetid[i]]][id_to_ori_id[i]] = palette[j]

# ID_list.extend([str(i) for i in range(graph.shape[1])])
# myvalue_list.extend(['uni']*graph.shape[1])
# wd_colors = [None] * 26
# uni_color = []
# for i in range(bigraph.shape[1]):
#     cur_colors = []
#     wd_list = []
#     for j in range(bigraph.shape[0]):
#         if bigraph[i][j] == 1:
#             cur_dataset_id = id_to_datasetid[j]
#             if cur_dataset_id != 4:
#                 cur_colors.append(stuff_colors[cur_dataset_id][id_to_ori_id[j]])
#             else:
#                 wd_list.append(id_to_ori_id[j])
    
#     flag = True
#     for colors in cur_colors:
#         if colors[0] != cur_colors[0][0] or colors[2] != cur_colors[0][2] or colors[2] != cur_colors[0][2]:
#             flag = False
#             break
#     if flag == True:
#         uni_color.append(cur_colors[0])  
#         for j in wd_list:
#             wd_colors[j] = cur_colors[0]
#     else:
#         uni_color.append(cur_colors)  
#         for j in wd_list:
#             wd_colors[j] = cur_colors
            
            
# # print(myvalue_list)


# print(f"uni: {uni_color}")
# print(f'wd: {wd_colors}')
# exit(0)
# for ds in datasets:
#     print(f"{ds}: {np.array2string(np.array(colors[ds]), separator=', ')}")
# fds
total_num = 0
for i in range(bigraph.shape[1]):
    flag = False
    for j in range(bigraph.shape[0]):
        if bigraph[j][i] == 1:
            total_num += 1
            break
print(total_num)

for i in range(bigraph.shape[0]):
    flag = False
    for j in range(bigraph.shape[1]):
        if bigraph[i][j] == 1:
            # from_list.append(str(j)+'_')
            # to_list.append(id_to_lb[i]+'_'+str(j))

            # ID_list.append(id_to_lb[i]+'_'+str(j))
            # myvalue_list.append(datasets[id_to_datasetid[i]])
            from_list.append(str(j))
            to_list.append(id_to_lb[i])
            if id_to_lb[i] not in ID_list:
                ID_list.append(id_to_lb[i])
                myvalue_list.append(datasets[id_to_datasetid[i]])
# print(bigraph.shape)

# exit(0)
# print(myvalue_list)
# ID_list.extend([str(i)+'_' for i in range(bigraph.shape[1])])
ID_list.extend([str(i) for i in range(bigraph.shape[1])])
myvalue_list.extend(['uni']*bigraph.shape[1])

for i in range(graph.shape[1]):
    from_list.append(str(i)+'_')
    to_list.append(str(i))


# with open('tools/uni_space.txt', 'r') as f:
#     lines = f.readlines()
#     for j, line in enumerate(lines):
#         if line != '\n':
#             targets = line.strip().split(';')[:-1]
#             for target in targets:
#                 lb = target.split('(')[0]
#                 from_list.append(str(j)+'_')
#                 to_list.append(lb)
#                 if lb not in ID_list:
#                     ID_list.append(lb)
#                     myvalue_list.append(lb.split(':')[0].replace(" ", ''))
                    
                
                


# dict = {}
# for id in ID_list:
#     if id not in dict:
#         dict[id] = 1
#     else:
#         dict[id] += 1
# for key in dict:
#     if dict[key] > 1:
#         print(key)
#         print(dict[key])
# print(ID_list)

# print(myvalue_list)
# for i in range(graph.shape[1]):
#     from_list.append('super')
#     to_list.append(str(i))
    
# ID_list.append('super')
# myvalue_list.append('super')
# Build a dataframe with your connections
df = pd.DataFrame({ 'from':from_list, 'to':to_list})

# And a data frame with characteristics for your nodes
carac = pd.DataFrame({ 'ID':ID_list, 'myvalue':myvalue_list})
fig = plt.figure(figsize=(150,150))
# Build your graph
# 建立图
G=nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.Graph() )
 
# The order of the node for networkX is the following order:
# 打印节点顺序
# G.nodes()
# Thus, we cannot give directly the 'myvalue' column to netowrkX, we need to arrange the order!
 
# Here is the tricky part: I need to reorder carac to assign the good color to each node
carac= carac.set_index('ID')
# print(carac[carac.index.duplicated()])
# print(ID_list.duplicated())
# 根据节点顺序设定值
carac=carac.reindex(G.nodes())
 
# And I need to transform my categorical column in a numerical value: group1->1, group2->2...
# 设定类别
carac['myvalue']=pd.Categorical(carac['myvalue'])
# carac['myvalue'].cat.codes
    
# Custom the nodes:
nx.draw(G, with_labels=True, node_color=carac['myvalue'].cat.codes, cmap=plt.cm.Set1, node_size=500, pos=nx.fruchterman_reingold_layout(G))
plt.savefig("Graph_relation_7ds_hrnet_2.png", format="PNG")