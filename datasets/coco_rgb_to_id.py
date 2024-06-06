import json
from PIL import Image, ImageDraw
import numpy as np



for mode in ['train', 'val']:
    with open(f'panoptic_{mode}2017.json', 'r') as f:
        data = json.load(f)

    categories_dict = {category['id']: category['name'] for category in data['categories']}


    if mode != 'train':
        for i, annotation in enumerate(data['annotations']):
            
            img_name = annotation['file_name']
            # print(annotation['file_name'])
            # print(annotation['image_id'])
            # print(img_name)
            img_path = f'../images/{mode}2017/' + img_name
            lb_path = f'panoptic_{mode}2017/' + img_name.replace('.jpg', '.png')
            # print(lb_path)
            lb = np.array(Image.open(lb_path).convert('RGB'))
            # print(lb.shape)
            imH, imW, _ = lb.shape
            annotation_image = np.array(Image.new('L', (imW, imH), 255))
            # print(annotation_image.shape)
            # draw = ImageDraw.Draw(annotation_image)
            if i % 1000 == 0:
                print(f'finish:{i}')
            
            for segment_info in annotation['segments_info']:
                id =  segment_info['id']
                category_id = segment_info['category_id']
                # category = categories_dict[category_id]
                rgb = (id % 256, (id // 256) % 256, id // (256 ** 2))
                # bgr = (id // (256 ** 2), (id // 256) % 256, id % 256)
                # print(bgr)
                annotation_image[np.where(np.all(lb == rgb, axis=-1))] = category_id
            Image.fromarray(annotation_image).save(lb_path.replace('.png','_L.png'))    
        
    out_line = []
    with open(f'{mode}_lb_info.txt', 'w') as fw:        
        for key, name in categories_dict.items():
            out_line.append("\"name\": \"{}\", \"id\": {}, \"trainId\": {}".format(name, key, key))
        fw.write('\n'.join(out_line))
            

    # for category in data['categories']:
    #     category_id = category['id']
    #     rgb = category['rgb']
    #     print(f"category_id: {category_id}, RGB: {rgb}")
