# Prepare Datasets

A dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog)
for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc). You can download the dataset and put it in the root/datasets directory

## Expected dataset structure for [COCO](https://cocodataset.org/#download):

```
root/
  datasets/
    coco/
      annotations/
        panoptic_{train,val}2017.json
        panoptic_{train,val}2017/  # rgb png annotations
        coco_rgb_to_id.py # generate label_id annotations
      images/
        {train,val}2017/

```

run following code to generate label_id annotations:
```
cp coco_rgb_to_id.py coco/annotations/
cd coco/annotations/
python coco_rgb_to_id.py
```

## Expected dataset structure for [cityscapes](https://www.cityscapes-dataset.com/downloads/):
```
root/
  datasets/
    cityscapes/
      gtFine/
        train/
        val/
        test/
      leftImg8bit/
        train/
        val/
        test/
```
No preprocessing is needed

## Expected dataset structure for [ADE20k](http://sceneparsing.csail.mit.edu/):
```
root/
  datasets/
    ADEChallengeData2016/
      images/
      annotations/
      objectInfo150.txt
```

No preprocessing is needed

## Expected dataset structure for [Mapillary Vistas](https://www.mapillary.com/dataset/vistas):
```
root/
  datasets/
    mapi/
      {training,validation}/
        images/
        v1.2/
          labels/
    mapi_rgb_conver_to_lb.py

```

run following code to generate label_id annotations:
```
python mapi_rgb_conver_to_lb.py
```

## Expected dataset structure for [IDD](https://idd.insaan.iiit.ac.in/dataset/details/):
```
root/
  datasets/
    idd/
      gtFine/
        train/
        val/
      leftImg8bit/
        train/
        val/

```
No preprocessing is needed

## Expected dataset structure for [BDD](https://github.com/bdd100k/bdd100k):
```
root/
  datasets/
    bdd100k/
      seg/
        images/
          train/
          val/
        labels/
          train/
          val/

```
No preprocessing is needed


## Expected dataset structure for [SUN RGBD](https://rgbd.cs.princeton.edu/):
```
root/
  datasets/
    sunrgb/
      image/
        train/
        test/
      label38/
        train/
        test/

```
No preprocessing is needed