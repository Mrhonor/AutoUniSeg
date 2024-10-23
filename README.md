# Automated Label Unification for Multi-Dataset Semantic Segmentation

### Features
* Automatically construct unified label space, enabling segmentation model to be trained simultaneously on multiple datasets.


## Installation

See [installation instructions](INSTALL.md).

## Preparing Datasets

See [Preparing Datasets](datasets/README.md).


### Training & Evaluation in Command Line

We have divided the training process into three distinct stages. *train_net_stage1.py* involves training the Multi-SegHead model and initializing the label mappings. In *train_net_stage2.py*, we alternately train the GNNs and the SegNet. Finally, in *train_net_stage3.py*, we train the last stage of the SegNet. Please select the appropriate configuration file from the *configs* directory and then execute the following training scripts in sequence.

```
python train_net_stage1.py \
  --config-file configs/7_datasets/train_stage1.yaml \
  --num-gpus 4 

python train_net_stage2.py \
  --config-file configs/7_datasets/train_stage2.yaml \
  --num-gpus 4 

python train_net_stage3.py \
  --config-file configs/7_datasets/train_final_stage.yaml \
  --num-gpus 4 
```

To evaluate a model's performance, use
```
python train_net_stage3.py \
  --config-file configs/7_datasets/train_final_stage.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

### Constructed Graph Visualized
Below is a visualization of the unified label space created on seven datasets.
![](img/Graph.png)
