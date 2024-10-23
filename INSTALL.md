## Installation

### Requirements
- Linux with Python ≥ 3.9
- PyTorch ≥ 1.9 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- Download llama-2-7b-hf model from [huggingface](https://huggingface.co/meta-llama/Llama-2-7b-hf) and place it in the root directory
- `pip install -r requirements.txt`


### Example conda environment setup
```bash
conda create --name autouniseg python=3.10 -y
conda activate autouniseg
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -U opencv-python
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .

# under this directory
pip install -r requirements.txt

```

### Model checkpoint download

|  Model   | Cityscapes | Mapillary | SUNRGBD | BDD100k | IDD | ADE20K | COCO  | Download |
|  ----  | ----  | ----  |----  |----  |----  |----  |----  |----  |
| 7ds | 80.7  | 43.7 | 47.5 | 65.5 | 68.6 | 42.0 | 46.7 | [Google Drive](https://drive.google.com/file/d/1TajfQSvGVUSrOxpVT8Vgtc3HM8inlUFo/view?usp=sharing)
