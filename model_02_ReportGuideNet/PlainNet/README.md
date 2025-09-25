# PlainNet (FLAIR 2D Binary Classification)

A minimal, **DDP-ready** PyTorch repo for slice-level binary classification on 2D FLAIR MRI.
This is an image-only baseline: no text encoder, no transformer decoder.

## 0) Environment (stable combo)
```bash
# You already built:
# torch==2.2.2+cu121, torchvision==0.17.2+cu121, numpy==1.26.4
pip install -r requirements.txt
```

## 1) Data CSV
Columns:
- `file_path` : image path (abs or relative to `data.root_dir`)
- `label`     : 0 or 1
- (optional) `split` : `train` / `val` / `test`

## 2) Config
Edit `configs/default.yaml`:
- `data.csv_path` : path/to/labels.csv
- `data.root_dir` : common prefix for relative paths
- `model.backbone`: `resnet18|resnet34`
- `train.pos_weight_auto`: handle class imbalance

## 3) Train (single server, multi-GPU)
```bash
bash scripts/train.sh
# Or explicitly:
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py --config configs/default.yaml
```

## 4) Evaluate
```bash
bash scripts/eval.sh runs/exp01/best.pth
```

## 5) Grad-CAM (optional)
```python
import torch
from PIL import Image
import torchvision.transforms as T
from models.plainnet import PlainNet
from cam.gradcam import GradCAM

device = "cuda"
model = PlainNet().to(device).eval()
# load checkpoint ...
cam = GradCAM(model)

im = Image.open("path/to/image.png").convert("L")
x = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize([0.5],[0.5])])(im).unsqueeze(0).to(device)
heatmap = cam(x)  # [H,W] in [0,1]
```
