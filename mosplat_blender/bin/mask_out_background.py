from pathlib import Path
from typing import List

import numpy as np
import torch
from torchvision.models.segmentation import fcn_resnet50

model = fcn_resnet50(pretrained=True, progress=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.eval().to(device)


COCO_DATASET_MEAN = [0.485, 0.456, 0.406]
COCO_DATASET_STD = [0.229, 0.224, 0.225]


def preprocess(
    frame_idx: int,
    media_files: List[Path],  # length C
    images: np.ndarray,  # (C, H, W, 3)
) -> np.ndarray:

    return images
