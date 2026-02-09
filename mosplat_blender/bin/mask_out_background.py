from pathlib import Path
from typing import List, Optional, TypeAlias

import torch
from jaxtyping import Bool, Float
from torchvision.models.segmentation import fcn_resnet50

model = fcn_resnet50(pretrained=True, progress=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.eval().to(device)


COCO_DATASET_MEAN = [0.485, 0.456, 0.406]
COCO_DATASET_STD = [0.229, 0.224, 0.225]

ImagesTensorType: TypeAlias = Float[torch.Tensor, "B 3 H W"]
MaskTensorType: TypeAlias = Bool[torch.Tensor, "B"]

_IMAGES_MASK: Optional[MaskTensorType] = None


def preprocess(
    frame_idx: int, media_files: List[Path], images: ImagesTensorType
) -> ImagesTensorType:
    global _IMAGES_MASK

    if _IMAGES_MASK is None:
        _IMAGES_MASK = _create_camera_mask(media_files, images.device)

    images[_IMAGES_MASK] = _rotate_180(images[_IMAGES_MASK])  # rotate by 180 degrees

    return images


def _rotate_180(x: ImagesTensorType) -> ImagesTensorType:
    return torch.flip(x, dims=(2, 3))


def _create_camera_mask(
    media_filenames: List[Path], device: torch.device
) -> MaskTensorType:
    all_cam_indices = torch.tensor(
        [
            int(file.name.replace("stream", "").split(".")[0])
            for file in media_filenames
        ],
        device=device,
        dtype=torch.int8,
    )
    # only need to apply rotation to cameras of these specific indices
    rotation_indices = torch.tensor([3, 4, 5, 6, 7], device=device, dtype=torch.int8)
    return torch.isin(all_cam_indices, rotation_indices)
