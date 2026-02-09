from pathlib import Path
from typing import List, Optional, TypeAlias

import torch
from jaxtyping import Bool, Float
from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50

ImagesTensorType: TypeAlias = Float[torch.Tensor, "B 3 H W"]
SingleImageTensorType: TypeAlias = Float[torch.Tensor, "3 H W"]
SegMaskTensorType: TypeAlias = Bool[torch.Tensor, "B H W"]
MaskTensorType: TypeAlias = Bool[torch.Tensor, "B"]

_IMAGES_MASK: Optional[MaskTensorType] = None
_PERSON_CLASS = 15

weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(pretrained=True, progress=False, weights=weights)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.eval().to(device)  # set to evaluate

COCO_DATASET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406], dtype=torch.float, device=device
)[None, :, None, None]

COCO_DATASET_STD = torch.tensor(
    [0.229, 0.224, 0.225], dtype=torch.float, device=device
)[None, :, None, None]


def preprocess(
    frame_idx: int, media_files: List[Path], images: ImagesTensorType
) -> ImagesTensorType:
    global _IMAGES_MASK

    if _IMAGES_MASK is None:
        _IMAGES_MASK = _create_camera_mask(media_files, images.device)

    images[_IMAGES_MASK] = _rotate_180(images[_IMAGES_MASK])  # rotate by 180 degrees

    masked_images = _mask_person_class(images)

    return masked_images


@torch.no_grad()
def _mask_person_class(
    images: ImagesTensorType,
) -> ImagesTensorType:
    normalized = normalize_to_coco_dataset(images)
    outputs: Float[torch.Tensor, "B 21 H W"] = model(normalized)["out"]

    # per-pixel, get the class which has the highest prediction likeliness
    class_map: SegMaskTensorType = outputs.argmax(dim=1)

    # expand to create a boolean mask of the pixel dimension
    person_mask: SegMaskTensorType = class_map == _PERSON_CLASS
    unsqueezed: Float[torch.Tensor, "B 1 H W"] = person_mask.unsqueeze(dim=1)

    # apply mask
    return images * unsqueezed


def normalize_to_coco_dataset(images: ImagesTensorType) -> ImagesTensorType:
    return (images - COCO_DATASET_MEAN) / COCO_DATASET_STD


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
