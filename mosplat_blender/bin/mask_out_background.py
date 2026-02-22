from pathlib import Path
from typing import List, Optional, Tuple, TypeAlias

import torch
from jaxtyping import Bool, Float32
from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50

ImagesTensor: TypeAlias = Float32[torch.Tensor, "S 3 H W"]
ImagesAlphaTensor: TypeAlias = Float32[torch.Tensor, "S 1 H W"]

CamMaskTensor: TypeAlias = Bool[torch.Tensor, "S"]

CAM_MASK: Optional[CamMaskTensor] = None
PERSON_CLASS = 15

weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(pretrained=True, progress=False, weights=weights)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.eval().to(device)  # set to evaluate

COCO_DATASET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406], dtype=torch.float32, device=device
)[None, :, None, None]

COCO_DATASET_STD = torch.tensor(
    [0.229, 0.224, 0.225], dtype=torch.float32, device=device
)[None, :, None, None]


def preprocess(
    frame_idx: int, media_files: List[Path], images: ImagesTensor
) -> Tuple[ImagesTensor, Optional[ImagesAlphaTensor]]:
    global CAM_MASK

    if CAM_MASK is None:
        CAM_MASK = _create_camera_mask(media_files, images.device)

    images[CAM_MASK] = _rotate_180(images[CAM_MASK])  # rotate by 180 degrees

    person_mask = _create_person_class_mask(images)

    return images, person_mask


@torch.no_grad()
def _create_person_class_mask(
    images: ImagesTensor,
) -> ImagesAlphaTensor:
    normalized = normalize_to_coco_dataset(images)

    # model has 21 classes in total
    outputs: Float32[torch.Tensor, "S 21 H W"] = model(normalized)["out"]

    # per-pixel, get the class which has the highest prediction likeliness
    class_map: ImagesAlphaTensor = outputs.argmax(dim=1, keepdim=True)

    # create a boolean mask of the pixel dimension
    person_mask: ImagesAlphaTensor = class_map == PERSON_CLASS

    return (person_mask).to(torch.float32)


def normalize_to_coco_dataset(images: ImagesTensor) -> ImagesTensor:
    return (images - COCO_DATASET_MEAN) / COCO_DATASET_STD


def _rotate_180(x: ImagesTensor) -> ImagesTensor:
    return torch.flip(x, dims=(2, 3))


def _create_camera_mask(
    media_filenames: List[Path], device: torch.device
) -> CamMaskTensor:
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
