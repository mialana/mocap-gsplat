"""
this file doubles as a useable preprocess script and example template that other
preprocess scripts can model off of.

to be used as a preprocess script, the module just needs to contain a function called
`preprocess` that shares the same signature as the one below.

the idea is to offer the ability to apply any necessary operations to the image data
through a contained snapshot of the pipeline.

for example, if frames 16-48 need additional explosure applied, simply branch on the
`frame_idx` parameter.

and as seen below, if edits needs to be applied per media file, use `media_files` and
your knowledge of what media files exist in the directory to apply transformations to
`images` with a single frame fittingly.

the specific transformations demonstrated in this file are specific to the CapturyLive
Motion Capture system used by the University of Pennsylvania's CG department.
the system contains 8 cameras, and when raw capture videos are exported, the videos
corresponding to the 4th, 5th, 6th, 7th, and 8th cameras have a 180 degree rotation
applied.
"""

from pathlib import Path
from typing import List, Optional, TypeAlias

import torch
from jaxtyping import Bool, Float32

ImagesTensorType: TypeAlias = Float32[torch.Tensor, "B 3 H W"]
MaskTensorType: TypeAlias = Bool[torch.Tensor, "B"]

_IMAGES_MASK: Optional[MaskTensorType] = None


def preprocess(
    frame_idx: int, media_files: List[Path], images: ImagesTensorType
) -> ImagesTensorType:
    """
    This function is called once per frame per media file with the parameter values filled accordingly.

    Args:
        frame_idx: Frame index corresponding to the current `images` data.
        media_files: List of source media identifiers corresponding
            to images[i] (e.g. ["stream00.mp4", "stream01.mp4", ...]).
            Will have length of B (batch) dimension of `images` tensor.
        images: Raw image data as a torch tensor of shape (B, 3, H, W),
            where B (batch) is the number of media streams / cameras,
            and the 2nd dimension are color channels. RGB values are normalized to the range 0.0-1.0 and are the data-type `float`. Lastly, H and W are directly the pixel size of the media files in the media directory.

    Returns:
        A Torch tensor array with the same shape and dtype as `images`,
        containing the transformed image data.
    """

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
    """creates a Torch boolean tensor for masking out cameras whose images need a 180
    degree transformation applied to them."""
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
