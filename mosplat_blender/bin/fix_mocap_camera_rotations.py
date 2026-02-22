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
from typing import Annotated, List, Optional, Tuple, TypeAlias

import torch
from dltype import BoolTensor, Float32Tensor, dltyped

ImagesTensor: TypeAlias = Annotated[torch.Tensor, Float32Tensor["S 3 H W"]]
ImagesAlphaTensor: TypeAlias = Annotated[torch.Tensor, Float32Tensor["S 1 H W"]]

CamMaskTensor: TypeAlias = Annotated[torch.Tensor, BoolTensor["S"]]

CAM_MASK: Optional[CamMaskTensor] = None


@dltyped()
def preprocess(
    frame_idx: int, media_files: List[Path], images: ImagesTensor
) -> Tuple[ImagesTensor, Optional[ImagesAlphaTensor]]:
    """
    This function is called once per frame per media file with the parameter values filled accordingly.

    Args:
        frame_idx: Frame index corresponding to the current `images` data.
        media_files: List of source media identifiers corresponding
            to images[i] (e.g. ["stream00.mp4", "stream01.mp4", ...]).
            Will have length of B (batch) dimension of `images` tensor.
        images: Raw image data as a torch tensor of shape (S, 3, H, W),
            where S (scene) is the number of media streams / cameras capturing each frame,
            and the 2nd dimension is color channels. RGB values are normalized to the range 0.0-1.0 and are the datatype `float32`. Lastly, H and W are directly the pixel size of the media files in the media directory.

    Returns:
        tuple: A tuple containing:
        - out_images: A Torch tensor with the same shape and dtype as `images`,
            containing the transformed image data.
        - images_alpha: An Torch tensor or `None` to optionally specify a mask of the relevant / target pixel data of the images. The mask tensor should be `torch.float32` data type, and range from 0.0-1.0.
    """

    global CAM_MASK

    if CAM_MASK is None:
        CAM_MASK = _create_camera_mask(media_files, images.device)

    images[CAM_MASK] = _rotate_180(images[CAM_MASK])  # rotate by 180 degrees
    return images, None


@dltyped()
def _rotate_180(x: ImagesTensor) -> ImagesTensor:
    return torch.flip(x, dims=(2, 3))


@dltyped()
def _create_camera_mask(
    media_filenames: List[Path], device: torch.device
) -> CamMaskTensor:
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
