"""
this file doubles as a useable preprocess script and example template that other
preprocess scripts can model off of.

to be used as a preprocess script, the module just needs to contain a function called
`preprocess` that shares the same signature as the one below.

the idea is to offer the ability to apply any necessary operations to the image
data through a contained snapshot of the pipeline.

for example, if frames 16-48 need additional explosure applied, simply branch on the
`frame_idx` parameter.

and as seen below, if edits needs to be applied per media file, use `media_file_names`
to apply transformations to `images` with a single frame fittingly.
"""

from typing import List, Optional

import numpy as np
import numpy.typing as npt

_IMAGES_MASK: Optional[npt.NDArray[np.bool_]] = None


def preprocess(
    frame_idx: int,
    media_file_names: List[str],  # length C
    images: np.ndarray,  # (C, H, W, 3)
) -> np.ndarray:
    """
    This function is called once per frame per media file with the parameter values filled accordingly.

    Args:
        frame_idx: Frame index corresponding to the current `images` data.
        media_file_names: List of source media identifiers corresponding
            to images[i] (e.g. ["stream00.mp4", "stream01.mp4", ...]).
        images: Raw image data as a NumPy array of shape (C, H, W, 3),
            where C is the number of media streams. The array is uint8 RGB.

    Returns:
        A NumPy array with the same shape and dtype as `images`,
        containing the transformed image data.
    """

    global _IMAGES_MASK

    if _IMAGES_MASK is None:
        _IMAGES_MASK = _build_mask(media_file_names)

    images[_IMAGES_MASK] = _rotate_180(images[_IMAGES_MASK])  # rotate by 180 degrees
    return images


def _rotate_180(x: np.ndarray) -> np.ndarray:
    return x[:, ::-1, ::-1, :]


def _build_mask(media_file_names: List[str]) -> npt.NDArray[np.bool_]:
    """creates a NumPy boolean array for masking out the images per frame that need to have a
    180 degree transformation applied to them."""
    cam_indices = np.array(
        [int(name.replace("stream", "").split(".")[0]) for name in media_file_names]
    )
    return np.isin(cam_indices, [3, 4, 5, 6, 7])
