import numpy as np

from dataclasses import dataclass


@dataclass
class ImageMasks:
    islet_mask: np.array
    exo_mask: np.array


@dataclass
class ImageInfo:
    image_name: str
    gt_masks: ImageMasks
    nn_masks: ImageMasks
    um_per_px: float
