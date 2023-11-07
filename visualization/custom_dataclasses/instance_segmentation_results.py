import numpy as np

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class InstanceData:
    bbox: np.array
    encoded_mask: dict
    score: float
    color_bgr: Tuple[int, int, int]


@dataclass
class InstanceSegmentationResults:
    image_name: str
    islet_instances: List[InstanceData]
    exo_instances: List[InstanceData]
