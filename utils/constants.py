from enum import Enum

# volume units converting
NL_PER_IE = 1.767146
NL_PER_UM3 = 0.000001

# minimal islet size
MIN_ISLET_SIZES = [0, 50, 100, 150, 200]

# instance segmentation
INSTANCE_SCORE_THRESHOLD = 0.5
OVERLAPPING_INSTANCES_IOU_THRESHOLD = 0.4
MAX_OVERLAPPING_PARTITION_OF_ISLET = 0.7


class MaskType(Enum):
    GT = "GT"
    NN = "NN"


class ModelType(Enum):
    SEMANTIC = "semantic"
    INSTANCE = "instance"


ADJACENT_ISLETS_DIR = "adjacent_islets"
