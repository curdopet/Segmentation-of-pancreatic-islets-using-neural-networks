from dataclasses import dataclass
from typing import List


@dataclass
class IsletPair:
    gt_islet_id: int
    nn_islet_id: int


@dataclass
class IsletMatchTypes:
    false_negative_islets: List[int]
    false_positive_islets: List[int]
    matched_islets: List[IsletPair]
    incorrectly_separated: List[IsletPair]
