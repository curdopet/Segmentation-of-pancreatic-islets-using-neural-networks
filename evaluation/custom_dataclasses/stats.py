from dataclasses import dataclass


@dataclass
class IsletStats:
    """.csv custom_dataclasses"""
    image_name: str
    id: int
    size_um: float
    area_um2: float
    volume_ellipse_ie: float
    volume_ricordi_short_ie: float
    instance_score: float
    contour: list = None


@dataclass
class IsletPairStats:
    """.csv custom_dataclasses"""
    image_name: str
    iou: float
    islet_gt: IsletStats
    islet_nn: IsletStats


@dataclass
class IsletGroupStats:
    islet_count: int
    total_area_um2: float
    total_volume_ellipse_ie: float
    total_volume_ricordi_short_ie: float


@dataclass
class SemanticMetrics:
    dice_score: float
    iou: float
    precision: float
    recall: float


@dataclass
class IsletTypesCounts:
    false_negative: int
    false_positive: int
    matched: int
    incorrectly_separated_nn: int
    incorrectly_separated_gt: int

@dataclass
class ImageStats:
    """.csv custom_dataclasses"""
    image_name: str
    islets_nn: IsletGroupStats
    islets_gt: IsletGroupStats
    metrics: SemanticMetrics
    islet_types_counts: IsletTypesCounts
