import cv2
import numpy as np
import pycocotools.mask as mask_util

from typing import List, Optional, Tuple

from evaluation.csv_generation.find_islet_matches import IsletMatching
from evaluation.csv_generation.islet_stats_calculation import IsletStatsCalculation
from evaluation.custom_dataclasses.islet_matches import IsletPair
from evaluation.custom_dataclasses.stats import ImageStats, IsletGroupStats, IsletPairStats, IsletStats, SemanticMetrics
from utils.constants import INSTANCE_SCORE_THRESHOLD, OVERLAPPING_INSTANCES_IOU_THRESHOLD
from visualization.visualization_helpers.instance_segmentation_results_parser import InstanceSegmentationResultsParser


class ImageStatsCalculation:
    def __init__(
            self,
            image_name: str,
            gt_mask: np.array,
            nn_mask: Optional[np.array],
            um_per_px: int,
            min_islet_size: int,
            pkl_file: Optional[str],
    ):
        self.image_name = image_name
        self.gt_mask = gt_mask
        self.nn_mask = nn_mask
        self.um_per_px = um_per_px
        self.min_islet_size = min_islet_size
        self.pkl_file = pkl_file
        self.is_instance_segmentation = self.pkl_file is not None
        self.instance_annotations = None

        assert self.nn_mask is not None or self.pkl_file is not None

        self.gt_islet_stats = None
        self.nn_islet_stats = None

        self.islet_matching = None
        self.islet_match_types = None
        self.image_stats = None

        if self.pkl_file is not None:
            self.load_annotations_from_pkl()
        self.calculate_gt_and_nn_islet_stats()
        self.calculate_islet_match_types()
        self.calculate_image_stats()

    def load_annotations_from_pkl(self):
        results_parser = InstanceSegmentationResultsParser(self.pkl_file)
        for result in results_parser.results:
            if result.image_name == self.image_name:
                self.instance_annotations = result.islet_instances
                break
        assert self.instance_annotations is not None

    def calculate_gt_and_nn_islet_stats(self):
        islet_stats_gt_nn = list()

        gt_contours = self.get_contours_from_mask(self.gt_mask)
        gt_instance_scores = [None] * len(gt_contours)
        if not self.is_instance_segmentation:
            nn_contours = self.get_contours_from_mask(self.nn_mask)
            nn_instance_scores = [None] * len(nn_contours)
        else:
            nn_contours, self.nn_mask, nn_instance_scores = self.get_instances_contours_and_mask(self.gt_mask.shape)

        for contours, mask, instance_scores in [(gt_contours, self.gt_mask, gt_instance_scores), (nn_contours, self.nn_mask, nn_instance_scores)]:
            islet_id = 1
            islet_stats = []
            for contour, instance_score in zip(contours, instance_scores):
                islet_stats_calculation = IsletStatsCalculation(
                    contour=contour,
                    mask=mask,
                    um_per_px=self.um_per_px,
                    instance_score=instance_score,
                )

                if not islet_stats_calculation.is_islet_big(self.min_islet_size):
                    continue

                islet_stats.append(islet_stats_calculation.get_islet_stats(islet_id, self.image_name))
                islet_id += 1

            islet_stats_gt_nn.append(islet_stats)

        self.gt_islet_stats = islet_stats_gt_nn[0]
        self.nn_islet_stats = islet_stats_gt_nn[1]

    def get_instances_contours_and_mask(self, mask_shape: tuple) -> Tuple[np.array, np.array, np.array]:
        contours = list()
        mask = np.zeros(mask_shape)
        instance_scores = list()

        self.instance_annotations.sort(key=lambda x: x.score, reverse=True)

        for instance in self.instance_annotations:
            if instance.score < INSTANCE_SCORE_THRESHOLD:
                continue

            instance_mask = mask_util.decode(instance.encoded_mask)
            instance_contours = self.get_contours_from_mask(instance_mask)

            for contour in instance_contours:
                contours.append(contour)
                mask = cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
                instance_scores.append(instance.score.item())

        return contours, mask, instance_scores

    @staticmethod
    def get_contours_from_mask(mask: np.array) -> np.array:
        *_, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = list(filter(lambda c: len(c) >= 5, contours))
        return contours

    def calculate_islet_match_types(self):
        self.islet_matching = IsletMatching(self.gt_islet_stats, self.nn_islet_stats, self.gt_mask.shape)
        self.islet_match_types = self.islet_matching.get_islet_matching_types()

    def calculate_image_stats(self):
        self.image_stats = ImageStats(
            image_name=self.image_name,
            islets_nn=self.get_stats_for_islet_group(self.nn_islet_stats),
            islets_gt=self.get_stats_for_islet_group(self.gt_islet_stats),
            metrics=self.get_metrics_for_image(),
            islet_types_counts=self.islet_matching.get_islet_types_counts(),
        )

    def get_image_stats(self):
        return self.image_stats

    def get_metrics_for_image(self) -> SemanticMetrics:
        intersection = np.count_nonzero(np.logical_and(self.gt_mask, self.nn_mask))
        union = np.count_nonzero(np.logical_or(self.gt_mask, self.nn_mask))

        return SemanticMetrics(
            dice_score=2 * intersection / (intersection + union),  # 2TP / (2TP + FP + FN)
            iou=intersection / union,
            precision=intersection / np.count_nonzero(self.nn_mask) if np.count_nonzero(self.nn_mask) != 0 else np.nan,  # TP / (TP + FP)
            recall=intersection / np.count_nonzero(self.gt_mask),  # TP / (TP + FN)
        )

    @staticmethod
    def get_stats_for_islet_group(islets_stats: List[IsletStats]) -> IsletGroupStats:
        return IsletGroupStats(
            islet_count=len(islets_stats),
            total_area_um2=sum([islet_stats.area_um2 for islet_stats in islets_stats]),
            total_volume_ellipse_ie=sum([islet_stats.volume_ellipse_ie for islet_stats in islets_stats]),
            total_volume_ricordi_short_ie=sum([islet_stats.volume_ricordi_short_ie for islet_stats in islets_stats]),
        )

    @staticmethod
    def get_stats_for_fp_or_fn_islets(islets_stats: List[IsletStats], fp_or_fn_islets: List[int]) -> List[IsletStats]:
        islet_stats = []

        for islet in islets_stats:
            if islet.id not in fp_or_fn_islets:
                continue

            islet_stats.append(islet)

        return islet_stats

    def get_stats_for_islet_pairs(self, islet_pairs: List[IsletPair]) -> List[IsletPairStats]:
        islet_pair_stats = []

        for matched_islet_pair in islet_pairs:
            nn_id = matched_islet_pair.nn_islet_id
            gt_id = matched_islet_pair.gt_islet_id

            nn_islet = self.nn_islet_stats[nn_id - 1]
            gt_islet = self.gt_islet_stats[gt_id - 1]

            mask_gt_islet = np.zeros(self.gt_mask.shape + (3,), dtype=np.uint8)
            cv2.drawContours(mask_gt_islet, [gt_islet.contour], -1, (1, 0, 0), cv2.FILLED)

            mask_nn_islet = np.zeros(self.gt_mask.shape + (3,), dtype=np.uint8)
            cv2.drawContours(mask_nn_islet, [nn_islet.contour], -1, (1, 0, 0), cv2.FILLED)

            intersection = np.sum(np.logical_and(mask_gt_islet, mask_nn_islet))
            union = np.sum(np.logical_or(mask_gt_islet, mask_nn_islet))

            islet_pair_stats.append(
                IsletPairStats(
                    image_name=nn_islet.image_name,
                    iou=intersection / union,
                    islet_nn=nn_islet,
                    islet_gt=gt_islet
                )
            )

        return islet_pair_stats

    def get_stats_for_fp_islets(self) -> List[IsletStats]:
        return self.get_stats_for_fp_or_fn_islets(self.nn_islet_stats, self.islet_match_types.false_positive_islets)

    def get_stats_for_fn_islets(self) -> List[IsletStats]:
        return self.get_stats_for_fp_or_fn_islets(self.gt_islet_stats, self.islet_match_types.false_negative_islets)

    def get_stats_for_matched_islets(self) -> List[IsletPairStats]:
        return self.get_stats_for_islet_pairs(self.islet_match_types.matched_islets)

    def get_stats_for_incorrectly_separated_islets(self) -> List[IsletPairStats]:
        return self.get_stats_for_islet_pairs(self.islet_match_types.incorrectly_separated)
