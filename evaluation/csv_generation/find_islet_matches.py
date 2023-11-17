import cv2
import numpy as np

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from evaluation.custom_dataclasses.islet_matches import IsletPair, IsletMatchTypes
from evaluation.custom_dataclasses.stats import IsletStats, IsletTypesCounts


class IsletMatching:
    def __init__(
            self,
            islets_gt_stats: List[IsletStats],
            islets_nn_stats: List[IsletStats],
            mask_shape: Tuple[int, int]
    ):
        self.islets_gt_stats = islets_gt_stats
        self.islets_nn_stats = islets_nn_stats
        self.gt_islets_cnt = len(islets_gt_stats)
        self.nn_islets_cnt = len(islets_nn_stats)

        self.mask_shape = mask_shape
        self.gt_canvas = self.get_gt_canvas()
        self.islet_matching_types = None

    def get_gt_canvas(self) -> list:
        num_gt_canvases = int(np.ceil(self.gt_islets_cnt / 256))
        gt_canvases = [np.zeros(self.mask_shape + (3,), dtype=np.uint8) for i in range(num_gt_canvases)]

        for gt_islet in self.islets_gt_stats:
            islet_id = gt_islet.id
            gt_canvas_id = int(np.floor(islet_id / 256))
            cv2.drawContours(gt_canvases[gt_canvas_id], [gt_islet.contour], -1, (gt_islet.id, 0, 0), cv2.FILLED)

        gt_canvas = np.zeros(self.mask_shape + (3,), dtype=int)
        for i in range(num_gt_canvases):
            int_canvas = gt_canvases[i].astype(int)
            int_canvas[int_canvas != 0] += i * 256
            gt_canvas[gt_canvas == 0] += int_canvas[gt_canvas == 0]

        return gt_canvas

    def find_matches(self) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
        gt_matches = defaultdict(set)
        nn_matches = defaultdict(set)

        for nn_islet in self.islets_nn_stats:
            nn_canvas = np.zeros(self.mask_shape + (3,), dtype=np.uint8)
            cv2.drawContours(nn_canvas, [nn_islet.contour], -1, (0, 255, 0), cv2.FILLED)

            overlaps = self.gt_canvas + nn_canvas

            for overlap in overlaps[np.logical_and(overlaps[:, :, 0] > 0, overlaps[:, :, 1] > 0)]:
                gt_idx, _, _ = overlap
                gt_matches[gt_idx].add(nn_islet.id)
                nn_matches[nn_islet.id].add(gt_idx)

        return gt_matches, nn_matches

    def calculate_islet_matching_types(self):
        gt_matches, nn_matches = self.find_matches()

        false_negative_islets = list()
        false_positive_islets = list()
        matched_islets = list()
        incorrectly_separated_islets = list()

        for gt_id, gt_match in gt_matches.items():
            if len(gt_match) == 1:
                nn_id = gt_match.pop()
                if len(nn_matches[nn_id]) == 1:
                    matched_islets.append(IsletPair(gt_islet_id=gt_id, nn_islet_id=nn_id))
                else:
                    incorrectly_separated_islets.append(IsletPair(gt_islet_id=gt_id, nn_islet_id=nn_id))
            else:
                for nn_id in gt_match:
                    incorrectly_separated_islets.append(IsletPair(gt_islet_id=gt_id, nn_islet_id=nn_id))

        for i in range(self.gt_islets_cnt):
            gt_id = i + 1
            if gt_id not in gt_matches.keys():
                false_negative_islets.append(gt_id)

        for i in range(self.nn_islets_cnt):
            nn_id = i + 1
            if nn_id not in nn_matches.keys():
                false_positive_islets.append(nn_id)

        self.islet_matching_types = IsletMatchTypes(
            false_negative_islets=false_negative_islets,
            false_positive_islets=false_positive_islets,
            matched_islets=matched_islets,
            incorrectly_separated=incorrectly_separated_islets,
        )

    def get_islet_matching_types(self) -> IsletMatchTypes:
        if self.islet_matching_types is None:
            self.calculate_islet_matching_types()
        return self.islet_matching_types

    def get_islet_types_counts(self) -> IsletTypesCounts:
        return IsletTypesCounts(
            false_negative=len(self.islet_matching_types.false_negative_islets),
            false_positive=len(self.islet_matching_types.false_positive_islets),
            matched=len(self.islet_matching_types.matched_islets),
            incorrectly_separated_gt=len(set(
                [islet_pair.gt_islet_id for islet_pair in self.islet_matching_types.incorrectly_separated]
            )),
            incorrectly_separated_nn=len(set(
                [islet_pair.nn_islet_id for islet_pair in self.islet_matching_types.incorrectly_separated]
            ))
        )
