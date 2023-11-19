import cv2
import numpy as np
import os
import pycocotools.mask as mask_util

from glob import glob
from typing import List, Tuple

from visualization.constants import BBOX_SCORE_BG_COLOR, BBOX_SCORE_FONT, BBOX_SCORE_FONT_COLOR, BBOX_SCORE_SCALE, \
    BBOX_SCORE_THICKNESS, BBOX_THICKNESS, CONTOUR_THICKNESS, FALSE_NEGATIVE_COLOR, FALSE_POSITIVE_COLOR, \
    INSTANCE_MASK_OPACITY, MATCHED_COLOR, SCORE_THRESHOLD
from visualization.custom_dataclasses.instance_segmentation_results import InstanceData


class InstanceSegmentationResultsVisualizer:
    def __init__(
            self,
            image_path: str,
            islet_instances: List[InstanceData],
            exo_instances: List[InstanceData],
            results_dir: str,
            gt_mask_dir: str,
            visualize_exo: bool = False,
            show_instance_scores: bool = True,
    ):
        self.image_name = image_path.split('/')[-1]
        self.image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self.islet_instances = islet_instances
        self.exo_instances = exo_instances
        self.visualize_exo = visualize_exo
        self.show_instance_scores = show_instance_scores

        self.islet_instances_mask = self.draw_instances_masks(self.islet_instances)
        self.exo_instances_mask = self.draw_instances_masks(self.exo_instances) if self.visualize_exo else None

        self.gt_mask_dir = gt_mask_dir
        self.islet_gt_mask, self.exo_gt_mask = self.load_gt_mask_for_image()

    def load_gt_mask_for_image(self) -> Tuple[np.array, np.array]:
        mask_paths = glob("{}*GT*".format(os.path.join(self.gt_mask_dir, self.image_name[:-4])))
        gt_mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)

        islet_mask = (gt_mask > 192).astype(np.uint16) * 255
        exo_mask = (gt_mask > 64).astype(np.uint8) * 255
        exo_mask[gt_mask > 192] = 0

        return islet_mask, exo_mask

    def save_all_visualizations(self):
        self.save_instances_bboxes()
        self.save_instances_contours()
        self.save_instances_masks()
        self.save_nn_gt_diff()

    def save_instances_bboxes(self):
        bbox_image = self.draw_instances_bboxes(self.islet_instances)
        self.save_image(bbox_image, os.path.join(self.results_dir, "islet_instances"), "bboxes_islets")

        if self.visualize_exo:
            bbox_image = self.draw_instances_bboxes(self.exo_instances)
            self.save_image(bbox_image, os.path.join(self.results_dir, "exo_instances"), "bboxes_exo")

    def save_instances_contours(self):
        contour_image = self.draw_instances_contours(self.islet_instances)
        self.save_image(contour_image, os.path.join(self.results_dir, "islet_instances"), "contours_islets")

        if self.visualize_exo:
            contour_image = self.draw_instances_contours(self.exo_instances)
            self.save_image(contour_image, os.path.join(self.results_dir, "exo_instances"), "contours_exo")

    def save_instances_masks(self):
        self.save_image(self.islet_instances_mask, os.path.join(self.results_dir, "islet_instances"), "masks_nn_islets")

        if self.visualize_exo:
            self.save_image(self.exo_instances_mask, os.path.join(self.results_dir, "exo_instances"), "masks_nn_exo")

    def save_nn_gt_diff(self):
        nn_gt_mask_diff = self.get_nn_gt_mask_diff(self.islet_instances_mask, self.islet_gt_mask)
        self.save_image(nn_gt_mask_diff, os.path.join(self.results_dir, "islet_instances"), "nn_gt_mask_diff_islets")

        if self.visualize_exo:
            nn_gt_mask_diff = self.get_nn_gt_mask_diff(self.exo_instances_mask, self.exo_gt_mask)
            self.save_image(nn_gt_mask_diff, os.path.join(self.results_dir, "exo_instances"), "nn_gt_mask_diff_exo")

    def draw_instances_bboxes(self, instances: List[InstanceData]) -> np.array:
        bbox_image = self.image.copy()

        for instance in instances:
            if instance.score < SCORE_THRESHOLD:
                continue

            x0, y0, x1, y1 = [int(i) for i in instance.bbox]
            bbox_image = cv2.rectangle(bbox_image, (x0, y0), (x1, y1), instance.color_bgr, BBOX_THICKNESS)
            if self.show_instance_scores:
                bbox_image = self.draw_bbox_score_label(bbox_image, instance.score, x0, y0)

        return bbox_image

    def draw_instances_contours(self, instances: List[InstanceData]) -> np.array:
        contour_image = self.image.copy()

        for instance in instances:
            if instance.score < SCORE_THRESHOLD:
                continue

            contours = self.get_instance_contours(instance.encoded_mask)
            contour_image = cv2.drawContours(contour_image, contours, -1, instance.color_bgr, CONTOUR_THICKNESS)

        return contour_image

    def draw_instances_masks(self, instances: List[InstanceData]) -> np.array:
        instances_mask = np.zeros_like(self.image, dtype=np.uint8)

        for instance in instances:
            if instance.score < SCORE_THRESHOLD:
                continue

            contours = self.get_instance_contours(instance.encoded_mask)

            bg = instances_mask.copy()
            bg = cv2.drawContours(bg, contours, -1, instance.color_bgr, cv2.FILLED)

            instances_mask = cv2.addWeighted(instances_mask, 1 - INSTANCE_MASK_OPACITY, bg, INSTANCE_MASK_OPACITY, 0)
            instances_mask = cv2.drawContours(instances_mask, contours, -1, instance.color_bgr, CONTOUR_THICKNESS)

        return instances_mask

    @staticmethod
    def get_nn_gt_mask_diff(nn_mask: np.array, gt_mask: np.array) -> np.array:
        nn_mask = nn_mask.copy()
        nn_mask[np.any(nn_mask > 0, axis=-1)] = 255
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2RGB).astype(int)

        union = gt_mask * nn_mask
        union = (union > 10).astype(np.uint16) * 255
        union[np.all(union == (255, 255, 255), axis=-1)] = MATCHED_COLOR

        fp = (nn_mask - gt_mask).astype(np.uint16)
        fp = (fp == 255).astype(np.uint16) * 255
        fp[np.all(fp == (255, 255, 255), axis=-1)] = FALSE_POSITIVE_COLOR

        fn = (gt_mask - nn_mask).astype(np.uint16)
        fn = (fn == 255).astype(np.uint16) * 255
        fn[np.all(fn == (255, 255, 255), axis=-1)] = FALSE_NEGATIVE_COLOR

        return (union + fp + fn).astype(np.uint8)

    def save_image(self, image_to_save: np.array, dest_dir: str, suffix: str):
        os.makedirs(dest_dir, exist_ok=True)
        file_name = "{}_{}.png".format(self.image_name[:-4], suffix)
        file_path = os.path.join(dest_dir, file_name)
        cv2.imwrite(file_path, image_to_save)

    @staticmethod
    def draw_bbox_score_label(bbox_image: np.array, score: float, x: int, y: int) -> np.array:
        text = "Score: {:.3f}".format(score)

        text_size, _ = cv2.getTextSize(text, BBOX_SCORE_FONT, BBOX_SCORE_SCALE, BBOX_SCORE_THICKNESS)
        text_w, text_h = text_size

        bbox_image = cv2.rectangle(bbox_image, (x, y - text_h - 2), (x + text_w, y - 1), BBOX_SCORE_BG_COLOR, cv2.FILLED)
        bbox_image = cv2.putText(bbox_image, text, (x, y - 2), BBOX_SCORE_FONT, BBOX_SCORE_SCALE,
                                 BBOX_SCORE_FONT_COLOR,
                                 BBOX_SCORE_THICKNESS)
        return bbox_image

    @staticmethod
    def get_instance_contours(encoded_mask: bytes) -> np.array:
        instance_mask = mask_util.decode(encoded_mask)
        *_, contours, _ = cv2.findContours(instance_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours

