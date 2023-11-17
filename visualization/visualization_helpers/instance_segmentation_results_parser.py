import pickle
import random

import pycocotools.mask as mask_util

from typing import List, Optional, Tuple

from visualization.constants import ISLET_LABEL, EXO_LABEL, CONTOUR_COLORS_RGB
from visualization.custom_dataclasses.instance_segmentation_results import InstanceData, InstanceSegmentationResults


class InstanceSegmentationResultsParser:
    def __init__(self, results_file_path: str):
        with open(results_file_path, 'rb') as f:
            self.raw_data = pickle.load(f)

        self.results = list()

        self.parse_instance_segmentation_results()

    def parse_instance_segmentation_results(self):
        for image_results in self.raw_data:
            self.results.append(
                InstanceSegmentationResults(
                    image_name=image_results["img_path"].split('/')[-1],
                    islet_instances=self.parse_instances(ISLET_LABEL, image_results["pred_instances"]),
                    exo_instances=self.parse_instances(EXO_LABEL, image_results["pred_instances"]),
                )
            )

    def parse_instances(self, label: int, pred_instances: dict) -> List[InstanceData]:
        instance_data = list()

        num_instances = len(pred_instances['labels'])
        for instance_idx in range(num_instances):
            if pred_instances['labels'][instance_idx] == label:
                instance_data.append(
                    InstanceData(
                        bbox=pred_instances['bboxes'][instance_idx],
                        encoded_mask=pred_instances['masks'][instance_idx],
                        score=pred_instances['scores'][instance_idx],
                        color_bgr=self.get_random_instance_color_bgr()
                    )
                )

        return instance_data

    def get_islet_results_for_image(self, image_name) -> Optional[List[InstanceData]]:
        for result in self.results:
            if result.image_name == image_name:
                return result.islet_instances
        return None

    def get_raw_results_for_image(self, image_name):
        for result in self.raw_data:
            if result["img_path"].split('/')[-1] == image_name:
                return result
        return None

    @staticmethod
    def decode_mask(encoded_mask):
        return mask_util.decode(encoded_mask)

    @staticmethod
    def get_random_instance_color_bgr() -> Tuple[int, int, int]:
        color = random.choice(CONTOUR_COLORS_RGB)
        return int(color[2]), int(color[1]), int(color[0])
