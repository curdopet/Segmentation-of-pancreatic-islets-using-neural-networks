import argparse
import cv2
import pickle

import numpy as np
import pycocotools.mask as mask_util

from progress.bar import Bar
from typing import List

from utils.constants import INSTANCE_SCORE_THRESHOLD, MAX_OVERLAPPING_PARTITION_OF_ISLET, \
    OVERLAPPING_INSTANCES_IOU_THRESHOLD
from visualization.custom_dataclasses.instance_segmentation_results import InstanceData


def parse_input_arguments() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_file", help="path to results file in pkl format")

    args = parser.parse_args()
    return args.pkl_file


def parse_instances(label: int, pred_instances: dict) -> List[InstanceData]:
    instance_data = list()

    num_instances = len(pred_instances['labels'])
    for instance_idx in range(num_instances):
        if pred_instances['labels'][instance_idx] == label:
            instance_data.append(
                InstanceData(
                    bbox=pred_instances['bboxes'][instance_idx],
                    encoded_mask=pred_instances['masks'][instance_idx],
                    score=pred_instances['scores'][instance_idx],
                    color_bgr=(0, 0, 0)
                )
            )

    return instance_data


def get_contours_from_mask(mask: np.array) -> np.array:
    *_, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = list(filter(lambda c: len(c) >= 5, contours))
    return contours


if __name__ == "__main__":
    pkl_file = parse_input_arguments()

    with open(pkl_file, 'rb') as f:
        raw_data = pickle.load(f)

    with Bar('Loading', max=len(raw_data), fill='█', suffix='%(percent).1f%% - %(eta)ds') as bar:
        for i in range(len(raw_data)):
            mask = np.zeros(raw_data[i]["ori_shape"])
            contours = list()

            pred_instances = raw_data[i]["pred_instances"].copy()

            bboxes = list()
            masks = list()
            scores = list()

            instance_annotations = parse_instances(0, pred_instances)
            instance_annotations.sort(key=lambda x: x.score, reverse=True)

            for instance in instance_annotations:
                if instance.score < INSTANCE_SCORE_THRESHOLD:
                    continue

                instance_mask = mask_util.decode(instance.encoded_mask)
                instance_contours = get_contours_from_mask(instance_mask)

                for contour in instance_contours:
                    contour_mask = np.zeros(raw_data[i]["ori_shape"], dtype=np.uint8)
                    contour_mask = cv2.drawContours(contour_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

                    max_iou = 0.

                    if (contour_mask * mask).sum() > 0:
                        for chosen_contour in contours:
                            chosen_contour_mask = np.zeros(raw_data[i]["ori_shape"], dtype=np.uint8)
                            chosen_contour_mask = \
                                cv2.drawContours(chosen_contour_mask, [chosen_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

                            contour_pxs = np.count_nonzero(contour_mask)
                            chosen_contour_pxs = np.count_nonzero(chosen_contour_mask)

                            intersection = np.count_nonzero(np.logical_and(chosen_contour_mask, contour_mask))
                            union = np.count_nonzero(np.logical_or(chosen_contour_mask, contour_mask))

                            max_iou = max(max_iou, intersection/union)

                            if intersection/contour_pxs > MAX_OVERLAPPING_PARTITION_OF_ISLET \
                                    or intersection/chosen_contour_pxs > MAX_OVERLAPPING_PARTITION_OF_ISLET:
                                max_iou = np.inf
                                break

                    if max_iou <= OVERLAPPING_INSTANCES_IOU_THRESHOLD:
                        contours.append(contour)
                        mask = cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

                        bboxes.append(instance.bbox)
                        masks.append(mask_util.encode(np.asfortranarray(contour_mask)))
                        scores.append(instance.score)
            bar.next()

            raw_data[i]["pred_instances"] = {
                "labels": np.zeros((len(contours),), dtype=int),
                "bboxes": np.array(bboxes, dtype=float),
                "masks": np.array(masks, dtype=object),
                "scores": np.array(scores, dtype=float),
            }

    with open("{}_filtered.pkl".format(pkl_file[:-4]), "wb") as f:
        pickle.dump(raw_data, f, protocol=pickle.HIGHEST_PROTOCOL)
