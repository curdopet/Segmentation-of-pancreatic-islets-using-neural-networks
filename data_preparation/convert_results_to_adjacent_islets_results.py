import argparse
import cv2
import os
import pickle

import numpy as np

from glob import glob
from progress.bar import Bar
from typing import List, Optional, Tuple

from evaluation.csv_generation.image_stats_calculation import ImageStatsCalculation
from utils.checks import is_image
from utils.constants import ADJACENT_ISLETS_DIR
from visualization.custom_dataclasses.instance_segmentation_results import InstanceData
from visualization.visualization_helpers.instance_segmentation_results_parser import InstanceSegmentationResultsParser


def parse_input_arguments() -> Tuple[str, Optional[str], Optional[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", help="path to folder where images & masks are stored")
    parser.add_argument("--nn_masks_path", help="path to folder where NN masks are stored", default=None)
    parser.add_argument("--pkl_file", help="path to results.pkl file (required for instance segmentation models)",
                        required=False,
                        default=None)

    args = parser.parse_args()
    return args.data_root, args.nn_masks_path, args.pkl_file


def get_islets_mask(dir: str, image_name: str, suffix: str = "GT") -> np.array:
    mask_paths = glob("{}*{}*".format(os.path.join(dir, image_name[:-4]), suffix))
    assert len(mask_paths) <= 1

    if len(mask_paths) == 0:  # all mask do not contain adjacent islets
        return None

    mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)

    islet_mask = mask.copy()
    islet_mask[islet_mask <= 192] = 0

    return islet_mask


def get_nn_mask_on_adjacent_islets(gt_mask: np.array, nn_mask: np.array) -> np.array:
    image_stats_calculation = ImageStatsCalculation(
        image_name="",
        gt_mask=gt_mask,
        nn_mask=nn_mask,
        um_per_px=1.5,
        min_islet_size=0,
        pkl_file=None,
    )

    adjacent_islets_mask = np.zeros((*nn_mask.shape, 3), dtype=np.uint8)

    for islet_pair in image_stats_calculation.get_stats_for_matched_islets():
        gt_islet = islet_pair.islet_gt
        cv2.drawContours(adjacent_islets_mask, [gt_islet.contour], -1, (255, 255, 255), cv2.FILLED)

    return adjacent_islets_mask


def get_adjacent_islets_results(
        gt_mask: np.array,
        islet_results: List[InstanceData],
        instance_results_parser: InstanceSegmentationResultsParser
) -> List[InstanceData]:
    adjacent_islets_results = list()

    for islet_result in islet_results:
        islet_mask = instance_results_parser.decode_mask(islet_result.encoded_mask)
        if (gt_mask * islet_mask).sum() > 0:
            adjacent_islets_results.append(islet_result)

    return adjacent_islets_results


def update_image_raw_data(image_raw_data: dict, adjacent_islets_results: List[InstanceData]) -> dict:
    num_adjacent_islets = len(adjacent_islets_results)

    image_raw_data["pred_instances"] = {
        "labels": np.zeros((num_adjacent_islets,), dtype=int),
        "bboxes": np.zeros((num_adjacent_islets, 4), dtype=float),
        "masks": np.empty((num_adjacent_islets,), dtype=object),
        "scores": np.zeros((num_adjacent_islets,), dtype=float),
    }

    for i in range(num_adjacent_islets):
        image_raw_data["pred_instances"]["bboxes"][i] = adjacent_islets_results[i].bbox
        image_raw_data["pred_instances"]["masks"][i] = adjacent_islets_results[i].encoded_mask
        image_raw_data["pred_instances"]["scores"][i] = adjacent_islets_results[i].score

    return image_raw_data


if __name__ == "__main__":
    data_root, nn_masks_path, pkl_file = parse_input_arguments()
    assert nn_masks_path is not None or pkl_file is not None

    image_names = [f for f in next(os.walk(os.path.join(data_root, "inputs")))[2] if "GT" not in f and is_image(f)]

    if nn_masks_path is not None:
        os.makedirs(os.path.join(nn_masks_path, ADJACENT_ISLETS_DIR), exist_ok=True)
    else:
        instance_results_parser = InstanceSegmentationResultsParser(pkl_file)
        adjacent_pkl_content = list()

    with Bar('Loading', max=len(image_names), fill='█',
             suffix='%(percent).1f%% - %(eta)ds') as bar:
        for image_name in image_names:
            gt_mask = get_islets_mask(os.path.join(data_root, "masks", ADJACENT_ISLETS_DIR), image_name)

            if gt_mask is None:  # all mask do not contain adjacent islets
                bar.next()
                continue

            if nn_masks_path is not None:
                nn_mask = get_islets_mask(nn_masks_path, image_name, "NN_islets")
                adjacent_islets_nn_mask = get_nn_mask_on_adjacent_islets(gt_mask, nn_mask)

                adjacent_islets_mask_name = "{}_NN_mask_adjacent_islets.png".format(image_name[:-4])
                cv2.imwrite(
                    os.path.join(nn_masks_path, ADJACENT_ISLETS_DIR, adjacent_islets_mask_name),
                    adjacent_islets_nn_mask,
                )
            else:
                islet_results = instance_results_parser.get_islet_results_for_image(image_name)
                adjacent_islets_results = get_adjacent_islets_results(gt_mask, islet_results, instance_results_parser)

                image_raw_data = instance_results_parser.get_raw_results_for_image(image_name)
                image_raw_data = update_image_raw_data(image_raw_data, adjacent_islets_results)

                adjacent_pkl_content.append(image_raw_data.copy())

            bar.next()

        if pkl_file is not None:
            with open("{}_adjacent_islets.pkl".format(pkl_file[:-4]), "wb") as f:
                pickle.dump(adjacent_pkl_content, f, protocol=pickle.HIGHEST_PROTOCOL)
