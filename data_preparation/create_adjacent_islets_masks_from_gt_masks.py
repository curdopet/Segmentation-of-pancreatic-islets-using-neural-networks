import argparse
import cv2
import os

import numpy as np

from progress.bar import Bar
from typing import List

from evaluation.csv_generation.image_stats_calculation import ImageStatsCalculation
from evaluation.custom_dataclasses.stats import IsletPairStats, IsletStats
from utils.checks import is_image
from utils.constants import ADJACENT_ISLETS_DIR


def parse_input_arguments() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", help="path to folder where masks are stored")

    args = parser.parse_args()
    return args.data_root


def get_islet_mask(mask_path: str) -> np.array:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    islet_mask = mask.copy()
    islet_mask[islet_mask <= 192] = 0

    return islet_mask


def filter_adjacent_islets(islet_pairs: List[IsletPairStats]) -> List[IsletStats]:
    adjacent_islets = list()

    for islet_pair in islet_pairs:
        gt_islet = islet_pair.islet_gt

        if gt_islet not in adjacent_islets:
            adjacent_islets.append(gt_islet)

    return adjacent_islets


def get_adjacent_islets(gt_mask: np.array) -> List[IsletStats]:
    kernel = np.ones((5, 5), np.uint8)
    dilated_gt_mask = cv2.dilate(gt_mask, kernel, iterations=1)

    image_stats_calculation = ImageStatsCalculation(
        image_name="",
        gt_mask=gt_mask,
        nn_mask=dilated_gt_mask,
        um_per_px=1.5,
        min_islet_size=0,
        pkl_file=None,
    )
    return filter_adjacent_islets(image_stats_calculation.get_stats_for_incorrectly_separated_islets())


def draw_adjacent_islet(adjacent_islets: List[IsletStats], gt_mask: np.array) -> np.array:
    adjacent_islets_mask = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)

    for gt_islet in adjacent_islets:
        cv2.drawContours(adjacent_islets_mask, [gt_islet.contour], -1, (255, 255, 255), cv2.FILLED)

    return cv2.cvtColor(adjacent_islets_mask, cv2.COLOR_BGR2GRAY)


def get_adjacent_islets_mask(gt_mask: np.array) -> np.array:
    adjacent_islets = get_adjacent_islets(gt_mask)
    return draw_adjacent_islet(adjacent_islets, gt_mask)


def get_images_cnt(data_root: str) -> int:
    return len([f for f in os.listdir(data_root) if is_image(f)])


if __name__ == "__main__":
    data_root = parse_input_arguments()

    os.makedirs(os.path.join(data_root, ADJACENT_ISLETS_DIR), exist_ok=True)

    with Bar('Loading', max=get_images_cnt(data_root), fill='█',
             suffix='%(percent).1f%% - %(eta)ds') as bar:
        for mask_name in next(os.walk(data_root))[2]:
            if mask_name.startswith(".") or not is_image(mask_name):
                continue

            gt_mask = get_islet_mask(os.path.join(data_root, mask_name))
            adjacent_islets_mask = get_adjacent_islets_mask(gt_mask)

            if adjacent_islets_mask.sum() > 0:
                adjacent_islets_mask_name = "{}_adjacent_islets.png".format(mask_name[:-4])
                cv2.imwrite(
                    os.path.join(data_root, ADJACENT_ISLETS_DIR, adjacent_islets_mask_name),
                    adjacent_islets_mask,
                )

            bar.next()
