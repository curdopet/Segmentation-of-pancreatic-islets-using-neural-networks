import argparse
import os

import cv2
import json
import numpy as np
import pandas as pd

from progress.bar import Bar
from typing import Optional


DATA_GROUPS = ['training', 'validation', 'test']
ISLET_CLASS = 0
MIN_BBOX_HEIGHT = 3
MIN_BBOX_WIDTH = 3


def parse_input_arguments() -> (str, str, str, str):
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", help="path to folder where images & masks are stored")
    parser.add_argument("data_group", help=f"data group - one of {DATA_GROUPS}")
    parser.add_argument("dest_dir", help=f"path to folder, where generated json will be stored")
    parser.add_argument("labels_csv", help=f"path to file where labels are stored", default="labels.csv")

    args = parser.parse_args()
    return args.data_root, args.data_group, args.dest_dir, args.labels_csv


def get_initial_json_dict(data_group: str) -> dict:
    return {
        'info': {'description': f'{data_group}-data-islets'},
        'images': [],
        'annotations': [],
        'categories': [
            {
                'id': ISLET_CLASS,
                'name': 'islet'
            },
        ]
    }


def get_image_id_and_mask_name(image_name: str, labels_csv: str) -> (int, str):
    labels_df = pd.read_csv(labels_csv)
    row = labels_df[labels_df.example_path == image_name]
    return int(row.index.values[0]), row.mask_path.values[0]


def get_image_json_dict(image_id: int, data_root: str, image_name: str) -> dict:
    image = cv2.imread(os.path.join(data_root, "inputs", image_name), cv2.IMREAD_COLOR)
    return {
        'id': image_id,
        'width': image.shape[1],
        'height': image.shape[0],
        'file_name': image_name
    }


def get_mask(data_root: str, mask_name: str) -> np.array:
    return cv2.imread(os.path.join(data_root, "masks", mask_name), cv2.IMREAD_GRAYSCALE)


def get_islet_and_exo_masks(mask: np.array):
    islet_mask = mask.copy()
    islet_mask[islet_mask <= 192] = 0

    exo_mask = mask.copy()
    exo_mask[exo_mask > 192] = 0
    exo_mask[exo_mask > 50] = 1

    return islet_mask, exo_mask


def get_contour_mask(contour: np.array, mask: np.array, translated: bool = True) -> np.array:
    if translated:
        x, y, w, h = cv2.boundingRect(contour)
        submask_islets = mask[y: y + h, x: x + w]

        translated_contour = contour.copy()
        translated_contour[:, 0, 0] -= x
        translated_contour[:, 0, 1] -= y
    else:
        submask_islets = mask.copy()
        translated_contour = contour.copy()

    contour_mask = np.zeros_like(submask_islets)
    return cv2.drawContours(
        contour_mask,
        [translated_contour],
        -1,
        (255, 255, 255),
        thickness=cv2.FILLED,
    )


def get_contour_area(contour: np.array, mask: np.array) -> np.array:
    contour_mask = get_contour_mask(contour, mask)
    return int(np.sum(contour_mask)//255)


def get_contours(mask: np.array) -> np.array:
    islet_mask, exo_mask = get_islet_and_exo_masks(mask)

    *_, islet_contours, _ = cv2.findContours(islet_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    islet_contours = list(filter(lambda contour: len(contour) >= 5, islet_contours))

    return islet_contours


def get_islet_annotation_json_dict(contour: np.array, contour_id: int, image_id: int, mask: np.array) -> Optional[dict]:
    x, y, w, h = cv2.boundingRect(contour)
    if w < MIN_BBOX_WIDTH or h < MIN_BBOX_HEIGHT:
        return None

    return {
        'id': contour_id,
        'image_id': image_id,
        'iscrowd': 0,
        'category_id': ISLET_CLASS,
        'segmentation': [contour.flatten().tolist()],
        'bbox': [x, y, w, h],
        'area': get_contour_area(contour, mask)
    }


def is_image(file_name: str) -> bool:
    return file_name.endswith(".png") or file_name.endswith(".jpg") or \
           file_name.endswith(".bmp") or file_name.endswith(".tif")


def get_images_cnt(data_root: str) -> int:
    return len([f for f in os.listdir(data_root) if "GT" not in f and is_image(f)])


if __name__ == "__main__":
    data_root, data_group, dest_dir, labels_csv = parse_input_arguments()
    json_dict = get_initial_json_dict(data_group)
    
    annotation_id = 0

    with Bar('Loading', max=get_images_cnt(data_root + "/inputs"), fill='█', suffix='%(percent).1f%% - %(eta)ds') as bar:
        for image_name in next(os.walk(data_root + "/inputs"))[2]:
            if "GT" in image_name or image_name.startswith(".") or not is_image(image_name):
                continue

            image_id, mask_name = get_image_id_and_mask_name(image_name, labels_csv)
            image_json_dict = get_image_json_dict(image_id, data_root, image_name)
            json_dict['images'].append(image_json_dict)

            mask = get_mask(data_root, mask_name)
            islet_contours = get_contours(mask)

            for contour in islet_contours:
                annotation_json_dict = get_islet_annotation_json_dict(contour, annotation_id, image_id, mask)
                if annotation_json_dict is None:
                    continue
                json_dict['annotations'].append(annotation_json_dict)
                annotation_id += 1

            bar.next()

    with open(os.path.join(dest_dir, f"coco-format-{data_group}-islets-only-gt{MIN_BBOX_HEIGHT-1}.json"), "w") as f:
        json.dump(json_dict, f)