import argparse
import os

import cv2
import json
import numpy as np
import pandas as pd

from itertools import groupby
from progress.bar import Bar


DATA_GROUPS = ['training', 'validation', 'test']
ISLET_CLASS = 0
EXO_CLASS = 1
SMALL_EXO_MAX_AREA_PX = 30


def parse_input_arguments() -> (str, str, str):
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", help="path to folder where images & masks are stored")
    parser.add_argument("data_group", help=f"data group - one of {DATA_GROUPS}")
    parser.add_argument("labels_csv", help=f"path to file where labels are stored", default="labels.csv")
    parser.add_argument(
        "remove_small_exo",
        help=f"whether to remove exo tissue of pixel area smaller than {SMALL_EXO_MAX_AREA_PX}",
        default="false",
    )   # we want the nn to focus on the larger segments

    args = parser.parse_args()
    return args.data_root, args.data_group, args.labels_csv, args.remove_small_exo


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
            {
                'id': EXO_CLASS,
                'name': 'exo'
            }
        ]
    }


def get_image_id_and_mask_name(image_name: str, labels_csv: str) -> (int, str):
    labels_df = pd.read_csv(labels_csv)
    row = labels_df[labels_df.example_path == image_name]
    return int(row.index.values[0]), row.mask_path.values[0]


def get_image_json_dict(image_id: int, data_root: str, image_name: str) -> dict:
    image = cv2.imread(os.path.join(data_root, image_name), cv2.IMREAD_COLOR)
    return {
        'id': image_id,
        'width': image.shape[1],
        'height': image.shape[0],
        'file_name': image_name
    }


def get_mask(data_root: str, mask_name: str) -> np.array:
    return cv2.imread(os.path.join(data_root, mask_name), cv2.IMREAD_GRAYSCALE)


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


def remove_small_exo_contours(contours: np.array, mask: np.array) -> np.array:
    exo_contours = list()
    for contour in contours:
        if get_contour_area(contour, mask) > SMALL_EXO_MAX_AREA_PX:
            exo_contours.append(contour)

    return exo_contours


def get_contours(mask: np.array, remove_small_exo: bool) -> (np.array, np.array):
    islet_mask, exo_mask = get_islet_and_exo_masks(mask)

    *_, islet_contours, _ = cv2.findContours(islet_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    islet_contours = list(filter(lambda contour: len(contour) >= 5, islet_contours))

    *_, exo_contours, _ = cv2.findContours(exo_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    exo_contours = list(filter(lambda contour: len(contour) >= 5, exo_contours))

    if remove_small_exo:
        exo_contours = remove_small_exo_contours(exo_contours, exo_mask)

    return islet_contours, exo_contours


def get_islet_annotation_json_dict(contour: np.array, contour_id: int, image_id: int, mask: np.array) -> dict:
    x, y, w, h = cv2.boundingRect(contour)

    return {
        'id': contour_id,
        'image_id': image_id,
        'iscrowd': 0,
        'category_id': ISLET_CLASS,
        'segmentation': [contour.flatten().tolist()],
        'bbox': [x, y, w, h],
        'area': get_contour_area(contour, mask)
    }


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def get_exo_annotation_json_dict(contour: np.array, contour_id: int, image_id: int, mask: np.array) -> dict:
    x, y, w, h = cv2.boundingRect(contour)
    contour_mask = get_contour_mask(contour, mask, translated=False)
    rle = binary_mask_to_rle(np.asfortranarray(contour_mask))

    return {
        'id': contour_id,
        'image_id': image_id,
        'iscrowd': 1,
        'category_id': EXO_CLASS,
        'segmentation': [rle.get('counts')],
        'bbox': [x, y, w, h],
        'area': int(np.sum(contour_mask)//255)
    }


def is_image(file_name: str) -> bool:
    return file_name.endswith(".png") or file_name.endswith(".jpg") or \
           file_name.endswith(".bmp") or file_name.endswith(".tif")


def get_images_cnt(data_root) -> int:
    return len([f for f in os.listdir(data_root) if "GT" not in f and is_image(f)])


if __name__ == "__main__":
    data_root, data_group, labels_csv, remove_small_exo = parse_input_arguments()
    json_dict = get_initial_json_dict(data_group)

    with Bar('Loading', max=get_images_cnt(data_root), fill='█', suffix='%(percent).1f%% - %(eta)ds') as bar:
        for image_name in next(os.walk(data_root))[2]:
            if "GT" in image_name or image_name.startswith(".") or not is_image(image_name):
                continue

            image_id, mask_name = get_image_id_and_mask_name(image_name, labels_csv)
            image_json_dict = get_image_json_dict(image_id, data_root, image_name)
            json_dict['images'].append(image_json_dict)

            mask = get_mask(data_root, mask_name)
            islet_contours, exo_contours = get_contours(mask, remove_small_exo)

            for contour_id, contour in enumerate(islet_contours):
                annotation_json_dict = get_islet_annotation_json_dict(contour, contour_id, image_id, mask)
                json_dict['annotations'].append(annotation_json_dict)

            for contour_id, contour in enumerate(exo_contours):
                annotation_json_dict = get_exo_annotation_json_dict(contour, len(islet_contours) + contour_id, image_id, mask)
                json_dict['annotations'].append(annotation_json_dict)

            bar.next()

    with open(f"coco-format-{data_group}.json", "w") as f:
        json.dump(json_dict, f)
