import argparse
import cv2
import os

from progress.bar import Bar

from utils.checks import is_image
from utils.constants import ISLETS_ONLY_MASKS_DIR


def parse_input_arguments() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("masks_dir", help="path to folder where masks are stored")

    args = parser.parse_args()
    return args.masks_dir

def get_images_cnt(data_root: str) -> int:
    return len([f for f in os.listdir(data_root) if is_image(f)])


if __name__ == "__main__":
    masks_dir = parse_input_arguments()

    os.makedirs(os.path.join(masks_dir, ISLETS_ONLY_MASKS_DIR), exist_ok=True)

    with Bar('Loading', max=get_images_cnt(masks_dir), fill='█', suffix='%(percent).1f%% - %(eta)ds') as bar:
        for mask_name in [f for f in os.listdir(masks_dir) if f.endswith(".png")]:
            mask = cv2.imread(os.path.join(masks_dir, mask_name), cv2.IMREAD_GRAYSCALE)
            mask[mask < 192] = 0

            new_mask_name = mask_name.split("_Exo")[0] + ".png"
            cv2.imwrite(os.path.join(masks_dir, ISLETS_ONLY_MASKS_DIR, new_mask_name), mask)
            bar.next()
