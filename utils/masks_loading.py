import cv2
import os
import pandas as pd
import numpy as np

from glob import glob
from typing import Optional

from utils.constants import MaskType, ModelType
from utils.dataclasses.image_info import ImageInfo, ImageMasks
from utils.log import Log, Severity


def separate_islet_and_exo_masks(mask: np.array) -> ImageMasks:
    islet_mask = (mask > 192).astype(np.uint8) * 255
    exo_mask = (mask > 64).astype(np.uint8) * 255
    exo_mask[mask > 192] = 0

    return ImageMasks(islet_mask=islet_mask, exo_mask=exo_mask)


class MaskLoading:
    def __init__(
            self,
            logging: Log,
            data_root: str,
            nn_masks_path: Optional[str],
            px_file: str,
            model_type: ModelType,
            no_mask_ok: bool = False
    ):
        self.logging = logging
        self.data_root = data_root
        self.nn_masks_path = nn_masks_path
        self.px_file = px_file
        self.model_type = model_type
        self.no_mask_ok = no_mask_ok

        assert self.nn_masks_path is not None or self.model_type == ModelType.INSTANCE

    def load_mask_for_image(self, masks_root: str, image_name: str, mask_type: MaskType) -> ImageMasks:
        self.logging.per_image_log(Severity.DEBUG, f"Searching for masks...")

        mask_paths = glob("{}*{}*".format(os.path.join(masks_root, image_name[:-4]), mask_type.value))

        if len(mask_paths) == 0 and self.no_mask_ok:
            self.logging.per_image_log(Severity.ERROR, f"Mask file was not found")
            return None

        if len(mask_paths) == 0:
            self.logging.per_image_log(Severity.ERROR, f"Mask file was not found")
            raise FileNotFoundError
        elif len(mask_paths) > 1:
            self.logging.per_image_log(Severity.ERROR, f"Too many mask files ({len(mask_paths)}) were found: {mask_paths}.")
            raise AssertionError

        mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)

        if mask is None:
            self.logging.per_image_log(Severity.ERROR, f"Could not read mask {mask_paths[0]}")
            raise AssertionError

        self.logging.per_image_log(Severity.DEBUG, f"Mask found: {mask_paths[0]}")

        return separate_islet_and_exo_masks(mask)

    def get_pixel_size_for_image(self, image_name: str) -> float:
        pixel_sizes = pd.read_csv(self.px_file)
        return float(pixel_sizes.loc[pixel_sizes["image_name"] == image_name]["µm/px"].values[0])

    def get_image_info(self, image_name: str) -> ImageInfo:
        gt_masks = self.load_mask_for_image(self.data_root, image_name, MaskType.GT)
        if gt_masks is None:
            return None

        return ImageInfo(
            image_name=image_name,
            gt_masks=gt_masks,
            nn_masks=self.load_mask_for_image(
                self.nn_masks_path, image_name, MaskType.NN
            ) if self.model_type == ModelType.SEMANTIC else None,
            um_per_px=self.get_pixel_size_for_image(image_name)
        )
