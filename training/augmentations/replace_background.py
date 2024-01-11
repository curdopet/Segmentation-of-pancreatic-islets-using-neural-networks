import os
import cv2
import random

import numpy as np
from mmdet.models.utils import mask2ndarray
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ReplaceBackground:
    """Replaces background with an image.

    Required Keys:

    - img (np.uint8)
    - gt_bboxes (HorizontalBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)

    Modified Keys:

    - img (np.uint8)

    Args:
        background_images_path (str): A path to the directory with the images of background used for replacement
        prob (float): The probability of performing ReplaceBackground transformation
    """
    def __init__(self, background_images_path: str, prob: float):
        self.background_images_path = background_images_path
        self.prob = prob

        self.background_files = os.listdir(background_images_path)

    def __call__(self, results):
        if random.random() < self.prob:
            img = results["img"]
            mask = self.get_mask(results["img"], results['gt_masks'])
            results["img"] = self.replace_background(img, mask)

        return results

    @staticmethod
    def get_mask(img, gt_masks):
        mask = np.zeros(img.shape[:-1])

        masks = mask2ndarray(gt_masks)
        for gt_mask in masks:
            mask += gt_mask

        return mask.astype(np.float32)

    def replace_background(self, img, mask):
        # blur mask to make a smooth transition
        mask[mask == 1] = 255
        blur = cv2.blur(mask, (7, 7))

        # expand the mask to all three channels.
        mask_color = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

        # Select random background and resize it to the size of the augmented image.
        background_path = random.choice(self.background_files)
        background = cv2.imread(os.path.join(self.background_images_path, background_path), cv2.IMREAD_COLOR)

        background = cv2.resize(background, (mask.shape[1], mask.shape[0]))

        blend = (np.multiply(background / 255., 1. - mask_color / 255.) + np.multiply(img / 255.,
                                                                                      mask_color / 255.)) * 255
        return blend.astype(np.uint8)
