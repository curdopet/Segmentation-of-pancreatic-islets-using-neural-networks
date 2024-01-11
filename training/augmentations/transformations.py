import cv2
import numpy as np
import random

from mmdet.models.utils import mask2ndarray
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes
from mmdet.structures.mask import BitmapMasks

MIN_BBOX_HEIGHT = 3
MIN_BBOX_WIDTH = 3


@TRANSFORMS.register_module()
class Rotation:
    """Randomly rotates the image and pads by zeros.

    Required Keys:

    - img (np.uint8)
    - gt_bboxes (optional)
    - gt_masks (optional)
    - gt_ignore_flags (optional)

    Modified Keys:

    - img (np.uint8)
    - gt_bboxes (optional)
    - gt_masks (optional)
    - gt_ignore_flags (optional)

    Args:
        max_angle (float): Max angle of the rotation
        prob (float): The probability of performing Rotation transformation
    """
    def __init__(self, max_angle: float, prob: float):
        print("ROTATION INIT")
        self.max_angle = max_angle
        self.prob = prob

    def __call__(self, results):
        if random.random() < self.prob:
            img = results["img"]
            img = img.astype(np.float32)

            rotate_angle = int(random.uniform(-self.max_angle, self.max_angle))
            height, width = img.shape[0], img.shape[1]
            M = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), rotate_angle, 1)

            results["img"] = cv2.warpAffine(img, M, (width, height))

            num_masks = len(results['gt_masks'])
            masks = mask2ndarray(results['gt_masks'])
            masks_rotated = list()

            for i in range(num_masks):
                mask_rotated = cv2.warpAffine(masks[i], M, (width, height))
                *_, contours, _ = cv2.findContours(mask_rotated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours = list(filter(lambda contour: len(contour) >= 5, contours))
                masks_rotated.append(mask_rotated.astype(np.uint8))

                if len(contours) == 0:
                    results['gt_ignore_flags'][i] = True
                    continue

                x, y, w, h = cv2.boundingRect(contours[0])
                results['gt_bboxes'][i] = HorizontalBoxes([[x, y, x + w, y + h]])

                if w < MIN_BBOX_WIDTH or h < MIN_BBOX_HEIGHT:
                    results['gt_ignore_flags'][i] = True
                    continue

            results['gt_masks'] = BitmapMasks(np.array(masks_rotated), height, width)

        return results


@TRANSFORMS.register_module()
class Perspective:
    """Randomly changes perspective of the image and pads by zeros.

    Required Keys:

    - img (np.uint8)
    - gt_bboxes (optional)
    - gt_masks (optional)
    - gt_ignore_flags (optional)

    Modified Keys:

    - img (np.uint8)
    - gt_bboxes (optional)
    - gt_masks (optional)
    - gt_ignore_flags (optional)

    Args:
        max_perturb (float): Max perturbation of the perspective
        prob (float): The probability of performing Perspective transformation
    """
    def __init__(self, max_perturb: float, prob: float):
        self.max_perturb = max_perturb
        self.prob = prob

    def __call__(self, results):
        if random.random() < self.prob:
            img = results["img"]
            img = img.astype(np.float32)
            height, width = img.shape[0], img.shape[1]

            M = self.get_perturb_matrix(height, width)
            results["img"] = cv2.warpPerspective(img, M, (width, height))

            num_masks = len(results['gt_masks'])
            masks = mask2ndarray(results['gt_masks'])
            masks_perturbed = list()

            for i in range(num_masks):
                mask_perturbed = cv2.warpPerspective(masks[i], M, (width, height))
                *_, contours, _ = cv2.findContours(mask_perturbed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours = list(filter(lambda contour: len(contour) >= 5, contours))
                masks_perturbed.append(mask_perturbed.astype(np.uint8))

                if len(contours) == 0:
                    results['gt_ignore_flags'][i] = True
                    continue

                x, y, w, h = cv2.boundingRect(contours[0])
                results['gt_bboxes'][i] = HorizontalBoxes([[x, y, x + w, y + h]])

                if w < MIN_BBOX_WIDTH or h < MIN_BBOX_HEIGHT:
                    results['gt_ignore_flags'][i] = True
                    continue

            results['gt_masks'] = BitmapMasks(np.array(masks_perturbed), height, width)

        return results

    def get_perturb_matrix(self, height, width):
        img_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')
        dst_corners = np.array([[self.get_perturbed_corner(width), self.get_perturbed_corner(height)],
                                [self.get_perturbed_corner(width) + width, self.get_perturbed_corner(height)],
                                [self.get_perturbed_corner(width) + width,
                                 self.get_perturbed_corner(height) + height],
                                [self.get_perturbed_corner(width), self.get_perturbed_corner(height) + height]],
                               dtype='float32')
        return cv2.getPerspectiveTransform(img_corners, dst_corners)

    def get_perturbed_corner(self, value):
        return int(random.uniform(-self.max_perturb * value, self.max_perturb * value))


@TRANSFORMS.register_module()
class Stretch:
    """Randomly stretches the image and pads by zeros.

    Required Keys:

    - img (np.uint8)
    - gt_bboxes (optional)
    - gt_masks (optional)
    - gt_ignore_flags (optional)

    Modified Keys:

    - img (np.uint8)
    - gt_bboxes (optional)
    - gt_masks (optional)
    - gt_ignore_flags (optional)

    Args:
        max_stretch (float): Max relative stretch of the image
        prob (float): The probability of performing Stretch transformation
    """
    def __init__(self, max_stretch: float, prob: float):
        self.max_stretch = max_stretch
        self.prob = prob

    def __call__(self, results):
        if random.random() < self.prob:
            img = results["img"]
            img = img.astype(np.float32)
            self.original_height, self.original_width = img.shape[0], img.shape[1]

            #print("Calculating stretch parameters ...")
            stretch_factor_horizontal = random.uniform(1 - self.max_stretch, 1 + self.max_stretch)
            stretch_factor_vertical = random.uniform(1 - self.max_stretch, 1 + self.max_stretch)
            new_width = self.original_width * stretch_factor_horizontal
            new_height = self.original_height * stretch_factor_vertical

            #print("Stretching image ...", stretch_factor_vertical, stretch_factor_horizontal)
            stretched_img = cv2.resize(img, (int(new_width), int(new_height)))
            results["img"] = self.pad(self.crop(stretched_img))

            #print("Counting masks ...")
            num_masks = len(results['gt_masks'])
            masks = mask2ndarray(results['gt_masks'])
            masks_stretched = list()

            for i in range(num_masks):
                #print("Stretching mask ...")
                mask_stretched = cv2.resize(masks[i], (int(new_width), int(new_height)))
                mask_cropped_padded = self.pad(self.crop(mask_stretched))

                #print("Finding contours ...")
                *_, contours, _ = cv2.findContours(mask_cropped_padded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours = list(filter(lambda contour: len(contour) >= 5, contours))
                masks_stretched.append(mask_cropped_padded.astype(np.uint8))

                if len(contours) == 0:
                    #print("NO CONTOUR")
                    results['gt_ignore_flags'][i] = True
                    continue

                #print("Calculating bounding rectangle ...")
                x, y, w, h = cv2.boundingRect(contours[0])
                results['gt_bboxes'][i] = HorizontalBoxes([[x, y, x + w, y + h]])

                if w < MIN_BBOX_WIDTH or h < MIN_BBOX_HEIGHT:
                    #print("BOUNDING RECTANGLE TOO SMALL")
                    results['gt_ignore_flags'][i] = True
                    continue

            results['gt_masks'] = BitmapMasks(np.array(masks_stretched), self.original_height, self.original_width)

        return results

    def crop(self, img):
        height, width = img.shape[0], img.shape[1]
        channels = img.shape[2] if len(img.shape) == 3 else 1

        if width == self.original_width and height == self.original_height:
            return img

        crop_horizontal = width - self.original_width if width > self.original_width else 0
        crop_vertical = height - self.original_height if height > self.original_height else 0
        crop_left = crop_horizontal // 2
        crop_top = crop_vertical // 2

        if channels > 1:
            return img[crop_vertical - crop_top:height - crop_top, crop_left:width - crop_horizontal + crop_left, :]
        else:
            return img[crop_vertical - crop_top:height - crop_top, crop_left:width - crop_horizontal + crop_left]

    def pad(self, img):
        height, width = img.shape[0], img.shape[1]
        pad_horizontal = max(0, self.original_width - width)
        pad_vertical = max(0, self.original_height - height)
        pad_left = pad_horizontal // 2
        pad_top = pad_vertical // 2
        return cv2.copyMakeBorder(img, pad_top, pad_vertical - pad_top, pad_left, pad_horizontal - pad_left,
                                  cv2.BORDER_CONSTANT)
