import cv2
import numpy as np
import random
import torch

from mmdet.models.utils import mask2ndarray
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes
from mmdet.structures.mask import BitmapMasks

from augmentations.transformations import crop, pad

ISLETS_TO_ADD_RATIO = 1 / 2
MAX_ROTATE_ANGLE = 180
MAX_STRETCH = 0.25
MAX_PERSPECTIVE_TRANSFORM = 0.05

MIN_BBOX_HEIGHT = 3
MIN_BBOX_WIDTH = 3

ISLET_LABEL = 0


def get_max_ax_px(islet_mask):
    *_, contours, _ = cv2.findContours(islet_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = list(filter(lambda contour: len(contour) >= 5, contours))

    if len(contours) == 0:
        return None

    ellipse = cv2.fitEllipse(contours[0])
    (_, _), (ax1, ax2), _ = ellipse
    return max(ax1, ax2)


def get_islet_size_px(contour):
    if contour is None:
        return

    ellipse = cv2.fitEllipse(contour)
    (_, _), (ax1, ax2), _ = ellipse
    ellipse_big_ax_px = max(ax1, ax2)
    ellipse_small_ax_px = min(ax1, ax2)

    if ellipse_small_ax_px <= 0 or ellipse_big_ax_px <= 0:
        return

    return (ellipse_small_ax_px + ellipse_big_ax_px) / 2


def get_islet_submask(img, contour):
    # find bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    new_h, new_w = (3 * h, 3 * w)

    # translate contour to fit within the bounding rectangle
    translated_contour = contour.copy()
    translated_contour[:, 0, 0] -= x - w
    translated_contour[:, 0, 1] -= y - h

    # draw contour in the bound rectangle
    contour_mask = np.zeros((new_h, new_w)).astype(np.uint8)
    contour_mask = cv2.drawContours(
        contour_mask,
        [translated_contour],
        -1,
        (255, 255, 255),
        thickness=cv2.FILLED,
    )

    # intersection of submask and imh
    subimg = img[max(y - h, 0): min(y + 2 * h, img.shape[0] - 1),
             max(x - w, 0): min(x + 2 * w, img.shape[1] - 1)]
    binary_submask = contour_mask.copy()

    kernel = np.ones((3, 3), np.uint8)
    binary_submask = cv2.dilate(binary_submask, kernel, iterations=3)

    binary_submask = cv2.cvtColor(binary_submask, cv2.COLOR_GRAY2RGB)
    subimg = cv2.cvtColor(subimg, cv2.COLOR_BGR2RGB)

    binary_submask[binary_submask < 192] = 0
    binary_submask[binary_submask >= 192] = 1

    if binary_submask.shape[0] != subimg.shape[0] or binary_submask.shape[1] != subimg.shape[1]:
        subimg = pad(binary_submask.shape[0], binary_submask.shape[1],
                     crop(binary_submask.shape[0], binary_submask.shape[1], subimg))

    islet_img = np.uint8(binary_submask) * np.uint8(subimg)
    islet_img = cv2.cvtColor(islet_img, cv2.COLOR_RGB2BGR)

    return contour_mask, islet_img


def get_islet_contour(mask):
    *_, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = list(filter(lambda contour: len(contour) >= 5, contours))

    if len(contours) == 0:
        return None
    return contours[0]


def get_bbox_from_islet_mask(mask):
    contour = get_islet_contour(mask)
    if contour is None:
        return None, None, None, None

    x, y, w, h = cv2.boundingRect(contour)
    return x, y, w, h


@TRANSFORMS.register_module()
class IsletAddition:
    """Adds random islets < max islet size in pixels to the image.

    Required Keys:

    - img (np.uint8)
    - gt_bboxes (optional)
    - gt_masks (optional)
    - gt_ignore_flags (optional)
    - gt_bboxes_labels (optional)

    Modified Keys:

    - img (np.uint8)
    - gt_bboxes (optional)
    - gt_masks (optional)
    - gt_ignore_flags (optional)
    - gt_bboxes_labels (optional)

    Args:
        max_islet_size_px (float): Maximum islet size in pixels of an islet added to the image.
        prob (float): The probability of performing Islet addition
    """

    def __init__(self, max_islet_size_px: float, prob: float):
        self.max_islet_size_px = max_islet_size_px
        self.prob = prob

    def __call__(self, results):
        if random.random() < self.prob:
            img = results["img"]
            img = img.astype(np.float32)

            islet_candidates_indexes = self.get_islet_candidates_indices(results)
            masks = mask2ndarray(results['gt_masks'])

            islets_to_add_cnt = np.ceil(len(islet_candidates_indexes) * ISLETS_TO_ADD_RATIO).astype(int)
            islets_to_add_indices = random.sample(islet_candidates_indexes, k=islets_to_add_cnt)

            for i in islets_to_add_indices:
                islet_contour = get_islet_contour(masks[i])
                transformed_islet_img, transformed_islet_mask = self.transform_islet(img, islet_contour)
                results = self.add_islet_to_results(results, transformed_islet_img, transformed_islet_mask)

        return results

    def get_islet_candidates_indices(self, results):
        num_masks = len(results['gt_masks'])
        masks = mask2ndarray(results['gt_masks'])

        islet_candidates_indexes = list()

        for i in range(num_masks):
            contour = get_islet_contour(masks[i])

            islet_size_px = get_islet_size_px(contour)
            if islet_size_px is None or islet_size_px > self.max_islet_size_px:
                continue

            islet_candidates_indexes.append(i)

        return islet_candidates_indexes

    def transform_islet(self, img, islet_contour):
        islet_mask, islet_img = get_islet_submask(img, islet_contour)
        rotated_img, rotated_mask = self.rotate(islet_img, islet_mask)
        stretched_img, stretched_mask = self.stretch(rotated_img, rotated_mask)
        transformed_img, transformed_mask = self.perspective_transform(stretched_img, stretched_mask)
        return transformed_img, transformed_mask

    def add_islet_to_results(self, results, islet_img, islet_mask):
        height, width = results['img'].shape[0], results['img'].shape[1]
        islet_center = self.get_random_islet_center(results, islet_mask)

        new_mask = self.get_transformed_islet_mask(islet_center, height, width, islet_mask)
        x, y, w, h = get_bbox_from_islet_mask(new_mask)

        if w < MIN_BBOX_WIDTH or h < MIN_BBOX_HEIGHT:
            return results

        results['img'] = self.add_islet_to_img(results['img'], islet_center, islet_img, islet_mask)
        masks = list(mask2ndarray(results['gt_masks']))
        masks.append(np.array(new_mask))
        results['gt_masks'] = BitmapMasks(masks, height, width)
        gt_bboxes = results['gt_bboxes'].tensor
        gt_bboxes = torch.cat((gt_bboxes, torch.FloatTensor([[x, y, x + w, y + h]])), 0)
        results['gt_bboxes'] = HorizontalBoxes(gt_bboxes)
        ignore_flags = list(results['gt_ignore_flags'])
        ignore_flags.append(False)
        results['gt_ignore_flags'] = np.array(ignore_flags)
        bboxes_labels = list(results['gt_bboxes_labels'])
        bboxes_labels.append(ISLET_LABEL)
        results['gt_bboxes_labels'] = np.array(bboxes_labels)

        return results

    def get_random_islet_center(self, results, islet_mask):
        img = results['img']
        max_radius = get_max_ax_px(islet_mask) / 2
        semantic_mask = self.get_complete_semantic_mask(results)
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(semantic_mask, kernel, iterations=np.ceil(max_radius).astype(np.uint8))

        bg_pixels = np.argwhere(dilated_mask == 0)
        bg_pixels = list(filter(lambda i: int(1.5 * islet_mask.shape[0]) <= i[0] < img.shape[0] - int(
            1.5 * islet_mask.shape[0]) and \
                                          int(1.5 * islet_mask.shape[1]) <= i[1] < img.shape[1] - int(
            1.5 * islet_mask.shape[1]), bg_pixels))
        return random.choice(bg_pixels)

    def add_islet_to_img(self, img, islet_center, islet_img, islet_mask):
        (x1, x2), (y1, y2) = self.get_transformation_coordinates(islet_center, islet_img)
        blur = cv2.blur(islet_mask, (3, 3))
        alpha_s = blur / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            img[y1:y2, x1:x2, c] = (alpha_s * islet_img[:, :, c] + alpha_l * img[y1:y2, x1:x2, c])
        return img

    def get_transformed_islet_mask(self, islet_center, height, width, islet_mask):
        (x1, x2), (y1, y2) = self.get_transformation_coordinates(islet_center, islet_mask)
        alpha_s = islet_mask / 255.0
        alpha_l = 1.0 - alpha_s

        mask = np.zeros((height, width)).astype(np.uint8)
        mask[y1:y2, x1:x2] = (alpha_s * islet_mask + alpha_l * mask[y1:y2, x1:x2])
        return mask

    @staticmethod
    def get_transformation_coordinates(islet_center, img):
        x_offset = islet_center[1] - int(img.shape[1] / 2)
        y_offset = islet_center[0] - int(img.shape[0] / 2)

        x1, x2 = x_offset, x_offset + img.shape[1]
        y1, y2 = y_offset, y_offset + img.shape[0]

        return (x1, x2), (y1, y2)

    @staticmethod
    def get_complete_semantic_mask(results):
        semantic_mask = np.zeros((results['img'].shape[0], results['img'].shape[1])).astype(np.uint8)
        masks = mask2ndarray(results['gt_masks'])
        for mask in masks:
            semantic_mask += mask
        return semantic_mask

    @staticmethod
    def rotate(img, mask):
        rotate_angle = int(random.uniform(-MAX_ROTATE_ANGLE, MAX_ROTATE_ANGLE))
        height, width = mask.shape[0], mask.shape[1]
        M = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), rotate_angle, 1)

        return cv2.warpAffine(img, M, (width, height)), cv2.warpAffine(mask, M, (width, height))

    @staticmethod
    def stretch(img, mask):
        original_height, original_width = mask.shape[0], mask.shape[1]

        stretch_factor_horizontal = random.uniform(1 - MAX_STRETCH, 1 + MAX_STRETCH)
        stretch_factor_vertical = random.uniform(1 - MAX_STRETCH, 1 + MAX_STRETCH)
        new_width = original_width * stretch_factor_horizontal
        new_height = original_height * stretch_factor_vertical

        resized_img = cv2.resize(img, (int(new_width), int(new_height)))
        resized_mask = cv2.resize(mask, (int(new_width), int(new_height)))

        return pad(original_height, original_width, crop(original_height, original_width, resized_img)), \
               pad(original_height, original_width, crop(original_height, original_width, resized_mask)),

    @staticmethod
    def get_perturbed_corner(value):
        return int(random.uniform(-MAX_PERSPECTIVE_TRANSFORM * value, MAX_PERSPECTIVE_TRANSFORM * value))

    def get_perturb_matrix(self, height, width):
        img_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')
        dst_corners = np.array([[self.get_perturbed_corner(width), self.get_perturbed_corner(height)],
                                [self.get_perturbed_corner(width) + width, self.get_perturbed_corner(height)],
                                [self.get_perturbed_corner(width) + width, self.get_perturbed_corner(height) + height],
                                [self.get_perturbed_corner(width), self.get_perturbed_corner(height) + height]],
                               dtype='float32')
        return cv2.getPerspectiveTransform(img_corners, dst_corners)

    def perspective_transform(self, img, mask):
        height, width = mask.shape[0], mask.shape[1]
        M = self.get_perturb_matrix(height, width)
        return cv2.warpPerspective(img, M, (width, height)), cv2.warpPerspective(mask, M, (width, height))
