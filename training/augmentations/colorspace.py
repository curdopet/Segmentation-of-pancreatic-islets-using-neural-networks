import cv2
import numpy as np
import random
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomSaturation:
    """Modifies saturation of an image.

    Required Keys:

    - img (np.uint8)

    Modified Keys:

    - img (np.uint8)

    Args:
        delta (float): Max relative change of the saturation value
        prob (float): The probability of performing RandomSaturation transformation
    """
    def __init__(self, delta: float, prob: float):
        self.delta = delta
        self.prob = prob

    def __call__(self, results):
        if random.random() < self.prob:
            img = results["img"]
            img = img.astype(np.float32)

            hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_image)
            s *= random.uniform(1 - self.delta, 1 + self.delta)
            s = s.clip(0, 1)
            merged = cv2.merge([h, s, v])

            results["img"] = cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)

        return results


@TRANSFORMS.register_module()
class RandomBrightness:
    """Modifies brightness of an image.

    Required Keys:

    - img (np.uint8)

    Modified Keys:

    - img (np.uint8)

    Args:
        delta (float): Max absolute change of the brightness value
        prob (float): The probability of performing RandomBrightness transformation
    """

    def __init__(self, delta: float, prob: float):
        self.delta = delta
        self.prob = prob

    def __call__(self, results):
        if random.random() < self.prob:
            img = results["img"]
            brightness = int(random.uniform(-self.delta, self.delta))
            results["img"] = cv2.convertScaleAbs(img, alpha=1.0, beta=brightness)

        return results


@TRANSFORMS.register_module()
class RandomContrast:
    """Modifies contrast of an image.

    Required Keys:

    - img (np.uint8)

    Modified Keys:

    - img (np.uint8)

    Args:
        delta (float): Max absolute change of the contrast value
        prob (float): The probability of performing RandomContrast transformation
    """

    def __init__(self, delta: float, prob: float):
        self.delta = delta
        self.prob = prob

    def __call__(self, results):
        if random.random() < self.prob:
            img = results["img"]
            contrast = 1.0 + random.uniform(-self.delta, self.delta)
            results["img"] = cv2.convertScaleAbs(img, alpha=contrast, beta=0)

        return results
