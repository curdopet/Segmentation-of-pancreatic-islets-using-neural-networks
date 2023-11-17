import cv2
import numpy as np

from evaluation.custom_dataclasses.stats import IsletStats
from utils.constants import NL_PER_IE, NL_PER_UM3


class IsletStatsCalculation:
    def __init__(self, contour: np.array, mask: np.array, um_per_px: float, instance_score: float):
        self.contour = contour
        self.mask = mask
        self.um_per_px = um_per_px
        self.instance_score = instance_score

        self.ellipse_big_ax_px = None
        self.ellipse_small_ax_px = None

        self.size_um = None
        self.area_um2 = None
        self.volume_ellipse_ie = None
        self.volume_ricordi_short_ie = None

        self.calculate_size_um()
        self.calculate_area_um2()
        self.calculate_volume_ellipse_ie()
        self.calculate_volume_ricordi_short_ie()

    def fit_ellipse(self):
        ellipse = cv2.fitEllipse(self.contour)
        (_, _), (ax1, ax2), _ = ellipse
        self.ellipse_big_ax_px = max(ax1, ax2)
        self.ellipse_small_ax_px = min(ax1, ax2)

    def calculate_size_um(self):
        if self.ellipse_big_ax_px is None or self.ellipse_small_ax_px is None:
            self.fit_ellipse()

        if self.ellipse_small_ax_px <= 0 or self.ellipse_big_ax_px <= 0:
            return

        mean_ax_ellipse_px = (self.ellipse_small_ax_px + self.ellipse_big_ax_px) / 2
        mean_ax_ellipse_um = mean_ax_ellipse_px * self.um_per_px

        self.size_um = mean_ax_ellipse_um

    def calculate_area_um2(self):
        # find bounding rectangle
        x, y, w, h = cv2.boundingRect(self.contour)
        submask_islets = self.mask[y: y + h, x: x + w]

        # translate contour to fit within the bounding rectangle
        translated_contour = self.contour.copy()
        translated_contour[:, 0, 0] -= x
        translated_contour[:, 0, 1] -= y

        # draw contour in the bound rectangle
        contour_mask = np.zeros_like(submask_islets)
        contour_mask = cv2.drawContours(
            contour_mask,
            [translated_contour],
            -1,
            (255, 255, 255),
            thickness=cv2.FILLED,
        )

        # intersection of submask and contour_mask
        contour_mask = np.int32(contour_mask) * np.int32(submask_islets)
        contour_mask[contour_mask > 0] = 255
        contour_mask = np.uint8(contour_mask)

        self.area_um2 = np.count_nonzero(contour_mask) * self.um_per_px ** 2

    def calculate_volume_ellipse_ie(self):
        if self.ellipse_big_ax_px is None or self.ellipse_small_ax_px is None:
            self.fit_ellipse()

        big_ax_ellipse_um = self.ellipse_big_ax_px * self.um_per_px
        small_ax_ellipse_um = self.ellipse_small_ax_px * self.um_per_px

        volume_ellipse_um3 = (4 / 3) * (
                np.pi * big_ax_ellipse_um * small_ax_ellipse_um * small_ax_ellipse_um / 8
        )
        volume_ellipse_nl = volume_ellipse_um3 * NL_PER_UM3
        self.volume_ellipse_ie = volume_ellipse_nl / NL_PER_IE

    def calculate_volume_ricordi_short_ie(self):
        if self.area_um2 is None:
            self.calculate_area_um2()

        diameter_um = 2 * np.sqrt(self.area_um2 / np.pi)
        interval_start = int(diameter_um // 50 * 50)

        self.volume_ricordi_short_ie = 0
        if interval_start == 50:
            self.volume_ricordi_short_ie = 1.0 / 6.0
        elif interval_start == 100:
            self.volume_ricordi_short_ie = 1.0 / 1.5
        elif interval_start == 150:
            self.volume_ricordi_short_ie = 1.7
        elif interval_start == 200:
            self.volume_ricordi_short_ie = 3.5
        elif interval_start == 250:
            self.volume_ricordi_short_ie = 6.3
        elif interval_start == 300:
            self.volume_ricordi_short_ie = 10.4
        elif interval_start >= 350:
            self.volume_ricordi_short_ie = 15.8

    def is_islet_big(self, min_islet_size: int) -> bool:
        if self.size_um is None or self.size_um < min_islet_size:
            return False

        return self.volume_ellipse_ie > 0

    def get_islet_stats(self, islet_id: int, image_name: str) -> IsletStats:
        return IsletStats(
            image_name=image_name,
            id=islet_id,
            size_um=self.size_um,
            area_um2=self.area_um2,
            volume_ellipse_ie=self.volume_ellipse_ie,
            volume_ricordi_short_ie=self.volume_ricordi_short_ie,
            instance_score=self.instance_score,
            contour=self.contour,
        )
