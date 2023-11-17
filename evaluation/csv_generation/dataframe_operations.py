import os
import numpy as np
import pandas as pd

from typing import List, Tuple

from evaluation.custom_dataclasses.stats import ImageStats, IsletStats, IsletPairStats

np.set_printoptions(threshold=np.inf)


class EvaluationDataframes:
    def __init__(self):
        self.image_df = self.init_image_dataframe()
        self.false_negative_islets_df = self.init_islet_dataframe()
        self.false_positive_islets_df = self.init_islet_dataframe()
        self.matched_islets_df = self.init_islet_pair_dataframe()
        self.incorrectly_separated_islets_df = self.init_islet_pair_dataframe()

    @staticmethod
    def init_islet_dataframe() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "image_name": pd.Series([], dtype=str),
                "islet_id": pd.Series([], dtype=int),
                "size_um": pd.Series([], dtype=float),
                "area_um2": pd.Series([], dtype=float),
                "volume_ellipse_ie": pd.Series([], dtype=float),
                "volume_ricordi_short_ie": pd.Series([], dtype=float),
                "instance_score": pd.Series([], dtype=float),
                "contour": pd.Series([], dtype=object),
            }
        )

    @staticmethod
    def init_islet_pair_dataframe() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "image_name": pd.Series([], dtype=str),
                "iou": pd.Series([], dtype=float),
                "islet_id_nn": pd.Series([], dtype=int),
                "size_um_nn": pd.Series([], dtype=float),
                "area_um2_nn": pd.Series([], dtype=float),
                "volume_ellipse_ie_nn": pd.Series([], dtype=float),
                "volume_ricordi_short_ie_nn": pd.Series([], dtype=float),
                "contour_nn": pd.Series([], dtype=object),
                "islet_id_gt": pd.Series([], dtype=int),
                "size_um_gt": pd.Series([], dtype=float),
                "area_um2_gt": pd.Series([], dtype=float),
                "volume_ellipse_ie_gt": pd.Series([], dtype=float),
                "volume_ricordi_short_ie_gt": pd.Series([], dtype=float),
                "instance_score": pd.Series([], dtype=float),
                "contour_gt": pd.Series([], dtype=object),
            }
        )

    @staticmethod
    def init_image_dataframe() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "image_name": pd.Series([], dtype=str),
                "dice_score": pd.Series([], dtype=float),
                "iou": pd.Series([], dtype=float),
                "precision": pd.Series([], dtype=float),
                "recall": pd.Series([], dtype=float),
                "islet_count_nn": pd.Series([], dtype=int),
                "total_area_um2_nn": pd.Series([], dtype=float),
                "total_volume_ellipse_ie_nn": pd.Series([], dtype=float),
                "total_volume_ricordi_short_ie_nn": pd.Series([], dtype=float),
                "islet_count_gt": pd.Series([], dtype=int),
                "total_area_um2_gt": pd.Series([], dtype=float),
                "total_volume_ellipse_ie_gt": pd.Series([], dtype=float),
                "total_volume_ricordi_short_ie_gt": pd.Series([], dtype=float),
                "false_negative_islets_count": pd.Series([], dtype=int),
                "false_positive_islets_count": pd.Series([], dtype=int),
                "matched_islets_count": pd.Series([], dtype=int),
                "incorrectly_separated_islets_count_nn": pd.Series([], dtype=int),
                "incorrectly_separated_islets_count_gt": pd.Series([], dtype=int),
            }
        )

    def update_image_dataframe(self, image_stats: ImageStats) -> pd.DataFrame:
        image_dict = {
            "image_name": image_stats.image_name,
            "dice_score": image_stats.metrics.dice_score,
            "iou": image_stats.metrics.iou,
            "precision": image_stats.metrics.precision,
            "recall": image_stats.metrics.recall,
            "islet_count_nn": image_stats.islets_nn.islet_count,
            "total_area_um2_nn": image_stats.islets_nn.total_area_um2,
            "total_volume_ellipse_ie_nn": image_stats.islets_nn.total_volume_ellipse_ie,
            "total_volume_ricordi_short_ie_nn": image_stats.islets_nn.total_volume_ricordi_short_ie,
            "islet_count_gt": image_stats.islets_gt.islet_count,
            "total_area_um2_gt": image_stats.islets_gt.total_area_um2,
            "total_volume_ellipse_ie_gt": image_stats.islets_gt.total_volume_ellipse_ie,
            "total_volume_ricordi_short_ie_gt": image_stats.islets_gt.total_volume_ricordi_short_ie,
            "false_negative_islets_count": image_stats.islet_types_counts.false_negative,
            "false_positive_islets_count": image_stats.islet_types_counts.false_positive,
            "matched_islets_count": image_stats.islet_types_counts.matched,
            "incorrectly_separated_islets_count_nn": image_stats.islet_types_counts.incorrectly_separated_nn,
            "incorrectly_separated_islets_count_gt": image_stats.islet_types_counts.incorrectly_separated_gt,
        }
        self.image_df = pd.concat([self.image_df, pd.DataFrame([image_dict])], ignore_index=True)

    def update_false_negative_islets_dataframe(self, islet_stats: List[IsletStats]):
        islet_df_from_stats = self.get_islet_dataframe_from_stats(islet_stats)
        self.false_negative_islets_df = pd.concat(
            [self.false_negative_islets_df, islet_df_from_stats],
            ignore_index=True
        )

    def update_false_positive_islets_dataframe(self, islet_stats: List[IsletStats]):
        islet_df_from_stats = self.get_islet_dataframe_from_stats(islet_stats)
        self.false_positive_islets_df = pd.concat(
            [self.false_positive_islets_df, islet_df_from_stats],
            ignore_index=True
        )

    def get_islet_dataframe_from_stats(self, islets_stats: List[IsletStats]) -> pd.DataFrame:
        islet_df = self.init_islet_dataframe()
        for islet_stats in islets_stats:
            islet_dict = {
                "image_name": islet_stats.image_name,
                "islet_id": islet_stats.id,
                "size_um": islet_stats.size_um,
                "area_um2": islet_stats.area_um2,
                "volume_ellipse_ie": islet_stats.volume_ellipse_ie,
                "volume_ricordi_short_ie": islet_stats.volume_ricordi_short_ie,
                "instance_score": islet_stats.instance_score,
                "contour": np.array2string(islet_stats.contour),
            }
            islet_df = pd.concat([islet_df, pd.DataFrame([islet_dict])], ignore_index=True)
        return islet_df

    def update_matched_islets_dataframe(self, islet_pairs_stats: List[IsletPairStats]):
        islet_pair_df_from_stats = self.get_islet_pair_dataframe(islet_pairs_stats)
        self.matched_islets_df = pd.concat(
            [self.matched_islets_df, islet_pair_df_from_stats],
            ignore_index=True
        )

    def update_incorrectly_separated_islets_dataframe(self, islet_pairs_stats: List[IsletPairStats]):
        islet_pair_df_from_stats = self.get_islet_pair_dataframe(islet_pairs_stats)
        self.incorrectly_separated_islets_df = pd.concat(
            [self.incorrectly_separated_islets_df, islet_pair_df_from_stats],
            ignore_index=True
        )

    def get_islet_pair_dataframe(self, islet_pairs_stats: List[IsletPairStats]) -> pd.DataFrame:
        islet_pair_df = self.init_islet_pair_dataframe()
        for islet_pair_stats in islet_pairs_stats:
            islet_pair_dict = {
                "image_name": islet_pair_stats.image_name,
                "iou": islet_pair_stats.iou,
                "islet_id_nn": islet_pair_stats.islet_nn.id,
                "size_um_nn": islet_pair_stats.islet_nn.size_um,
                "area_um2_nn": islet_pair_stats.islet_nn.area_um2,
                "volume_ellipse_ie_nn": islet_pair_stats.islet_nn.volume_ellipse_ie,
                "volume_ricordi_short_ie_nn": islet_pair_stats.islet_nn.volume_ricordi_short_ie,
                "instance_score": islet_pair_stats.islet_nn.instance_score,
                "contour_nn": islet_pair_stats.islet_nn.contour,
                "islet_id_gt": islet_pair_stats.islet_gt.id,
                "size_um_gt": islet_pair_stats.islet_gt.size_um,
                "area_um2_gt": islet_pair_stats.islet_gt.area_um2,
                "volume_ellipse_ie_gt": islet_pair_stats.islet_gt.volume_ellipse_ie,
                "volume_ricordi_short_ie_gt": islet_pair_stats.islet_gt.volume_ricordi_short_ie,
                "contour_gt": islet_pair_stats.islet_gt.contour,
            }
            islet_pair_df = pd.concat([islet_pair_df, pd.DataFrame([islet_pair_dict])], ignore_index=True)
        return islet_pair_df

    def save_dataframes(self, dest_dir: str, min_islet_size: int):
        os.makedirs(dest_dir, exist_ok=True)
        self.image_df.to_csv(
            os.path.join(
                dest_dir, "min{}-image-stats.csv".format(min_islet_size)
            ),
            index=None
        )
        self.false_negative_islets_df.to_csv(
            os.path.join(
                dest_dir, "min{}-false-negative-islets-stats.csv".format(min_islet_size)
            ),
            index=None
        )
        self.false_positive_islets_df.to_csv(
            os.path.join(
                dest_dir, "min{}-false-positive-islets-stats.csv".format(min_islet_size)
            ),
            index=None
        )
        self.matched_islets_df.to_csv(
            os.path.join(
                dest_dir, "min{}-matched-islets-stats.csv".format(min_islet_size)
            ),
            index=None
        )
        self.incorrectly_separated_islets_df.to_csv(
            os.path.join(
                dest_dir, "min{}-incorrectly-separated-islets-stats.csv".format(min_islet_size)
            ),
            index=None
        )
