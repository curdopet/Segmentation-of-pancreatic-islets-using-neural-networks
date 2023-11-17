import argparse
import os

from typing import Optional, Tuple

from evaluation.csv_generation.dataframe_operations import EvaluationDataframes
from evaluation.csv_generation.image_stats_calculation import ImageStatsCalculation
from utils.constants import ADJACENT_ISLETS_DIR, MIN_ISLET_SIZES, ModelType
from utils.log import Log, Severity
from utils.masks_loading import MaskLoading


def parse_input_arguments() -> Tuple[str, str, Optional[str], Optional[str], Optional[str], Optional[bool]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", help="path to folder where images & masks are stored")
    parser.add_argument("nn_type", help="type of neural network - valid values are 'semantic' or 'instance'")
    parser.add_argument("--nn_masks_path", help="path to folder where NN masks are stored", default=None)
    parser.add_argument("--px_file", help="path to file where pixel sizes are stored", required=False, default=None)
    parser.add_argument("--pkl_file", help="path to results.pkl file (required for instance segmentation models)",
                        required=False,
                        default=None)
    parser.add_argument("--only_adjacent_islets", help="whether to do report only for adjacent islets",
                        required=False,
                        default=False)

    args = parser.parse_args()
    px_file = os.path.join(args.data_root, "pixel-sizes.csv") if args.px_file is None else args.px_file
    return args.data_root, args.nn_masks_path, args.nn_type, px_file, args.pkl_file, args.only_adjacent_islets


def update_evaluation_dfs_with_image_stats(
    evaluation_dfs: EvaluationDataframes,
    image_stats_calculation: ImageStatsCalculation,
) -> EvaluationDataframes:
    evaluation_dfs.update_image_dataframe(image_stats_calculation.get_image_stats())
    evaluation_dfs.update_false_negative_islets_dataframe(image_stats_calculation.get_stats_for_fn_islets())
    evaluation_dfs.update_false_positive_islets_dataframe(image_stats_calculation.get_stats_for_fp_islets())
    evaluation_dfs.update_matched_islets_dataframe(image_stats_calculation.get_stats_for_matched_islets())
    evaluation_dfs.update_incorrectly_separated_islets_dataframe(
        image_stats_calculation.get_stats_for_incorrectly_separated_islets())

    return evaluation_dfs


if __name__ == "__main__":
    data_root, nn_masks_path, nn_type, px_file, pkl_file, only_adjacent_islets = parse_input_arguments()
    model_type = ModelType(nn_type)

    image_names = [f for f in next(os.walk(os.path.join(data_root, "inputs")))[2]
                   if "GT" not in f and not f.startswith(".") and not f.endswith(".csv") and not f.endswith(".ini")]
    n_images = len(image_names)

    logging = Log(total_images_cnt=n_images)
    mask_loading = MaskLoading(
        logging=logging,
        data_root=os.path.join(data_root, "masks/{}".format(ADJACENT_ISLETS_DIR) if only_adjacent_islets else "masks"),
        nn_masks_path=nn_masks_path,
        px_file=px_file,
        model_type=model_type,
        no_mask_ok=only_adjacent_islets,
    )

    for min_islet_size in MIN_ISLET_SIZES:
        logging.log(Severity.INFO, "Generating csvs for min. islet size {}.".format(min_islet_size))
        logging.reset_image_log()

        evaluation_dfs = EvaluationDataframes()

        for image_name in image_names:
            logging.update_per_image_log(image_name)

            image_info = mask_loading.get_image_info(image_name)
            if image_info is None:
                continue

            image_stats_calculation = ImageStatsCalculation(
                image_name=image_name,
                gt_mask=image_info.gt_masks.islet_mask,
                nn_mask=image_info.nn_masks.islet_mask if model_type == ModelType.SEMANTIC else None,
                um_per_px=image_info.um_per_px,
                min_islet_size=min_islet_size,
                pkl_file=pkl_file if model_type == ModelType.INSTANCE else None,
            )
            evaluation_dfs = update_evaluation_dfs_with_image_stats(evaluation_dfs, image_stats_calculation)

        evaluation_dfs.save_dataframes(
            os.path.join(data_root, "adjacent_islets_results" if only_adjacent_islets else "results"),
            min_islet_size=min_islet_size
        )
