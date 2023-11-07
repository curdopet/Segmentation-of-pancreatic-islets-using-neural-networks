import argparse
import os

from typing import Tuple

from utils.log import Log, Severity
from visualization.visualization_helpers.instance_segmentation_results_parser import InstanceSegmentationResultsParser
from visualization.visualization_helpers.instance_segmentation_results_visualizer import \
    InstanceSegmentationResultsVisualizer


def parse_input_arguments() -> Tuple[str, str, str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_images_dir", help="path to folder where input images are stored")
    parser.add_argument("gt_masks_dir", help="path to folder where GT masks are stored")
    parser.add_argument("results_dir", help="path to file where the visualization results will be stored")
    parser.add_argument("results_file", help="path to file where results in .pkl format are stored")

    args = parser.parse_args()
    return args.input_images_dir, args.gt_masks_dir, args.results_dir, args.results_file


if __name__ == "__main__":
    input_images_dir, gt_masks_dir, results_dir, results_file = parse_input_arguments()

    results_parser = InstanceSegmentationResultsParser(results_file)

    n_images = len(results_parser.results)
    logging = Log(total_images_cnt=n_images)

    for result in results_parser.results:
        logging.update_per_image_log(result.image_name)
        logging.per_image_log(Severity.INFO, "Generating visualizations ...")

        image_path = os.path.join(input_images_dir, result.image_name)

        results_visualizer = InstanceSegmentationResultsVisualizer(
            image_path=image_path,
            islet_instances=result.islet_instances,
            exo_instances=result.exo_instances,
            results_dir=results_dir,
            gt_mask_dir=gt_masks_dir,
        )
        results_visualizer.save_all_visualizations()

        logging.per_image_log(Severity.INFO, "Visualizations saved to {}".format(results_dir))
