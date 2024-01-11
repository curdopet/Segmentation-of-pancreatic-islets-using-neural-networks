# Segmentation of pancreatic islets and exocrine tissue from microscopic images using neural networks based approaches

## Prepare environment
Create conda environment:
```bash
conda env create -f environment.yml
conda activate islet-instance-segmentation
```
Instal MMEngine and MMCV and MMDetection
```bash
mim install mmengine
mim install "mmcv>=2.0.0"

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

## Training
Training is done using [MMDetection framework v3.2.0](https://github.com/open-mmlab/mmdetection).

### Preprocessing
Ground truths and masks for input data need to be converted into [COCO format](https://mmdetection.readthedocs.io/en/dev-3.x/advanced_guides/customize_dataset.html) by running:
```bash
pyhton -m data_preparation.generate_json_for_data <data_root> <data_group> <dest_dir> <labels_csv>
```
- `data_root` is path to a folder that contains two folders:
  - `inputs/` - contains input images
  - `masks/` - contains GT semantic masks
- `data_group` is either "training", "validation", or "test"
- `dest_dir` is path to a directory where to save the generated json file
- `labels.csv` path to .csv file that contains the list of input images and the associated masks

### Training
TODO

### Postprocessing
Instance segmentation results file `results.pkl` may contain some unwanted predictions that have to be filtered out:

- **Instances with low score**: As instances with low score are usually low quality, an `INSTANCE_SCORE_THRESHOLD` 
is defined in [utils/constants.py](utils/constants.py) to filter out all instances with score lower than the threshold
- **Overlapping instances**: Pancreatic islets in microscopic images can overlap, however, in instance segmentation, 
the same islets can be found multiple times as different instances. To handle this issue, two additional thresholds are 
defined to filter out overlapping instances:
  - `OVERLAPPING_INSTANCES_IOU_THRESHOLD` - maximum allowed IoU of two instances
  - `MAX_OVERLAPPING_PARTITION_OF_ISLET` - the maximum percentage of an instance pixel area that can overlap with 
  another instance (as IoU can be low, but the entire instance can be in another instance)

To filter out unwanted predictions, run:
```bash
python -m data_preparation.filter_out_overlapping_instances_from_results <pkl_file>
```
- `pkl_file` is a file with the results of instance segmentation in .pkl format

It creates a file with filtered results with name suffix `_filtered.pkl` in the same folder as the `pkl_file` is.

## Evaluation
Model evaluation process consists of two steps:
1. Generating csvs from GT and NN masks
2. Generating evaluation graphs

### Generating csvs from masks
Csvs can be generated using following command:
```bash
python -m evaluation.generate_csvs_for_report <data_root> <nn_type> <px_file> [--nn_masks_root <...>] [--pkl_file <...>] [--only_adjacent_islets <...>]
```
- `data_root` is path to a folder that contains two folders:
  - `inputs/` - contains input images
  - `masks/` - contains GT semantic masks
- `nn_type` - type of the neural network: `semantic` or `instance`
- `px_file` - path to a .csv file that contains the list of μm/px ratios for input images
- `--nn_masks_root` is a path to the directory with predicted NN masks (only for `nn_type == semantic`)
- `--pkl_file` is a file with the results of instance segmentation in .pkl format (only for `nn_type == instance`)
- `--only_adjacent_islets` whether to generate results only for adjacent islets - default is `false`, for more 
information see section [Evaluation only on adjacent islets](#adj-islets-eval)

It creates a folder `results` in `data_root` with all generated csvs that can be used for plotting the evaluation graphs.

### Plot graphs
There are two jupyter notebooks with evaluation graphs:
- [evaluation_summary.ipynb](evaluation/evaluation_summary.ipynb) contains few most important graph for the evaluation 
of model's performance in semantic/instance segmentation and the quality of islets separation. The notebook plots the 
results for a single model, or it can show comparison of two models. The paths to generated csvs has to be set in the 
notebook in the `user constants` section.
- [detailed_evaluation_graphs.ipynb](evaluation/detailed_evaluation_graphs.ipynb) contains many graphs that can be used
for a detailed evaluation and deeper understanding of the strengths and weaknesses of the evaluated model. The path
to generated csvs has to be set in the notebook in the `user constants` section.

<h3 id="adj-islets-eval">Evaluation only on adjacent islets</h3>

#### 1. Prepare GT data
To evaluate model only on adjacent islets, first, create GT masks that contains only adjacent islets by running:
```bash
python -m data_preparation.create_masks_for_adjacent_islets <data_root>
```
- `data_root` is path to a folder that contains only GT masks

It creates a new directory `adjacent_islets` within the `data_root` that contains GT masks only with adjacent islets that are not empty.

#### 2. Prepare results for adjacent islets
Then filter out all results of the model that has no overlap with the GT adjacent islets by running:
```bash
python -m data_preparation.convert_results_to_adjacent_islets_results <data_root> [--nn_masks_path <...>] [--pkl_file <...>]
```
- `data_root` is path to a folder that contains two folders:
  - `inputs/` - contains input images
  - `masks/` - contains GT semantic masks and `adjacent_islets` directory with adjacent islets GT masks
- `--nn_masks_path` is a path to the directory with predicted NN masks (only for semantic models)
- `--pkl_file` is a file with the results of instance segmentation in .pkl format (only for instance models)

In the case of a **semantic model**, it created a directory `adjacent_islets` within the `nn_masks_path` that contains 
NN masks with islets that have overlap with GT adjacent islets.

In the case of an **instance segmentation model**, it creates a file with name suffix `_adjacent_islets.pkl` 
in the same folder as the `pkl_file` is, that contains instances that have overlap with GT adjacent islets.
