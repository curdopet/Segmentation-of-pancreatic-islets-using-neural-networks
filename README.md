# Segmentation of pancreatic islets and exocrine tissue from microscopic images using neural networks based approaches

## Prepare environment
Before running any code a conda environment with requierd libraries needs to be installed first:

1. Clone the repo and go to the directory `Segmentation-of-pancreatic-islets-using-neural-networks`

2. Create conda environment:
```bash
conda create --name islets-instance-segmentation python=3.8 -y
conda activate islets-instance-segmentation
conda install pytorch torchvision -c pytorch -y
```
Make sure that you install PyTorch version compatible with your version of CUDA ([tutorial](https://pytorch.org/get-started/locally/)).

3. Instal MMEngine and MMCV and MMDetection
```bash
./training/local_scripts/install_environment.sh
```

### MetaCentrum
If you do not want to run the training locally and use [MetaCentrum](https://metavo.metacentrum.cz/) then you have to install the environment on MetaCentrum.

1. Create conda module on the MetaCentrum fronted node:
   1. Download Anaconda distribuion for Linux from https://www.anaconda.com/products/distribution on the frontend node and install it.
   2. Create folder `my-modules` and copy the file `conda-module-file` and rename it after the version of conda you've installed in pattern `conda-<version>` - for instance `conda-22.9.0`.
2. Open the [install_environment.sh](training/metacentrum_scripts/install_environment.sh) and change the values in brackets `<>` accordingly:
   1. `<path-to-your-directory-on-frontend-node-with-repo>` = absolute path to your directory on MetaCentrum, that contains this repo
   2. `<name-of-your-directory-on-frontend-node-with-repo>` = name of your directory on MetaCentrum, that contains this repo
   3. `<your-modules-directory-on-frontend-node>` = absolute path to your "my-modules" directory
   4. `<name-of-the-conda-module-file>` = name of the conda module file in "my-modules"

Finally, install the environment by running:
```bash
qsub training/metacentrum_scripts/install_envirnonment.sh 
```

## Dataset
The dataset for this thesis was obtained from the Laboratory for the Islets of Langerhans, Experimental Medicine Centre (EMC), Institute for Clinical and Experimental Medicine (IKEM), Prague, CZ.

To work with this code, you will need a data organized in a following directory structure:
```
   .
   └── data_split
       ├── backgrounds
       │   ├── bg1.jpg
       │   └── ...
       ├── test_data
       │   ├── inputs/
       │   │    ├── input1.jpg
       │   │    └── ...
       │   └── masks/
       │   │    ├── input1_Exo_xx_GTyy.jpg
       │   │    └── ...
       ├── training_data
       │   ├── inputs/ ...
       │   └── masks/ ...
       ├── validation_data
       │   ├── inputs/ ...
       │   └── masks/ ...
       ├── labels.csv
       ├── pixel-sizes.py
       └── split.csv
 ```
The final directory structure should be organized as follows:
```
   .
   └── root_directory
       ├── data/
       ├── mmdetection/
       └── Segmentation-of-pancreatic-islets-using-neural-networks/
```

Before running the training, the data needs to be converted to the COCO annotation format. Go to `Segmentation-of-pancreatic-islets-using-neural-networks` and run:
```bash
python -m data_preparation.generate_jsons_for_data ../data_split/training_data training ../data_split/jsons ../data_split/labels.csv
python -m data_preparation.generate_jsons_for_data ../data_split/validation_data validation ../data_split/jsons ../data_split/labels.csv
python -m data_preparation.generate_jsons_for_data ../data_split/test_data test ../data_split/jsons ../data_split/labels.csv
```

If you want to train a HTC framework, you will need semantic masks of only islets. These can be prepared by running:
```bash
python -m data_preparation.create_islet_only_masks ../data_split/training_data/masks/
python -m data_preparation.create_islet_only_masks ../data_split/validation_data/masks/
python -m data_preparation.create_islet_only_masks ../data_split/test_data/masks/
```

Additionally, if you will want to evaluate the islets only on the mask of adjacent islets, run the following to generate the adjacent islet masks:
```bash
python -m data_preparation.create_adjacent_islets_masks_from_gt_masks ../data_split/validation_data/masks/
python -m data_preparation.create_adjacent_islets_masks_from_gt_masks ../data_split/test_data/masks/
```

## Training
Training is done using [MMDetection framework v3.2.0](https://github.com/open-mmlab/mmdetection).

### Locally
To train the model locally, open the [training_pipeline.sh](training/local_scripts/training_pipeline.sh) and change 
the values in brackets `<>` accordingly:
  1. `<wandb-access-key>` = access key to your WandB account - used for the training progress monitoring
  2. `<path-to-model-config-file>` = path to the config file of the model you want to train, for example `training/configs/config-mask-rcnn-resnet50-run01.py`.

And run:
```bash
./training/local_scripts/training_pipeline.sh
```

### MetaCentrum
To train the model using MetaCentrum, open the [training_pipeline.sh](training/metacentrum_scripts/training_pipeline.sh) and change 
the values in brackets `<>` accordingly:
   1. `<path-to-your-directory-on-frontend-node-with-repo>` = absolute path to your directory on MetaCentrum, that contains this repo
   2. `<name-of-your-directory-on-frontend-node-with-repo>` = name of your directory on MetaCentrum, that contains this repo
   3. `<your-modules-directory-on-frontend-node>` = absolute path to your "my-modules" directory
   4. `<name-of-the-conda-module-file>` = name of the conda module file in "my-modules"
   5. `<wandb-access-key>` = access key to your WandB account - used for the training progress monitoring
   6. `<path-to-model-config-file>` = path to the config file of the model you want to train, for example `training/configs/config-mask-rcnn-resnet50-run01.py`.

And run:
```bash
qsub training/metacentrum_scripts/training_pipeline.sh
```

## Validation
The prediction on the validation dataset can be also obtained locally or by using MetaCentrum.

### Locally
To get the predictions of the model locally, open the [validation_pipeline.sh](training/local_scripts/validation_pipeline.sh) and change 
the values in brackets `<>` accordingly:
  1. `<wandb-access-key>` = access key to your WandB account - used for the training progress monitoring
  2. `<path-to-model-config-file>` = path to the config file of the model you want to train, for example `training/configs/config-mask-rcnn-resnet50-run01.py`.
  3. `<path-to-the-last-training-epoch>` = path to the file with the last epoch, for example `work_dirs/config-mask-rcnn-resnet50-run01/epoch_12.pth` 
  4. `<path-to-output-file-for-results>` = path where to store the results, for example `work_dirs/mask-rcnn-resnet50-run01-results.pkl`

And run:
```bash
./training/local_scripts/validation_pipeline.sh
```

### MetaCentrum
To get the predictions of the model using MetaCentrum, open the [validation_pipeline.sh](training/metacentrum_scripts/validation_pipeline.sh) and change 
the values in brackets `<>` accordingly:
   1. `<path-to-your-directory-on-frontend-node-with-repo>` = absolute path to your directory on MetaCentrum, that contains this repo
   2. `<name-of-your-directory-on-frontend-node-with-repo>` = name of your directory on MetaCentrum, that contains this repo
   3. `<your-modules-directory-on-frontend-node>` = absolute path to your "my-modules" directory
   4. `<name-of-the-conda-module-file>` = name of the conda module file in "my-modules"
   5. `<wandb-access-key>` = access key to your WandB account - used for the training progress monitoring
   6. `<path-to-model-config-file>` = path to the config file of the model you want to train, for example `training/configs/config-mask-rcnn-resnet50-run01.py`.
   7. `<path-to-the-last-training-epoch>` = path to the file with the last epoch, for example `work_dirs/config-mask-rcnn-resnet50-run01/epoch_12.pth` 
   8. `<path-to-output-file-for-results>` = path where to store the results, for example `work_dirs/mask-rcnn-resnet50-run01-results.pkl`

And run:
```bash
qsub training/metacentrum_scripts/validation_pipeline.sh
```

### Postprocessing of the results
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
- `--px_file` - path to a .csv file that contains the list of μm/px ratios for input images
- `--nn_masks_path` is a path to the directory with predicted NN masks (only for `nn_type == semantic`)
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
python -m data_preparation.create_adjacent_islets_masks_from_gt_masks <data_root>
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

## Visualization of the results
The results of instance segmentation can be visualized by the following command:
```bash
python -m visualization.visualize_instance_segmentation_results <input_images_dir> <gt_masks_dir> <results_dir> <results_file>
```
- `input_images_dir` is a path to folder where input images are stored
- `gt_masks_dir` is a path to folder where GT masks are stored
- `results_dir` is a path to folder where the visualization results will be stored
- `results_file` is a path to file where results in .pkl format are stored
