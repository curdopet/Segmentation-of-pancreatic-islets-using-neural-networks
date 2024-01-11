#!/bin/bash

conda install -c conda-forge progress -y
conda install -c conda-forge wandb -y

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

cd ..
git clone --branch v3.2.0 https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
