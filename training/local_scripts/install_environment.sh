#!/bin/bash

conda install -c conda-forge progress=1.6 wandb=0.16.2 plotly=5.18.0 jupyter=1.0.0 jupyter_server=1.23.4 scikit-learn=1.3.0 -y

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

cd ..
git clone --branch v3.2.0 https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
