#!/bin/bash

# Prepare
wandb login <wandb-access-key> # if you do not want to use WandB, just comment this line
export PYTHONPATH=$PYTHONPATH:$(pwd)/training/

# Predict
python ../mmdetection/tools/test.py <path-to-the-model-config> <path-to-the-last-training-epoch> --out <path-to-output-file-for-results>
