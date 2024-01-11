#!/bin/bash

# Prepare
wandb login <wandb-access-key> # if you do not want to use WandB, just comment this line
export PYTHONPATH=$PYTHONPATH:$(pwd)/training/

# Train model
python ../mmdetection/tools/train.py training/configs/<path-to-model-config-file>
