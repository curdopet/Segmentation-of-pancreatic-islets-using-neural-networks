#!/bin/bash
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:ngpus=1:mem=16gb:scratch_local=12gb:scratch_ssd=12gb:cluster=galdor
#PBS -N train_<model_name>

# Load data and scripts
DATADIR=<path-to-your-directory-on-frontend-node-with-repo>
cp -r $DATADIR $SCRATCHDIR
cd $SCRATCHDIR/<name-of-your-directory-on-frontend-node-with-repo>

# Load and activate conda environment
export MODULEPATH=$MODULEPATH:<your-modules-directory-on-frontend-node>
module load <name-of-the-conda-module-file>
conda init
source ~/.bashrc
conda activate islets-instance-segmentation

# Install other requirements
cd mmdetection
pip install -v -e .
cd ..

wandb login <wandb-access-key>
export PYTHONPATH=$PYTHONPATH:$(pwd)/Segmentation-of-pancreatic-islets-using-neural-networks/training/

# Train model
cd Segmentation-of-pancreatic-islets-using-neural-networks/
python ../mmdetection/tools/train.py training/configs/<path-to-model-config-file>

# Copy results
cp -r work_dirs/  $DATADIR/Segmentation-of-pancreatic-islets-using-neural-networks/work_dirs/

# Clean scratchdir
cd ../..
rm -rf *
clean_scratch

