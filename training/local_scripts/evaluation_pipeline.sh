#!/bin/bash
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=1:ngpus=1:mem=8gb:scratch_local=10gb:scratch_ssd=10gb:cluster=galdor
#PBS -N val_<model_name>

# Load data and scripts
DATADIR=<your-project-directory-on-frontend-node>
cp -r $DATADIR $SCRATCHDIR
cd $SCRATCHDIR

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

# Predict
python mmdetection/tools/test.py <path-to-the-model-config> <path-to-the-last-training-epoch> --out <path-to-output-file-for-results>

# Copy results
cp -r ./work_dirs/*.pkl $DATADIR/work_dirs/

# Clean scratchdir
cd ..
rm -rf *
clean_scratch

