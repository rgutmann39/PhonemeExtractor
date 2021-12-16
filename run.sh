#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=0-01:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-gpu=16GB

# set up job
module load python/3.7.12
pushd /home/rcgutman/PhonemeExtractor
source venv/bin/activate
pip install datasets
pip install transformers
pip install jiwer

python training.py