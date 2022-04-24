#!/bin/bash
#SBATCH --job-name=train
#SBATCH --time=2-
#SBATCH --gpus 2
#SBATCH --cpus-per-task 8
#SBATCH --mem 16G
#SBATCH --ntasks=1 
#SBATCH --partition=gpu
#SBATCH --output=out/classifier_test.o
#
module load miniconda
conda activate scEpilock
python3 binary_classifier
