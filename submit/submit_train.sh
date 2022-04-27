#!/bin/bash
#SBATCH --job-name=train
#SBATCH --time=1-
#SBATCH --gpus=rtx3090:1
#SBATCH --cpus-per-task 8
#SBATCH --mem 16G
#SBATCH --ntasks=1 
#SBATCH --partition=gpu
#SBATCH --out=out/classifier.o

module load miniconda
conda activate scEpilock
python3 binary_classifier
