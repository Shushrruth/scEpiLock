#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=1-
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem-per-cpu 5g
#SBATCH --ntasks=1 
#SBATCH --partition=gpu
#
module load miniconda
conda activate ScEpilock
python Test/tester.py



