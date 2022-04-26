#!/bin/bash
#SBATCH --job-name=train
#SBATCH --time=1-
#SBATCH --gpus 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 16G
#SBATCH --ntasks=1 
#SBATCH --partition=gpu
<<<<<<< HEAD
#SBATCH --output=out/classifier.o
=======
#SBATCH --output=out/classifier_test_test.o
>>>>>>> 1780997c1d056ee763730e95d1f45a71fe6d6063
#
module load miniconda
conda activate scEpilock
python3 binary_classifier
