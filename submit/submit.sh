#!/bin/bash
#SBATCH --job-name=train_multi_task
#SBATCH --time=2-
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem 16G
#SBATCH --ntasks=1 
#SBATCH --partition=zhanglab.p
#SBATCH --out=out/classifier_multi_task.o

module purge
module load anaconda
conda activate scEpilock
python3 multi_task 'Multi_task'
