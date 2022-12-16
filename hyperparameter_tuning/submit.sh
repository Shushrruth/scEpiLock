#!/bin/bash
#SBATCH --job-name=Tuning
#SBATCH --time=4-
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem 16G
#SBATCH --ntasks=1 
#SBATCH --partition=zhanglab.p
#SBATCH --out=tuning.o


module load anaconda
source /pkg/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate scEpilock
python3 hyperparameter_tuning.py