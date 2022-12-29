#!/bin/bash
#SBATCH --job-name=Tuned_model_YG
#SBATCH --time=2-
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem 16G
#SBATCH --ntasks=1 
#SBATCH --partition=zhanglab.p
#SBATCH --out=out/tuned_model_2_YG.o


module load anaconda
source /pkg/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate scEpilock
python3 multi_task 'tuned_model_2_YG'
