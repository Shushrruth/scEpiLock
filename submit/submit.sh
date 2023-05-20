#!/bin/bash
#SBATCH --job-name=10k_test
#SBATCH --time=10-
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem 120G
#SBATCH --ntasks=1 
#SBATCH --partition=zhanglab.p
#SBATCH --out=out/10k.o
#SBATCH -w laniakea

module load anaconda
source /pkg/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate scEpilock
python3 multi_task '10k'
