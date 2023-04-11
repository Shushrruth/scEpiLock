#!/bin/bash
#SBATCH --job-name=vivek
#SBATCH --time=2-
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem 16G
#SBATCH --ntasks=1 
#SBATCH --partition=zhanglab.p
#SBATCH --out=out/vivek_1.o


module load anaconda
source /pkg/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate scEpilock
python3 multi_task 'vivek_1'
