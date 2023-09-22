#!/bin/bash
#SBATCH --job-name=200bp
#SBATCH --time=4-
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem 10G
#SBATCH --ntasks=1 
#SBATCH --partition=zhanglab.p
#SBATCH --out=out/200bp.o
#SBATCH -w galaxy

module load anaconda
source /pkg/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate scEpilock
python3 multi_task '200'
