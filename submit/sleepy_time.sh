#!/bin/bash
#
#SBATCH --job-name=sleepy_galaxy
#SBATCH --output=sleepy_time_%j.out
#SBATCH --error=sleepy_time_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00
#SBATCH -p zhanglab.p
#SBATCH -w laniakea

srun hostname
srun sleep 180

