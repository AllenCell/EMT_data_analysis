#!/bin/bash
#SBATCH --job-name=colony_mask
#SBATCH --mem=10G
#SBATCH -c 1
#SBATCH --time=1:00:00
#SBATCH --partition=aics_cpu_general
#SBATCH --array=0-388%10
#SBATCH --output=/allen/aics/assay-dev/users/Filip/Public_Repos/emt-data-analysis/logs/%x_%A_%a.out

echo "Starting task $SLURM_ARRAY_TASK_ID"

echo "Running python script"

/allen/aics/apps/hpc_shared/mod/anaconda3-5.3.0/envs/filip_emtAnalysis/bin/python3.10 \
 /allen/aics/assay-dev/users/Filip/Public_Repos/emt-data-analysis/run_timelapse.py \
 --idx $SLURM_ARRAY_TASK_ID