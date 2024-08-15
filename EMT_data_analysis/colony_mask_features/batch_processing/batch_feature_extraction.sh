#!/bin/bash
#SBATCH --job-name=colony_mask
#SBATCH --mem=10G
#SBATCH -c 1
#SBATCH --time=3:00:00

# Make sure to change the partition to the one you have access to, preferably one that only has cpu nodes
#SBATCH --partition=aics_cpu_general

# Make sure to change the number of tasks [388 here] to the number of images you have
# Change the %10 to the number of tasks you want to run in parallel
#SBATCH --array=0-388%10

# Include path to the folder where you want to save job jogs, or delete this line
#SBATCH --output=[path/to/logs/folder]/%x_%A_%a.out

echo "Starting task $SLURM_ARRAY_TASK_ID"

echo "Running python script"

# Add path to the data csv file and output folder
export DATA_PATH = ""
export OUTPUT_PATH = ""

# Add to the path below the location of the virtual environment you created
source [path/to/environment]/venv/bin/activate

# Add to the path below the location that you place this repo into
python \
 [path/to/repo]/emt-data-analysis/run_timelapse.py \
 --data_path $DATA_PATH \
 --output_path $OUTPUT_PATH \
 --idx $SLURM_ARRAY_TASK_ID