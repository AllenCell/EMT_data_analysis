# Colony Mask Feature Extraction

In this repo we provide two approaches that can be used to reproduce the colony mask feature calculation. 

## Local Processing
The first is to process the dataset locally on your machine in a sequential order (on colony timelapse at a time). This approach is simpler but depending on hardware can take a long time to process.

Assuming you have already created the virtual environment as instructed, run the following commands.

```bash
# activate virtual environment
source venv/bin/activate

# navigate to folder with the github repo if you aren't already
cd [path/to/repo/folder]
# enter the subfolder for sequential processing
cd EMT_data_analysis/colony_mask_features/sequential_processing/

# run the sequential processing script
# replace the brackets with the appropraite paths
python run_dataset.py \
    --data_path [path/to/dataset/manifest.csv] \
    --ouput [path/to/output/directory]
```

This will start the data analyis process and you will see its progress as outputs in the terminal. If you computer/process shuts down halfway through, you can rerun the command with the same output path and the code will detect when data has already been processed and pick up where it was left off.

## Parallel Processing with SLURM
**Note 1:** This approach assumes that you have access to a high performance computing (hpc) cluster using the SLURM task scheduler, and are familiar with its use. 

**Note 2:** The example script provided assumes that your hpc cluster is configured such that you will be able to access the data, repo, and virtual environment. If not, additional steps will be required that can be unique to your situtation. If you are not familiar with how to get our data, repo, and virtual environment onto you hpc cluster, please consult with the IT or sys-admin team responcible for your hpc cluster for guidance.

Since the dataset is so large, it is faster to be able to process multiple colonies in parallel. The approach we provide here is to run the processing as a SLURM job array using an `sbatch` script.

### Steps
1) Make a copy of the `batch_feature_extraction.sh` and modify the arguments inside it as instructed to match the locations of the code and data on your system.

2) **Inside your hpc cluster,** navigate to the directory with your copy of `batch_feature_extraction.sh` and run the command

```bash
cd [path/to/your/custom/script/folder]

sbatch batch_feature_extraction.sh
```

3) You should be able to view the progress of your job array using the `squeue` command.