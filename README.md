# EMT data analysis
This repository contains code for reproducing the plots shown in our manuscript. This repository uses outputs generated by the [EMT_image_analysis](https://github.com/AllenCell/EMT_image_analysis) repository, such as image segmentations and 3D meshes.

# Note
This code has been tested on Ubuntu 18.04.2 LTS and Windows 10 using Python 3.11.

# Installation
1. Install python>=3.11 and pip>=24.0.0.
2. Create a new virtual environment.
```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

# How to run:

## 1 - Feature extraction

Run: `python Feature_extraction.py`

This will generate one CSV for each movie with the extracted features. CSVs are stored in the folder `EMT_data_analysis/results/feature_extraction`

## 2 - Metric computation

Run: `python Metric_computation.py`

This will generate a single CSV containing information about all the movies to be used for analysis. The manifest is saved as `EMT_data_analysis/results/metric_computation/Image_analysis_extracted_features.csv`.

## 3 - Nuclei localization

This will generate CSV for individual nuclei classified as inside the basement memebrane or not over the course of the timelapse for EOMES and H2B movies. The manifest is saved as `EMT_data_analysis/results/nuclei_localization/Migration_timing_trough_mesh_extracted_feature.csv`.

## 4 - Analysis Plots

Run: `python Analysis_plots.py`

This will generate the plots in the manuscript and store them in `results/figures` folder. The manifests used as inputs in this workflow are automatically downloaded from [AWS](https://open.quiltdata.com/b/allencell/tree/aics/emt_timelapse_dataset/manifests/) by default. The user can opt to also use local version of these manifests if they produced locally by running the scripts `Feature_extraction.py`, `Metric_computation.py` and `Nuclei_localization.py`. To use local version of the manifests, please set `load_from_aws=False` everywhere in the script `Analysis_plots.py`.
