# EMT_data_analysis
This repo contains code for EMT data analysis including generation of plots in the manuscript and has dependencies associated with the output from EMT_image_analysis repo.

The analysis plots part of this repo can be run independently by importing the pre-generated feature manifests from Amazon s3 links (given below).

# Installation
Note: this repository currently depends on aicsfiles and therefore is only usable within AICS
1. Install python>=3.11 and pip>=24.0.0.
2. Install in a new virtual environment.
```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

# How to run:

## 1 - Feature extraction
This will generate one CSV for each movie with the extracted features. CSVs are stored in the folder `EMT_data_analysis/results/feature_extraction`

## 2 - Metric computation
This will generate the entire compiled feature manifest csv (Image_analysis_extracted_features.csv)for all the movies to be used for analysis and stores it in the folder 'EMT_data_analysis/results/Feature_manifests'

## 3 - Nuclei localization
This will generate csv for individual nuclei classified as inside the basement memebrane or not over the course of the timelapse for EOMES and H2B movies. This csv (Migration_timing_trough_mesh_extracted_feature.csv) is used for generation of plots.

## 4 - Analysis Plots
This will generate the plots in the manuscript and store them in Figures_manuscript folder.
This workflow takes two input manifests:
    1. Output from Metric computation - Image_analysis_extracted_features.csv
    2. Output from Nuclei localization - Migration_timing_through_mesh_extracted_features.csv
There are two ways to run this workflow:
1. Run the entire workflow from steps 1 to 3 to get the required manifests.
2. Access the pre-generated csvs located in publically available s3 storage 
    Links are here: 1. 
                    2.
