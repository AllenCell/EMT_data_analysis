import os
import pandas as pd
from pathlib import Path, PurePosixPath

def convert_to_windows_path(linux_path: Path):
    return PurePosixPath(linux_path)

def load_imaging_and_segmentation_dataset():
    # Replace with AWS path once the CSV is public
    df = pd.read_csv("/Users/matheus.viana/Downloads/imaging_and_segmentation_data_V11.csv")
    n_movies = df['Movie ID'].nunique()
    print(f'Total number of movies in the dataset: {n_movies}')
    return df

def get_results_directory_name():
    base_results_dir = Path(__file__).parent.parent / "results"
    return base_results_dir

def setup_base_directory_name(name: str):
    base_results_dir = get_results_directory_name() / name
    base_results_dir.mkdir(exist_ok=True, parents=True)
    return base_results_dir

def load_bf_colony_features():
    df = pd.DataFrame()
    feats_results_dir = get_results_directory_name() / "feature_extraction"
    for file in os.listdir(feats_results_dir):
        if not file.endswith(".csv"):
            continue
        df_sub = pd.read_csv(feats_results_dir / file)
        df = pd.concat([df, df_sub])
    return df

def load_image_analysis_extracted_features(load_from_aws: bool = True):
    # Replace with AWS path once the CSV is public
    path = "/Users/matheus.viana/Documents/EMT_data_analysis/EMT_data_analysis/results/metric_computation/Image_analysis_extracted_features.csv"
    if not load_from_aws:
        # Or read from local if the user decides to run Metric_computation.py
        metric_comp_results_dir = get_results_directory_name() / "metric_computation"
        path = metric_comp_results_dir / "Image_analysis_extracted_features.csv"
    df = pd.read_csv(path, index_col=0)
    return df

def load_inside_outside_classification(load_from_aws: bool = True):
    # Replace with AWS path once the CSV is public
    path = "/Users/matheus.viana/Documents/EMT_data_analysis/EMT_data_analysis/results/nuclei_localization/Migration_timing_trough_mesh_extracted_features.csv"
    if not load_from_aws:
        # Or read from local if the user decides to run Nuclei_localization.py
        metric_comp_results_dir = get_results_directory_name() / "nuclei_localization"
        path = metric_comp_results_dir / "Migration_timing_trough_mesh_extracted_features.csv"
    df = pd.read_csv(path, index_col=0)
    return df
