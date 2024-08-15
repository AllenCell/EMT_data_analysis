import os
import pandas as pd
from pathlib import Path, PurePosixPath

def convert_to_windows_path(linux_path: Path):
    return PurePosixPath(linux_path)

def load_imaging_and_segmentation_dataset():
    df = pd.read_csv("https://allencell.s3.amazonaws.com/aics/emt_timelapse_dataset/manifests/imaging_and_segmentation_data.csv")
    n_movies = df['Movie ID'].nunique()
    print(f'Total number of movies in the dataset: {n_movies}')
    return df
def load_image_analysis_extracted_features(load_from_aws: bool = True):
    path = "https://allencell.s3.amazonaws.com/aics/emt_timelapse_dataset/manifests/Image_analysis_extracted_features.csv"
    if not load_from_aws:
        # Or read from local if the user decides to run Metric_computation.py
        metric_comp_results_dir = get_results_directory_name() / "metric_computation"
        path = metric_comp_results_dir / "Image_analysis_extracted_features.csv"
    df = pd.read_csv(path)
    return df

def load_inside_outside_classification(load_from_aws: bool = True):
    path = "https://allencell.s3.amazonaws.com/aics/emt_timelapse_dataset/manifests/Migration_timing_through_mesh_extracted_features.csv"
    if not load_from_aws:
        # Or read from local if the user decides to run Nuclei_localization.py
        metric_comp_results_dir = get_results_directory_name() / "nuclei_localization"
        path = metric_comp_results_dir / "Migration_timing_trough_mesh_extracted_features.csv"
    df = pd.read_csv(path)
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
