import os
import pandas as pd
from pathlib import Path, PurePosixPath

def convert_to_windows_path(linux_path: Path):
    return PurePosixPath(linux_path)

def load_imaging_and_segmentation_dataset(load_from_aws: bool = True, local_path: str = None):
    """
    Load the imaging and segmentation dataset.

    Parameters
    ----------
    load_from_aws : bool, default True
        If True, load from AWS S3. If False, load from local file.
    local_path : str, optional
        Path to local CSV file. If not provided and load_from_aws=False,
        will look for 'imaging_and_segmentation_data.csv' in the project root.

    Returns
    -------
    df : DataFrame
        The imaging and segmentation dataset
    """
    if load_from_aws:
        path = "https://allencell.s3.amazonaws.com/aics/emt_timelapse_dataset/manifests/imaging_and_segmentation_data.csv"
    else:
        if local_path is not None:
            path = local_path
        else:
            # Default local path: project root (parent of EMT_data_analysis package)
            project_root = Path(__file__).parent.parent.parent
            path = project_root / "imaging_and_segmentation_data.csv"
        print(f'Loading from local file: {path}')

    df = pd.read_csv(path)
    n_movies = df['Data ID'].nunique()
    print(f'Total number of movies in the dataset: {n_movies}')
    return df

def load_image_analysis_extracted_features(load_from_aws: bool = True):
    metric_comp_results_dir = get_results_directory_name() / "metric_computation"
    path = metric_comp_results_dir / "Image_analysis_extracted_features.csv"
    try:
        print('Trying to load features from local path.')
        df = pd.read_csv(path)
    except Exception:
        print(f'Features not found at {path}. Loading from AWS instead. This may take a while...')
        path = "https://allencell.s3.amazonaws.com/aics/emt_timelapse_dataset/manifests/Image_analysis_extracted_features.csv?versionId=ehxRXxC0FpidcpgXU_z.51T.nkWB0Yuj"
        df = pd.read_csv(path)
    return df

def load_inside_outside_classification(load_from_aws: bool = True):
    path = "https://allencell.s3.amazonaws.com/aics/emt_timelapse_dataset/manifests/Migration_timing_through_mesh_extracted_features.csv?versionId=jXI9TfrFDcQZgS3LssIumKkPO47H_MzN"
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
