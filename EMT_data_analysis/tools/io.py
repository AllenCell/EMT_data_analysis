import os
import pandas as pd
from pathlib import Path, PurePosixPath

def convert_to_windows_path(linux_path: Path):
    return PurePosixPath(linux_path)

def load_imaging_and_segmentation_dataset():
    # Replace with AWS path once the CSV is public
    df = pd.read_csv("/Users/matheus.viana/Downloads/imaging_and_segmentation_data_FOR_Matheus_test.csv")#Dataset_1_V20_V8_SM.csv")
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
