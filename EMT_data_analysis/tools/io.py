import numpy as np
import pandas as pd
from pathlib import Path, PurePosixPath

def convert_to_windows_path(linux_path: Path):
    return PurePosixPath(linux_path)

def load_imaging_and_segmentation_dataset():
    df = pd.read_csv("/Users/matheus.viana/Downloads/imaging_and_segmentation_data_FOR_Matheus_test.csv")#Dataset_1_V20_V8_SM.csv")
    return df

def setup_base_directory_name(name: str):
    base_results_dir = Path(__file__).parent.parent.parent / f"results/{name}"
    base_results_dir.mkdir(exist_ok=True, parents=True)
    return base_results_dir