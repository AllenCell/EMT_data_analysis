import pandas as pd
from pathlib import Path
import argparse
parser = argparse.ArgumentParser()

from EMT_data_analysis.analysis_scripts.Feature_extraction import compute_bf_colony_features

def main(csv_path, output_path, idx):
    df_data = pd.read_csv(csv_path)
        
    # Compute features
    compute_bf_colony_features(
        df_fms, 
        output_path, 
        align=True
    )
    
if __name__ == '__main__':
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    
    args = parser.parse_args()
    main(args.data_path, args.output_path, args.idx)