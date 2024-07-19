import pandas as pd
import numpy as np
from pathlib import Path
import argparse
parser = argparse.ArgumentParser()

from EMT_data_analysis.analysis_scripts.Feature_extraction import compute_bf_colony_features

def main(csv_path, output_path, idx):
    df_data = pd.read_csv(csv_path)
    
    fms = df_data.iloc[idx]['fms_id']
    df_fms = df_data[df_data['fms_id'] == fms]
    
    if (Path(output_path) / f'Features_bf_colony_mask_{fms}.csv').exists():
        print(f'Colony {fms} already processed. Skipping... \n')
        return
    
    print(f'Processing colony {fms}... \n')
    
    # Compute features
    compute_bf_colony_features(
        df_fms, 
        output_path, 
        align=True
    )
    
if __name__ == '__main__':
    parser.add_argument('--data_path', type=str, default='/allen/aics/assay-dev/users/Filip/Public_Repos/emt-data-analysis/colony_mask_data.csv')
    parser.add_argument('--output_path', type=str, default='/allen/aics/assay-dev/users/Filip/Data/EMT-colony-mask-features/')
    parser.add_argument('--idx', type=int)
    
    args = parser.parse_args()
    main(args.data_path, args.output_path, args.idx)