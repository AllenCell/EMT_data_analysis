import warnings
import platform
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from bioio import BioImage
from EMT_data_analysis.tools import io, alignment

warnings.filterwarnings("ignore")

def compute_bf_colony_features_all_movies(df, output_folder, align=True):
    '''
    Computes area of the bright field colony mask at every z position
    and extracts corresponding intensity values from the fluorescence
    channel. It also adds other features from the bright field colony
    mask to the final dataframe that is saved at the end.
    Parameters
    ----------
    df: pd.DataFrame
        Data manifest to process
    save_folder: path
        Folder path where feature csv for each movie is stored

    align: bool
        Enable alignment of the image using the barcode of the movie
    '''

    print(f"Dataset loaded. Shape: {df.shape}.")

    for movie_id, df_movie in tqdm(df.groupby('Movie ID')):
    
        out_fn = Path(output_folder) / f"Features_bf_colony_mask_{movie_id}.csv"
        if out_fn.exists():
            continue

        print(f"Movie: {movie_id}")
        
        print("Getting raw data...")
        raw_path = df_movie["File Path"].values[0]
        raw_reader = BioImage(raw_path)
    
        # print("Getting colony mask....")
        # seg_path = df_movie["All Cells Mask URL"].values[0]
        # seg_reader = BioImage(seg_path)
        
        print("Computing features....")

        df_result = []
        # We only process the first 48 hours (98 timepoints)
        # max_timepoint = int(np.min([98, df_movie['Image Size T'].values[0]]))
        max_timepoint = df_movie['Timepoint'].max()

        for _, df_tp in tqdm(df_movie.iterrows(), total=len(df_movie.index)):
            raw_img = raw_reader.get_image_dask_data("ZYX", C=1, T=df_tp['Timepoint'])
            raw_img = raw_img.compute() 
                
            seg_reader = BioImage(df_tp["All Cells Mask URL"])
            seg_img = seg_reader.get_image_dask_data("ZYX")
            seg_img = seg_img.compute()

            if align:
                matrix_string = df_movie["Camera Alignment Matrix"].values[0]
                matrix = alignment.parse_rotation_matrix_from_string(matrix_string)
                transform = alignment.get_alignment_matrix(matrix)
                transform = transform.inverse
            
            for z, seg in enumerate(seg_img):
                if align:
                    seg = alignment.align_image(seg, transform)

                mask = np.bool_(seg)
                area = np.count_nonzero(mask)
                mean_intensity = np.mean(raw_img[z][mask])
                total_intensity = np.sum(raw_img[z][mask])

                row = {
                    "Z plane": z,
                    "Timepoint": df_tp['Timepoint'],
                    "Movie ID": movie_id,
                    "Mean intensity per Z": mean_intensity,
                    "Total intensity per Z": total_intensity,
                    "Area of all cells mask per Z (pixels)": area
                }
                df_result.append(row)
                    
        df_result = pd.DataFrame(df_result)
        df_result["Gene"] = df_movie.Gene.values[0]
        df_result["Experimental Condition"] = df_movie["Experimental Condition"].values[0]
        df_result.to_csv(out_fn)

if __name__ == '__main__':

    manifest = pd.read_csv('/allen/aics/users/filip.sluzewski/Public_Repos/emt-data-analysis/EMT_EOMES-new-timelapse/7062/manifest.csv', index_col=None)
    result_dir = '/allen/aics/users/filip.sluzewski/Public_Repos/emt-data-analysis/EMT_EOMES-new-timelapse/7062/feature-extraction'
    compute_bf_colony_features_all_movies(df=manifest, output_folder=result_dir)





    
