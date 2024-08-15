import platform
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from bioio import BioImage

import warnings
warnings.filterwarnings("ignore")

from EMT_data_analysis.tools import io, alignment

def compute_bf_colony_features_all_movies(output_folder, align=True):
    '''
    Computes area of the bright field colony mask at every z position
    and extracts corresponding intensity values from the fluorescence
    channel. It also adds other features from the bright field colony
    mask to the final dataframe that is saved at the end.
    Parameters
    ----------
    save_folder: path
        Folder path where feature csv for each movie is stored

    align: bool
        Enable alignment of the image using the barcode of the movie
    '''

    df = io.load_imaging_and_segmentation_dataset()
    print(f"Dataset loaded. Shape: {df.shape}.")

    for movie_id, df_movie in tqdm(df.groupby('Movie ID')):
    
        print(f"Movie: {movie_id}")
        
        print("Getting raw data...")
        raw_path=df_movie["File Path"].values[0]
        if platform.system() == "Windows":
            raw_path = io.convert_to_windows_path(Path(raw_path))

        raw_reader = BioImage(raw_path)
    
        print("Getting colony mask....")
        seg_path = df_movie["All Cells Mask BFF path"].values[0]

        if platform.system() == "Windows":
           seg_path = io.convert_to_windows_path(Path(seg_path))
        
        seg_reader = BioImage(seg_path)
        
        print("Computing features....")

        df_result = []
        # We only process the first 48 hours (98 timepoints)
        max_timepoint = int(np.min([98, df_movie['Image Size T'].values[0]]))

        for frame in tqdm(range(max_timepoint), total=max_timepoint):
            raw_img = raw_reader.get_image_dask_data("ZYX", C=1, T=frame)
            raw_img = raw_img.compute() 
                
            seg_img = seg_reader.get_image_dask_data("ZYX", T=frame)
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
                    "Timepoint": frame,
                    "Movie ID": movie_id,
                    "Mean intensity per Z": mean_intensity,
                    "Total intensity per Z": total_intensity,
                    "Area of all cells mask per Z (pixels)": area
                }
                df_result.append(row)
                    
        df_result = pd.DataFrame(df_result)
        df_result["Gene"] = df_movie.Gene.values[0]
        df_result["Experimental Condition"] = df_movie["Experimental Condition"].values[0]
        df_result.to_csv(output_folder / f"Features_bf_colony_mask_{movie_id}.csv")

if __name__ == '__main__':

    base_results_dir = io.setup_base_directory_name("feature_extraction")
    compute_bf_colony_features_all_movies(output_folder=base_results_dir)





    
