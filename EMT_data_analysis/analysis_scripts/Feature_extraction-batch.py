import warnings
import platform
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from bioio import BioImage
from EMT_data_analysis.tools import io, alignment
from tqdm import tqdm
import pickle as pkl
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    # print(f"Dataset loaded. Shape: {df.shape}.")

    for movie_id, df_movie in df.groupby('Movie ID'):
    
        out_fn = Path(output_folder) / f"Features_bf_colony_mask_{movie_id}.csv"
        if out_fn.exists():
            print(out_fn.name, ' exists.')
            # df_out = pd.read_csv(out_fn, index_col=None)
            # df_out['File ID'] = df_movie['File ID'].values[0]
            # df_out.to_csv(out_fn, index=False)
            continue

        # print(f"Movie: {movie_id}")
        
        # print("Getting raw data...")
        raw_path = df_movie["File Path"].values[0]
        raw_reader = BioImage(raw_path)
        # print(raw_path)
        # print(raw_reader.shape)
    
        # print("Computing features....")

        df_result = []
        max_timepoint = df_movie['Timepoint'].max()

        for _, df_tp in tqdm(df_movie.iterrows(), total=len(df_movie.index), desc=f'Processing {movie_id}'):
            if raw_reader.shape[1] == 1:
                channel = 0
            else:
                channel = 1
            # print('Loading Timepoint')
            raw_img = raw_reader.get_image_dask_data("ZYX", C=channel, T=df_tp['Timepoint'])
            raw_img = raw_img.compute() 
                
            
            # print('Loading Segmentation ', Path(df_tp['All Cells Mask URL']).name)
            seg_reader = BioImage(df_tp["All Cells Mask URL"])
            seg_img = seg_reader.get_image_dask_data("ZYX")
            seg_img = seg_img.compute()

            if align:
                # print('Image Alignment', )
                matrix_string = df_movie["Camera Alignment Matrix"].values[0]
                matrix = alignment.parse_rotation_matrix_from_string(matrix_string)
                transform = alignment.get_alignment_matrix(matrix)
                transform = transform.inverse
            
            # print('Computing Metrics')
            tp = df_tp['Timepoint']
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

        return True

def main():
    manifest = '/allen/aics/users/filip.sluzewski/Public_Repos/emt-data-analysis/EMT-new-timelapse/final_new_release_April_24_w_raw_acm.csv'
    output_dir = Path('/allen/aics/users/filip.sluzewski/Public_Repos/emt-data-analysis/EMT-new-timelapse/metric')
    alignment_dir = Path('/allen/aics/users/filip.sluzewski/Projects/camera_alignment/EMT-Resubmission/camera-alignment/alignment_info')
    df_manifest = pd.read_csv(manifest, index_col=None)
    df_manifest = df_manifest[df_manifest['ACM_path']!='NA']
    df_manifest['ACM_path'].replace('',np.nan, inplace=True)
    df_manifest.dropna(subset=['ACM_path'], inplace=True)
    tasks = []

    for _, row in tqdm(df_manifest.iterrows(), total=len(df_manifest.index), desc='Compiling jobs'):
        barcode = str(row['Plate Barcode'])
        barcode_abrv = barcode[-4:]
        feature_output = output_dir / barcode_abrv / 'feature-extraction'
        feature_output.mkdir(exist_ok=True, parents=True)
        # print(feature_output)

        seg_path = Path(row['ACM_path'])
        if not seg_path.exists():
            print(seg_path,' is missing')
            continue
        pos = str(int(row['Position Index']))
        pos = f'P{pos}'
        well = row['Well Label']

        movie_id = f'{barcode}_{pos}-{well}'

        out_fn = Path(feature_output) / f"Features_bf_colony_mask_{movie_id}.csv"
        if out_fn.exists():
            print(movie_id, ' exists ...')
            continue

        df_scene = []
        for seg_fn in seg_path.glob('*.tif'):
            aMat = pkl.load(open(alignment_dir / f'{barcode}_alignmentmatrix.pkl','rb'))

            df_scene.append(
                pd.DataFrame({
                    'Movie ID': movie_id,
                    'File ID': row['File ID'],
                    'File Path': row['Raw_movie_path'],
                    'All Cells Mask URL': seg_fn,
                    'Timepoint': int(seg_fn.stem.split('_')[-1]),
                    'Camera Alignment Matrix': str(aMat),
                    'Gene': row['Gene'],
                    'Experimental Condition': row['Experimental Condition'],
                }, index=[0])
            )
        if len(df_scene)>0:
            print(movie_id, ' job added')
            df_scene = pd.concat(df_scene, ignore_index=True)
            tasks.append((df_scene, feature_output, True))

    print(len(tasks), ' total tasks to do')

    from multiprocessing import Pool
    with Pool(20) as p:
        res = p.starmap_async(compute_bf_colony_features_all_movies, tqdm(tasks, total=len(tasks), desc='Processing timelapses'))
        
        p.close()
        p.join()

    # with ThreadPoolExecutor(max_workers=8) as executor:
    #     futures = [executor.submit(compute_bf_colony_features_all_movies, *task) for task in tasks]
    #     for future in tqdm(as_completed(futures), total=len(futures), desc="Processing timelapses"):
    #         try:
    #             _ = future.result()
    #         except Exception as e:
    #             print(f"Error processing timepoint: {e}")


if __name__ == '__main__':
    main()


    
