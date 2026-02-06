#%%[markdown]
'''
This script adds metrics to generate the entire feature manifest used
for analysis_plot.py file. It requires the input of the folder path where
the files from feature_extraction.py file are stored.
'''
import warnings
import time
import numpy as np
import pandas as pd
import scipy.ndimage
from tqdm import tqdm
from bioio import BioImage
from scipy.signal import savgol_filter
from EMT_data_analysis.tools import io
from pathlib import Path
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")


def _process_single_movie_area(data_id, z_bottom, mask_url, timepoints, max_retries=5):
    """
    Helper function to compute area at the glass for a single movie.
    Designed to be called in parallel. Includes retry logic for network errors.
    """
    if pd.isna(mask_url):
        return [], None

    results = []

    # Retry logic for loading BioImage
    img_seg = None
    last_error = None
    for attempt in range(max_retries):
        try:
            img_seg = BioImage(mask_url)
            break  # Success, exit retry loop
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 30)  # Exponential backoff: 1, 2, 4, 8, 16 (max 30)
                time.sleep(wait_time)
            continue

    if img_seg is None:
        print(f"Error loading {data_id} after {max_retries} retries: {last_error}")
        return [], data_id

    try:
        # Check actual T dimension of the image
        actual_t_size = img_seg.dims.T
        is_fixed_cell = (actual_t_size == 1)

        for t in timepoints:
            # For fixed cells (single timepoint), always use T=0 regardless of computed timepoint
            t_index = 0 if is_fixed_cell else min(int(t), actual_t_size - 1)
            img_seg_tl = img_seg.get_image_dask_data("ZYX", T=t_index)
            img_z = img_seg_tl[z_bottom:z_bottom+2]
            z_max_proj = np.max(img_z, axis=0)
            img_fh = scipy.ndimage.binary_fill_holes(z_max_proj).astype(int)

            area_pixels = np.count_nonzero(img_fh)
            results.append({
                'Data ID': data_id,
                'Timepoint': t,
                'Area at the glass (pixels)': area_pixels,
                'Area at the glass(square micrometer)': area_pixels * (0.271 * 0.271)
            })
        return results, None
    except Exception as e:
        print(f"Error processing {data_id}: {e}")
        return [], data_id

# %% [markdown]
def add_bottom_z(df):
    """
    This function adds bottom Z - Zplane corresponding to the glass - defined
    from the sum of area over all time points vs Z

    Parameters
    ----------
    df: DataFrame
        Dataframe obtained from feature extraction using all-cells masks

    Returns
    -------
    df_merged: DataFrame
        Returns the input DataFrame with'Normalized Z plane' and 'Bottom Z plane' columns"""

    df['Area of all cells mask per Z (square micrometer)']=df['Area of all cells mask per Z (pixels)']*(0.271*0.271)
    area_time=df.groupby(['Data ID','Z plane'])['Area of all cells mask per Z (square micrometer)'].agg('sum').reset_index()
    file_id, z_bottom=[],[]
    for id, df_id in tqdm(area_time.groupby('Data ID')):
        file_id.append(id)
        df_id=df_id.reset_index()

        raw_values=df_id['Area of all cells mask per Z (square micrometer)'].values
        dy=np.diff(raw_values)
        max_dy=np.max(dy)
        idx_max= np.where(dy== max_dy)[0]

        zo=df_id['Z plane'][idx_max].values[0]+1

        z_bottom.append(zo)

    df_bottom_z=pd.DataFrame(zip(file_id,z_bottom), columns=['Data ID','Bottom Z plane'])

    df_normalized_z=pd.merge(df,df_bottom_z, on=['Data ID'])

    # Vectorized subtraction (much faster than apply with lambda)
    df_normalized_z['Normalized Z plane'] = df_normalized_z['Z plane'] - df_normalized_z['Bottom Z plane']

    return df_normalized_z


def add_bottom_mip_migration(df_merged, n_jobs=32):
    '''
    This adds area of MIP of bottom 2Z planes to get area at the glass and compute migration time from that.

    Parameters
    ----------
    df_merged: DataFrame
        Dataframe with 'Data ID', 'Timepoint', 'Bottom Z plane', and 'All Cells Mask URL' columns.
    n_jobs: int
        Number of parallel jobs. -1 uses all available cores. Default: 32.

    Returns
    -------
    df_mm: DataFrame
        Returns DataFrame with 'Data ID', 'Timepoint', 'Area at the glass (pixels)',
        'Area at the glass(square micrometer)' columns
    '''

    # Prepare arguments for parallel processing
    movie_args = []
    for data_id, df_id in df_merged.groupby('Data ID'):
        z_bottom = df_id['Bottom Z plane'].values[0]
        mask_url = df_id['All Cells Mask URL'].values[0]
        timepoints = sorted(df_id['Timepoint'].unique())
        movie_args.append((data_id, z_bottom, mask_url, timepoints))

    print(f"Processing {len(movie_args)} movies with {n_jobs} parallel jobs...")

    # Process movies in parallel
    all_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_single_movie_area)(data_id, z_bottom, mask_url, timepoints)
        for data_id, z_bottom, mask_url, timepoints in movie_args
    )

    # Flatten results list and collect failed movies
    results = []
    failed_movies = []
    for movie_results, failed_id in all_results:
        results.extend(movie_results)
        if failed_id is not None:
            failed_movies.append(failed_id)

    if failed_movies:
        print(f"\nWARNING: {len(failed_movies)} movies failed after retries:")
        for movie_id in failed_movies:
            print(f"  - {movie_id}")

    # Single DataFrame creation at the end
    df_mm = pd.DataFrame(results)
    return df_mm


def add_metadata_only_movies(df_features, Imaging_and_segmentation_data):
    """
    Add metadata-only movies (those without All Cells Mask) to the feature manifest.
    These movies have metadata but couldn't be processed by feature extraction
    because they lack segmentation masks.

    Parameters
    ----------
    df_features: DataFrame
        The main feature dataframe with processed movies
    Imaging_and_segmentation_data: DataFrame
        Full imaging and segmentation metadata

    Returns
    -------
    df_combined: DataFrame
        Combined dataframe with both processed and metadata-only movies
    """
    # Find movies without All Cells Mask URL (metadata-only)
    df_no_mask = Imaging_and_segmentation_data[
        Imaging_and_segmentation_data['All Cells Mask URL'].isna()
    ]
    metadata_only_ids = set(df_no_mask['Data ID'].unique())

    # Exclude any that might already be in features (shouldn't happen, but safe check)
    processed_ids = set(df_features['Data ID'].unique())
    metadata_only_ids = metadata_only_ids - processed_ids

    if len(metadata_only_ids) == 0:
        print("No metadata-only movies to add")
        return df_features

    print(f"Adding {len(metadata_only_ids)} metadata-only movies (no All Cells Mask)")

    # Get metadata for these movies (one row per movie)
    df_metadata_only = Imaging_and_segmentation_data[
        Imaging_and_segmentation_data['Data ID'].isin(metadata_only_ids)
    ].drop_duplicates('Data ID')

    # Combine with existing features
    df_combined = pd.concat([df_features, df_metadata_only], ignore_index=True, sort=False)

    return df_combined


def add_gene_metrics(df_features):
    '''
    This function adds genen specific metrics ( 'Time of max EOMES expression (h)',
       'Time of inflection of E-cad expression (h)',
       'Time of half-maximal SOX2 expression (h)') to the compiled feature manifest.

    Parameters
    ----------
    df_features: DataFrame
        Dataframe with intensity values extracted from all-cells masks

    Returns
    -------
    df_features_addons: DataFrame

    '''
    #filtering to 10 z-slices over which the mean intensity is calculated
    df_z=df_features[(df_features['Normalized Z plane']>=0) & (df_features['Normalized Z plane']<10)]

    #Grouping by condition and gene and each movie to get mean itnensity over time for each movie
    df_int=df_z.groupby(['Experimental Condition','Gene','Data ID','Timepoint']).agg({'Total intensity per Z':'sum','Area of all cells mask per Z (pixels)':'sum'}).reset_index()
    df_int['mean_intensity']=df_int['Total intensity per Z']/df_int['Area of all cells mask per Z (pixels)']

    ######--computing Time of max EOMES expression ------ #####
    df_eomes=df_int[df_int.Gene=='EOMES']
    Movie_ids_eomes, time_max_eomes=[],[]
    for id, df_id in df_eomes.groupby('Data ID'):
        df_id=df_id.sort_values('Timepoint')
        #smoothing the mean intensity curve
        df_id['int_smooth']=savgol_filter(df_id.mean_intensity.values,polyorder=2, window_length=10)
        int_max=max(df_id.int_smooth)
        t_max=df_id['Timepoint'][df_id.int_smooth==int_max].values[0]
        Movie_ids_eomes.append(id)
        time_max_eomes.append(t_max*(30/60))
    df_eomes_metrics=pd.DataFrame(zip(Movie_ids_eomes, time_max_eomes), columns=['Data ID','Time of max EOMES expression (h)'])

    ######--computing Time of max TBXT expression ------ #####
    print('computing TBXT expression')
    df_tbxt=df_int[df_int.Gene=='TBXT']
    Movie_ids_tbxt, time_max_tbxt=[],[]
    for id, df_id in df_tbxt.groupby('Data ID'):
        df_id=df_id.sort_values('Timepoint')
        #smoothing the mean intensity curve
        df_id['int_smooth']=savgol_filter(df_id.mean_intensity.values,polyorder=2, window_length=10)
        int_max=max(df_id.int_smooth)
        t_max=df_id['Timepoint'][df_id.int_smooth==int_max].values[0]
        Movie_ids_tbxt.append(id)
        time_max_tbxt.append(t_max*(30/60))
    df_tbxt_metrics=pd.DataFrame(zip(Movie_ids_tbxt, time_max_tbxt), columns=['Data ID','Time of max TBXT expression (h)'])

    ######--computing Time of inflection of E-cad expression ------ #####

    df_cdh=df_int[df_int.Gene=='CDH1']
    Movie_ids_cdh, time_inflection_cdh=[],[]
    for id, df_id in df_cdh.groupby('Data ID'):
        df_id=df_id.sort_values('Timepoint')
         #smoothing and getting second derivative of the mean intensity curve
        df_id['dy2']=savgol_filter(df_id['mean_intensity'].values,polyorder=2, window_length=40, deriv=2)
        d_filt=df_id[(df_id.Timepoint>=35)&(df_id.Timepoint<=78)]
        index_infl=d_filt['dy2'].idxmin() #identifying hte inflection point
        x_p=df_id['Timepoint'][index_infl]
        time_inflection_cdh.append(x_p*(30/60))
        Movie_ids_cdh.append(id)
    df_cdh_metrics=pd.DataFrame(zip(Movie_ids_cdh,time_inflection_cdh), columns=['Data ID','Time of inflection of E-cad expression (h)'])

    ######--computing Time of half-maximal SOX2 expression ------ #####
    df_sox=df_int[df_int.Gene=='SOX2']
    Movie_ids_sox, time_half_maximal_sox=[],[]
    for id, df_id in df_sox.groupby('Data ID'):
        df_id=df_id.sort_values('Timepoint')
        df_id['int_smooth']=savgol_filter(df_id.mean_intensity.values,polyorder=2, window_length=10)
        # Use the first timepoint intensity value - biologically meaningful
        int_50=(df_id.int_smooth.values[0]+min(df_id.int_smooth))/2
        t_50=min(df_id['Timepoint'][(df_id.int_smooth<=int_50)])
        Movie_ids_sox.append(id)
        time_half_maximal_sox.append(t_50*(30/60))
    df_sox_metrics=pd.DataFrame(zip(Movie_ids_sox, time_half_maximal_sox), columns=['Data ID','Time of half-maximal SOX2 expression (h)'])

    #merging eomes metrics with feature manifest
    df_metrics = pd.concat([df_eomes_metrics, df_tbxt_metrics, df_cdh_metrics, df_sox_metrics], ignore_index=True)
    df_features_addons=pd.merge(df_features, df_metrics, on=['Data ID'], how='left')

    return df_features_addons

# %% [markdown]
## master function to implement the pipeline
def compute_metrics(output_folder, load_from_aws: bool = True, local_imaging_csv: str = None):
    '''
    This is a master function that implements every function and post processing to save a compiled final manifest to be used with analysis_plots.py

    Parameters
    ----------
    output_folder: Path
        Path to the folder to save the final feature manifest

    load_from_aws: bool, default True
        If True, load imaging_and_segmentation_data from AWS S3.
        If False, load from local file.

    local_imaging_csv: str, optional
        Path to local imaging_and_segmentation_data.csv file.
        Only used when load_from_aws=False. If not provided, uses default local path.

    Returns
    -------
    df_features_final: DataFrame
        Returns and saves the final dataframe with all the required metrics fro analysis
    '''
    print('compiling intensity and z features into a single dataframe')

    df = io.load_bf_colony_features()
    #df = df[df['Data ID'] == '3500005824_35']

    print('computing glass information for normalized z position')
    df_all_z=add_bottom_z(df)
    print(len(df_all_z.index))

    print('merging the bottom z information with the colony mask path csv')
    Imaging_and_segmentation_data = io.load_imaging_and_segmentation_dataset(
        load_from_aws=load_from_aws,
        local_path=local_imaging_csv
    )

    # Pass only needed columns to add_bottom_mip_migration (reduces memory and speeds up groupby)
    df_for_area = df_all_z[['Data ID', 'Timepoint', 'Bottom Z plane']].drop_duplicates()
    df_for_area = pd.merge(
        df_for_area,
        Imaging_and_segmentation_data[['Data ID', 'All Cells Mask URL']],
        on='Data ID',
        how='left'
    )

    print('computing area at the glass (bottom 2 z MIP) and migration time')
    df_mm = add_bottom_mip_migration(df_for_area)
    print(len(df_mm.index))

    print('merging everything into a single feature manifest')
    # Merge area results back to full dataframe
    df_features = pd.merge(df_all_z, df_mm, on=['Data ID', 'Timepoint'], how='left')
    # Merge imaging metadata (only columns not already in df_features to avoid duplicates)
    existing_cols = set(df_features.columns)
    new_cols = ['Data ID'] + [col for col in Imaging_and_segmentation_data.columns
                               if col not in existing_cols]
    df_features = pd.merge(df_features, Imaging_and_segmentation_data[new_cols], on=['Data ID'], how='left')
    print(len(df_features.index))

    # Round migration onset times to nearest 0.5 hour to align with Time hr grid
    footprint_col = 'Migration Onset Time (Footprint Area Based)'
    if footprint_col in df_features.columns:
        df_features[footprint_col] = (df_features[footprint_col] * 2).round() / 2
        print(f'Rounded {footprint_col} to nearest 0.5 hour')

    io_col = 'Migration Onset Time (Inside/Outside Basement Membrane Based)'
    if io_col in df_features.columns:
        df_features[io_col] = (df_features[io_col] * 2).round() / 2
        print(f'Rounded {io_col} to nearest 0.5 hour')

    manual_col = 'Migration Onset Time (Manual First Cell Detection)'
    if manual_col in df_features.columns:
        df_features[manual_col] = (df_features[manual_col] * 2).round() / 2
        print(f'Rounded {manual_col} to nearest 0.5 hour')

    print('adding gene specific metrics...')
    df_features_addons=add_gene_metrics(df_features)

    # Reorder columns to put key analysis columns first, then metadata
    priority_columns = ['Data ID', 'Experimental Condition', 'Gene',
       'Single Colony Or Lumenoid At Time of Migration',
       'Absence Of Migrating Cells Coming From Colony Out Of FOV At Time Of Migration',
       'Timelapse Interval', 'Timepoint', 'Z plane',
       'Area of all cells mask per Z (pixels)',
       'Area of all cells mask per Z (square micrometer)',
       'Mean intensity per Z', 'Total intensity per Z',
       'Mean intensity per Z (Channel 2)', 'Total intensity per Z (Channel 2)',
       'Mean intensity per Z (Channel 3)', 'Total intensity per Z (Channel 3)',
       'Bottom Z plane',
       'Normalized Z plane', 'Area at the glass (pixels)',
       'Area at the glass(square micrometer)',
       'Time of max EOMES expression (h)',
       'Time of max TBXT expression (h)',
       'Time of inflection of E-cad expression (h)',
       'Time of half-maximal SOX2 expression (h)',
       'Migration Onset Time (Inside/Outside Basement Membrane Based)',
       'Migration Onset Time (Manual First Cell Detection)']

    # Get priority columns that exist, then add remaining columns
    existing_priority = [col for col in priority_columns if col in df_features_addons.columns]
    other_columns = [col for col in df_features_addons.columns if col not in priority_columns]
    all_columns = existing_priority + other_columns

    df_features_final = df_features_addons[all_columns]
    print(f"Final columns: {len(df_features_final.columns)}")
    print(len(df_features_final.index))

    # Add metadata-only movies (those without image data but with metadata)
    print('adding metadata-only movies...')
    df_features_final = add_metadata_only_movies(df_features_final, Imaging_and_segmentation_data)
    print(f"Total Data IDs after adding metadata-only movies: {df_features_final['Data ID'].nunique()}")

    print('saving the final feature file')
    df_features_final.to_csv(output_folder / f"Image_analysis_extracted_features.csv", index=False)



# %% [markdown]
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute metrics for EMT analysis')
    parser.add_argument('--local', action='store_true',
                        help='Load imaging_and_segmentation_data from local file instead of AWS')
    parser.add_argument('--local-csv', type=str, default=None,
                        help='Path to local imaging_and_segmentation_data.csv (only used with --local)')
    args = parser.parse_args()

    base_results_dir = io.setup_base_directory_name("metric_computation")
    df_features_all = compute_metrics(
        output_folder=base_results_dir,
        load_from_aws=not args.local,
        local_imaging_csv=args.local_csv
    )
