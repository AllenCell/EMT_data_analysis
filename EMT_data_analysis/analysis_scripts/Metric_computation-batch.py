#%%[markdown]
'''
This script adds metrics to generate the entire feature manifest used
for analysis_plot.py file. It requires the input of the folder path where
the files from feature_extraction.py file are stored.
'''
import warnings
import numpy as np
import pandas as pd
import scipy.ndimage
from tqdm import tqdm
from bioio import BioImage
from scipy.signal import savgol_filter
from EMT_data_analysis.tools import io
from pathlib import Path

warnings.filterwarnings("ignore")

# %% [markdown]

def import_folder(folder_path):
    """
     This function compiles all the intensity metric csvs into a single manifest.
      Parameters
    ----------
    folder_path: Path
        Path to the folder where all the intensity metric csvs were been saved

    Returns
    -------
    df: DataFrame
        Returns the input DataFrame with 'z_norm' and 'z_bottom' columns """

    df=pd.DataFrame() 
    for file in Path(folder_path).glob('*.csv'):
        f1=pd.read_csv(file, index_col=0)
        df=pd.concat([df,f1])
        
    df['Gene'] = df['Gene'].apply(lambda x: 'EOMES' if 'EOMES' in x else x)
    df['Experimental Condition'] = df['Experimental Condition'].apply(lambda x: '2D PLF colony EMT' if x=='2D PLF EMT 1:60 MG' else '3D lumenoid EMT' if x=='3D MG EMT 1:60 MG' else x)

    return df

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
    area_time=df.groupby(['Movie ID','Z plane'])['Area of all cells mask per Z (square micrometer)'].agg('sum').reset_index()
    file_id, z_bottom=[],[]
    for id, df_id in area_time.groupby('Movie ID'):
        file_id.append(id)
        df_id=df_id.reset_index()
    
        raw_values=df_id['Area of all cells mask per Z (square micrometer)'].values
        dy=np.diff(raw_values)
        max_dy=np.max(dy)
        idx_max= np.where(dy== max_dy)[0]
        
        zo=df_id['Z plane'][idx_max].values[0]+1

        z_bottom.append(zo)

    df_bottom_z=pd.DataFrame(zip(file_id,z_bottom), columns=['Movie ID','Bottom Z plane'])

    df_normalized_z=pd.merge(df,df_bottom_z, on=['Movie ID'])

    df_normalized_z['Normalized Z plane']=df_normalized_z.apply(lambda x: x['Z plane']-x['Bottom Z plane'], axis=1)

    return df_normalized_z


def add_bottom_mip_migration(df_merged):
    '''
    This adds area of MIP of bottom 2Z planes to get area at the glass and compute migration time from that.
    
    Parameters
    ----------
    df_merged: DataFrame
        Dataframe with Bottom Z plane column and All-cells mask paths for each movie (merging df_normalized_z with Imaging_and_segmentation_data.csv)

    Returns
    -------
    df_mm: DataFrame
        Returns the input DataFrame with 'Area at the glass (pixels)','Area at the glass(square micrometer)' and 'Migration time (h)' columns
        '''
     
    df_mm=pd.DataFrame()
    for id, df_id in df_merged.groupby('Movie ID'):
        ar_v,tp=[],[]

        l = df_id['Timepoint'].max()
        # if l>97:
            # l=97

        for t, df_tp in df_id.groupby('Timepoint'):
            if t > l:
                break
            z_bottom=df_tp['Bottom Z plane'].values[0]
            img_seg = BioImage(df_tp['All Cells Mask URL'].values[0])
            img_seg_tl = img_seg.get_image_dask_data("ZYX")
            img_z=img_seg_tl[z_bottom:z_bottom+2]
            z_max_proj = np.max(img_z,axis=0)
            img_fh=scipy.ndimage.binary_fill_holes(z_max_proj).astype(int)
        
            ar2=np.count_nonzero(img_fh)
            ar_v.append(ar2)
            tp.append(t)
        df_area=pd.DataFrame(zip(tp,ar_v), columns=['Timepoint','Area at the glass (pixels)'])
        
        raw_values=df_area['Area at the glass (pixels)'].values
        df_area['dy2']=savgol_filter(raw_values,polyorder=2, window_length=40, deriv=2)
        d_filt=df_area[(df_area.Timepoint>=35)&(df_area.Timepoint<=80)]
        index_infl=d_filt['dy2'].idxmax()

        x_p=df_area['Timepoint'][index_infl]
        df_area['Migration time (h)']=x_p*(30/60)
        df_area['Area at the glass(square micrometer)']=df_area['Area at the glass (pixels)']*(0.271*0.271)
        df_area['Movie ID']=id
        df_merged_area=pd.merge(df_id,df_area, on=['Movie ID','Timepoint'])
        df_mm=pd.concat([df_mm,df_merged_area])

    return df_mm
    

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
    df_int=df_z.groupby(['Experimental Condition','Gene','Movie ID','Timepoint']).agg({'Total intensity per Z':'sum','Area at the glass (pixels)':'sum'}).reset_index()
    df_int['mean_intensity']=df_int['Total intensity per Z']/df_int['Area at the glass (pixels)']

    Movie_ids, time_max=[],[]
    for id, df_id in df_int.groupby('Movie ID'):
        df_id=df_id.sort_values('Timepoint')
        #smoothing the mean intensity curve
        df_id['int_smooth']=savgol_filter(df_id.mean_intensity.values,polyorder=2, window_length=10) 
        int_max=max(df_id.int_smooth)
        try:
            t_max=df_id['Timepoint'][df_id.int_smooth==int_max].values[0]
        except:
            t_max = 0
        Movie_ids.append(id)
        time_max.append(t_max*(30/60))
    df_metrics=pd.DataFrame(zip(Movie_ids, time_max), columns=['Movie ID','Time of max expression (h)'])

    #merging eomes metrics with feature manifest
    df_features_addons=pd.merge(df_features, df_metrics, on=['Movie ID'], how='left')

    return df_features_addons

# %% [markdown]
## master function to implement the pipeline
def compute_metrics(path_manifest, df, out_csv):
    '''
    This is a master function that implements every function and post processing to save a compiled final manifest to be used with analysis_plots.py

    Parameters
    ----------
    Imaging_and_segmentation_data: DataFrame
        Dataframe with imaging and segmentation information for each movie

    all_cells_feature_csvs_folder: Folder path
        Path to the folder where csvs per movie for the features extracted from all-cells masks is stored

    final_feature_folder: folder path
        Path to the folder to save the final feature manifest
    Returns
    -------
    df_features_final: DataFrame
        Returns and saves the final dataframe with all the required metrics fro analysis
    '''
    # print('compiling intensity and z features into a single dataframe')

    # df=import_folder(save_folder)
    func = lambda x: x.replace('.0','')
    df['Movie ID'] = df['Movie ID'].map(func)
    path_manifest['Movie ID'] = path_manifest['Movie ID'].map(func)
    job_barcode = path_manifest['Plate Barcode'].values[0]
    out_csv = Path(out_csv)

    if out_csv.exists():
        # print(f'Metrics already exist for {job_barcode}')
        df_features_final = pd.read_csv(out_csv, index_col=None)
    else:
        df_features_final = None

    # df = pd.concat([df[df['Movie ID'] == m_id] for m_id in path_manifest['Movie ID'].unique()], ignore_index=True)
    df = df[df['Timepoint'] <= 97]
    # print(len(df.index))
    
    # print('computing glass information for normalized z position')
    df_all_z=add_bottom_z(df)
    # print(len(df_all_z.index))

    # print('merging the bottom z information with the colony mask path csv')
    # df_all_z = df_all_z.groupby('Movie ID')['Bottom Z plane'].agg('first').reset_index()
    
    df_merged=pd.merge(df_all_z,path_manifest, how='left',on=['Movie ID', 'Timepoint'])

    # print('computing area at the glass (bottom 2 z MIP) and migration time')
    df_mm=add_bottom_mip_migration(df_merged)

    # print('merging everything into a single feature manifest')
    df_features=pd.merge(df_all_z,df_mm, on=['Movie ID','Timepoint','Z plane'], suffixes=("","_remove"), how='left')
    df_features.drop([i for i in df_features.columns if 'remove' in i], axis=1, inplace=True)

    # print('adding gene specific metrics...')
    df_features_addons=add_gene_metrics(df_features)
    #only including the columns of interest
    features = ['Movie ID', 'Experimental Condition', 'Gene',
       'Single Colony Or Lumenoid At Time of Migration',
       'Absence Of Migrating Cells Coming From Colony Out Of FOV At Time Of Migration',
       'Timelapse Interval', 'Timepoint', 'Z plane',
       'Area of all cells mask per Z (pixels)',
       'Area of all cells mask per Z (square micrometer)',
       'Mean intensity per Z', 'Total intensity per Z', 'Bottom Z plane',
       'Normalized Z plane', 'Area at the glass (pixels)',
       'Area at the glass(square micrometer)', 'Migration time (h)',
       'Time of max expression (h)']
    features = [feat for feat in features if feat in df_features_addons.columns]
    
    if df_features_final is None:
        df_features_final=df_features_addons[features]
    else:
        df_features_final = pd.concat([df_features_final,df_features_addons[features]], ignore_index=True)
    
    # print(len(df_features_final.index))

    # print('saving the final feature file')
    Path(out_csv.parent).mkdir(parents=True, exist_ok=True)
    df_features_final.to_csv(out_csv, index=False)
    print(job_barcode, ' saved to ', out_csv)
    return



# %% [markdown]
## running the pipeline to generate and save feature manifest
def main():
    dataset_manifest=pd.read_csv(f'/allen/aics/users/filip.sluzewski/Public_Repos/emt-data-analysis/EMT-new-timelapse/final_new_release_April_24_w_raw_acm.csv')
    dataset_manifest['Movie ID'] = dataset_manifest.apply(lambda x: x['Plate Barcode'] + '_P' + str(int(x['Position Index'])) + '-' + x['Well Label'], axis=1)

    barcode_dir = Path('/allen/aics/users/filip.sluzewski/Public_Repos/emt-data-analysis/EMT-new-timelapse/metric')
    tasks = []
    for csv_dir in tqdm(barcode_dir.iterdir(), desc='Compiling jobs'):
        if not csv_dir.is_dir():
            continue

        feature_dir = csv_dir / 'feature-extraction'
        df_features = pd.concat([pd.read_csv(fn,index_col=None) for fn in feature_dir.glob('*.csv')], ignore_index=True)
        func = lambda x: x.replace('.0','')
        df_features['Movie ID'] = df_features['Movie ID'].map(func)
        df_features['Gene'] = df_features['Gene'].apply(lambda x: 'EOMES' if 'EOMES' in x else 'H2B' if 'H2B' in x else x)
        df_features['Experimental Condition'] = df_features['Experimental Condition'].apply(lambda x: '2D PLF colony EMT' if x=='2D PLF EMT 1:60 MG' else '3D lumenoid EMT' if x=='3D MG EMT 1:60 MG' else '2D colony EMT' if x=='2D MG EMT 1:60 MG' else x)

        
        gene = df_features['Gene'].unique()[0]
        barcode_abrv = csv_dir.name
        out_csv = Path(f'/allen/aics/emt/data_analysis_plots/Colony_Metrics/Resubmission/{gene}/{barcode_abrv}/Image_analysis_extracted_features_final.csv')
        if out_csv.exists():
            df_complete = pd.read_csv(out_csv, index_col=None)
            mov_complete = df_complete['Movie ID'].unique()
            mov_todo = [mov for mov in df_features['Movie ID'].unique() if mov not in mov_complete]
            if len(mov_todo)==0:
                print('No movies to process for ', barcode_abrv)
                continue
            df_features = pd.concat([df_features[df_features['Movie ID']==m_id] for m_id in mov_todo], ignore_index=True)
        movie_ids = df_features['Movie ID'].unique()
        
        print('\n', len(movie_ids), ' timelapses to do in barcode ', barcode_abrv)
        df_barcode = pd.concat([dataset_manifest[dataset_manifest['Movie ID']==m_id] for m_id in movie_ids], ignore_index=True)
        print(len(df_barcode['Movie ID']), ' scenes in master file')
        if len(df_barcode['Movie ID'])==0:
            continue

        df_features = pd.concat([df_features[df_features['Movie ID']==m_id] for m_id in df_barcode['Movie ID'].unique()])
        df_tps = []
        for _, row in df_barcode.iterrows():
            seg_path = Path(row['ACM_path'])
            for seg_fn in seg_path.glob('*.tif'):
                tp = pd.DataFrame(row.copy(deep=True)).transpose()
                tp['Timepoint'] = int(seg_fn.stem.split('_')[-1])
                tp['All Cells Mask URL'] = seg_fn
                df_tps.append(tp)
        df_tps = pd.concat(df_tps, ignore_index=True)

        tasks.append((
            df_tps, 
            df_features,
            out_csv
        ))

    # for task in tasks:
    #     compute_metrics(*task)

    # return

    import traceback
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=40) as executor:
        futures = [executor.submit(compute_metrics, *task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Barcodes"):
            try:
                _ = future.result()
            except Exception as e:
                print(f"Error processing timepoint: {e}")
                print(traceback.format_exc())


if __name__ == '__main__':
    main()