#%%[markdown]
###### This script adds metrics to generate the entire feature manifest used for analysis_plot.py file. It requires the input of the folder path where the files from feature_extraction.py file are stored.
###### importing the required libraries

import numpy as np
import pandas as pd
import os

import warnings
warnings.filterwarnings("ignore")

import scipy.ndimage
from scipy.signal import savgol_filter
from bioio import BioImage
import bioio_ome_zarr
from tqdm import tqdm

# %% [markdown]
## Defining the required functions



def import_folder(folder_path):
    """
     This function compiles all the intensity metric csvs into a single manifest.
      Parameters
    ----------
    folder_path: Path
        Path to the folder where all the intensity metric csvs have been saved using the feature_extraction.py script

    Returns
    -------
    df: DataFrame
        Returns the compiled dataframe with area and intensity values per z extracted from all-cells masks """

    df=pd.DataFrame() 
    for file in os.listdir(folder_path):
        f1=pd.read_csv(folder_path+'/'+file)
        df=pd.concat([df,f1])
    return df



######----adding normalized Z to get the Z plane corresponding to the glass-------#####
def add_bottom_z(df):
    """
    This function adds bottom Z - Zplane corresponding to the glass- defined from the sum of area over all time points vs Z
    
    Parameters
    ----------
    df: DataFrame
        Dataframe obtained from feature extraction using all-cells masks

    Returns
    -------
    df_merged: DataFrame
        Returns the input DataFrame with'Normalized Z plane' and 'Bottom Z plane' columns"""
    
    df['Area of all cells mask per Z (square micrometer)']=df['Area of all cells mask per Z (pixels)']*(0.271*0.271)
    area_time=df.groupby(['Movie Unique ID','Z plane'])['Area of all cells mask per Z (square micrometer)'].agg('sum').reset_index()
    file_id, z_bottom=[],[]
    for id, df_id in tqdm(area_time.groupby('Movie Unique ID')):
        file_id.append(id)
        df_id=df_id.reset_index()
    
        raw_values=df_id['Area of all cells mask per Z (square micrometer)'].values
        dy=np.diff(raw_values)
        max_dy=np.max(dy)
        idx_max= np.where(dy== max_dy)[0]
        
        zo=df_id['Z plane'][idx_max].values[0]+1

        
        z_bottom.append(zo)
    df_bottom_z=pd.DataFrame(zip(file_id,z_bottom), columns=['Movie Unique ID','Bottom Z plane'])

    df_normalized_z=pd.merge(df,df_bottom_z, on=['Movie Unique ID'])

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
    for id, df_id in tqdm(df_merged.groupby('Movie Unique ID')):
        ar_v,tp=[],[]
        z_bottom=df_id['Bottom Z plane'].values[0]

        seg_path = df_id['s3 All Cells Mask BFF path'].values[0]

        img_seg = BioImage(seg_path)
        l = df_id['Image Size T'].values[0]
        if l>97:
            l=97

        for tm in np.arange(l):
            img_seg_tl = img_seg.get_image_dask_data("ZYX", T=tm)
            img_z=img_seg_tl[z_bottom:z_bottom+2]
            z_max_proj = np.max(img_z,axis=0)
            img_fh=scipy.ndimage.binary_fill_holes(z_max_proj).astype(int)
        
            ar2=np.count_nonzero(img_fh)
            ar_v.append(ar2)
            tp.append(tm)
        df_area=pd.DataFrame(zip(tp,ar_v), columns=['Timepoint','Area at the glass (pixels)'])
        
        raw_values=df_area['Area at the glass (pixels)'].values
        df_area['dy2']=savgol_filter(raw_values,polyorder=2, window_length=40, deriv=2)
        d_filt=df_area[(df_area.Timepoint>=35)&(df_area.Timepoint<=80)]
        index_infl=d_filt['dy2'].idxmax()

        x_p=df_area['Timepoint'][index_infl]
        df_area['Migration time (h)']=x_p*(30/60)
        df_area['Area at the glass(square micrometer)']=df_area['Area at the glass (pixels)']*(0.271*0.271)
        df_area['Movie Unique ID']=id
        df_merged_area=pd.merge(df_id,df_area, on=['Movie Unique ID'])
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
    df_int=df_z.groupby(['Experimental Condition','Gene','Movie Unique ID','Timepoint']).agg({'Total intensity per Z':'sum','Area at the glass (pixels)':'sum'}).reset_index()
    df_int['mean_intensity']=df_int['Total intensity per Z']/df_int['Area at the glass (pixels)']

    ######--computing Time of max EOMES expression ------ #####
    df_eomes=df_int[df_int.Gene=='EOMES|TBR2']
    Movie_ids_eomes, time_max_eomes=[],[]
    for id, df_id in df_eomes.groupby('Movie Unique ID'):
        df_id=df_id.sort_values('Timepoint')
        #smoothing the mean intensity curve
        df_id['int_smooth']=savgol_filter(df_id.mean_intensity.values,polyorder=2, window_length=10) 
        int_max=max(df_id.int_smooth)
        t_max=df_id['Timepoint'][df_id.int_smooth==int_max].values[0]
        Movie_ids_eomes.append(id)
        time_max_eomes.append(t_max*(30/60))
    df_eomes_metrics=pd.DataFrame(zip(Movie_ids_eomes, time_max_eomes), columns=['Movie Unique ID','Time of max EOMES expression (h)'])

    ######--computing Time of inflection of E-cad expression ------ #####

    df_cdh=df_int[df_int.Gene=='CDH1']
    Movie_ids_cdh, time_inflection_cdh=[],[]
    for id, df_id in df_cdh.groupby('Movie Unique ID'):
        df_id=df_id.sort_values('Timepoint')
         #smoothing and getting second derivative of the mean intensity curve
        df_id['dy2']=savgol_filter(df_id['mean_intensity'].values,polyorder=2, window_length=40, deriv=2)
        d_filt=df_id[(df_id.Timepoint>=35)&(df_id.Timepoint<=78)]
        index_infl=d_filt['dy2'].idxmin() #identifying hte inflection point
        x_p=df_id['Timepoint'][index_infl]
        time_inflection_cdh.append(x_p*(30/60))
        Movie_ids_cdh.append(id)
    df_cdh_metrics=pd.DataFrame(zip(Movie_ids_cdh,time_inflection_cdh), columns=['Movie Unique ID','Time of inflection of E-cad expression (h)'])

    ######--computing Time of half-maximal SOX2 expression ------ #####
    df_sox=df_int[df_int.Gene=='SOX2']
    Movie_ids_sox, time_half_maximal_sox=[],[]
    for id, df_id in df_sox.groupby('Movie Unique ID'):
        df_id=df_id.sort_values('Timepoint')
        df_id['int_smooth']=savgol_filter(df_id.mean_intensity.values,polyorder=2, window_length=10) 
        int_50=(max(df_id.int_smooth)+min(df_id.int_smooth))/2
        t_50=min(df_id['Timepoint'][(df_id.int_smooth<=int_50)])
        Movie_ids_sox.append(id)
        time_half_maximal_sox.append(t_50)
    df_sox_metrics=pd.DataFrame(zip(Movie_ids_sox, time_half_maximal_sox), columns=['Movie Unique ID','Time of half-maximal SOX2 expression (h)'])

    #merging eomes metrics with feature manifest
    df_features_addons=pd.merge(df_features, df_eomes_metrics, on=['Movie Unique ID'], how='left').merge(df_cdh_metrics, on=['Movie Unique ID'], how='left').merge(df_sox_metrics, on=['Movie Unique ID'], how='left')

    return df_features_addons

# %% [markdown]
## master function to implement the pipeline


def compute_metrics(Imaging_and_segmentation_data , all_cells_feature_csvs_folder, final_feature_folder):
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
    print('compiling intensity and z features into a single dataframe')
    df=import_folder(all_cells_feature_csvs_folder)


    print('computing glass information for normalized z position')
    df_all_z=add_bottom_z(df)

    print('merging the bottom z information with the colony mask path csv')
    df_z=df_all_z.groupby('Movie Unique ID')['Bottom Z plane'].agg('first').reset_index()
    df_merged=pd.merge(df_z,Imaging_and_segmentation_data, how='left',on=['Movie Unique ID'])


    print('computing area at the glass (bottom 2 z MIP) and migration time')
    df_mm=add_bottom_mip_migration(df_merged)

    print('merging everything into a single feature manifest')
    df_features=pd.merge(df_all_z,df_mm, on=['Movie Unique ID','Timepoint'], suffixes=("","_remove"))
    df_features.drop([i for i in df_features.columns if 'remove' in i], axis=1, inplace=True)

    print('adding gene specific metrics...')
    df_features_addons=add_gene_metrics(df_features)
    #only including the columns of interest
    df_features_final=df_features_addons[['Movie Unique ID', 'Experimental Condition', 'Gene',
       'Single Colony Or Lumenoid At Time of Migration',
       'Absence Of Migrating Cells Coming From Colony Out Of FOV At Time Of Migration',
       'Timelapse Interval', 'Timepoint', 'Z plane',
       'Area of all cells mask per Z (pixels)',
       'Area of all cells mask per Z (square micrometer)',
       'Mean intensity per Z', 'Total intensity per Z', 'Bottom Z plane',
       'Normalized Z plane', 'Area at the glass (pixels)',
       'Area at the glass(square micrometer)', 'Migration time (h)',
       'Time of max EOMES expression (h)',
       'Time of inflection of E-cad expression (h)',
       'Time of half-maximal SOX2 expression (h)']]

    n_movies=df_features_final['Movie Unique ID'].nunique()
    print('saving the final feature file')
    df_features_final.to_csv(rf'{final_feature_folder}/Feature_manifest_number_of_movies_{n_movies}_test.csv')
    return df_features_final


# %% [markdown]
##### running the pipeline to generate and save feature manifest
Imaging_and_segmentation_data=pd.read_csv(r'/allen/aics/assay-dev/users/Nivedita/EMT/EMT_deliverable/BF_colony_mask/Manifests/Dataset_1_V20_V8_SM.csv')
all_cells_feature_csvs_folder=r'/allen/aics/assay-dev/users/Filip/Data/EMT-colony-mask-features'
final_feature_folder=r'/allen/aics/assay-dev/users/Nivedita/EMT/EMT_deliverable/BF_colony_mask/Manifests'

df_features_all=compute_metrics(Imaging_and_segmentation_data , all_cells_feature_csvs_folder, final_feature_folder)