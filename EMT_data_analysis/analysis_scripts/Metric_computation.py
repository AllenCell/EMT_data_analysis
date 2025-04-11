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
    for id, df_id in tqdm(area_time.groupby('Movie ID')):
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


def add_bottom_mip(df_merged):
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
    for id, df_id in tqdm(df_merged.groupby('Movie ID')):
        ar_v,tp=[],[]

        l = df_id['Timepoint'].max()
        if l>97:
            l=97

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

    ######--computing Time of max EOMES expression ------ #####
    df_eomes=df_int[df_int.Gene=='EOMES|TBR2']
    Movie_ids_eomes, time_max_eomes=[],[]
    for id, df_id in df_eomes.groupby('Movie ID'):
        df_id=df_id.sort_values('Timepoint')
        #smoothing the mean intensity curve
        df_id['int_smooth']=savgol_filter(df_id.mean_intensity.values,polyorder=2, window_length=10) 
        int_max=max(df_id.int_smooth)
        t_max=df_id['Timepoint'][df_id.int_smooth==int_max].values[0]
        Movie_ids_eomes.append(id)
        time_max_eomes.append(t_max*(30/60))
    df_eomes_metrics=pd.DataFrame(zip(Movie_ids_eomes, time_max_eomes), columns=['Movie ID','Time of max EOMES expression (h)'])

    ######--computing Time of inflection of E-cad expression ------ #####

    df_cdh=df_int[df_int.Gene=='CDH1']
    Movie_ids_cdh, time_inflection_cdh=[],[]
    for id, df_id in df_cdh.groupby('Movie ID'):
        df_id=df_id.sort_values('Timepoint')
         #smoothing and getting second derivative of the mean intensity curve
        df_id['dy2']=savgol_filter(df_id['mean_intensity'].values,polyorder=2, window_length=40, deriv=2)
        d_filt=df_id[(df_id.Timepoint>=35)&(df_id.Timepoint<=78)]
        index_infl=d_filt['dy2'].idxmin() #identifying hte inflection point
        x_p=df_id['Timepoint'][index_infl]
        time_inflection_cdh.append(x_p*(30/60))
        Movie_ids_cdh.append(id)
    df_cdh_metrics=pd.DataFrame(zip(Movie_ids_cdh,time_inflection_cdh), columns=['Movie ID','Time of inflection of E-cad expression (h)'])

    ######--computing Time of half-maximal SOX2 expression ------ #####
    df_sox=df_int[df_int.Gene=='SOX2']
    Movie_ids_sox, time_half_maximal_sox=[],[]
    for id, df_id in df_sox.groupby('Movie ID'):
        df_id=df_id.sort_values('Timepoint')
        df_id['int_smooth']=savgol_filter(df_id.mean_intensity.values,polyorder=2, window_length=10) 
        int_50=(max(df_id.int_smooth)+min(df_id.int_smooth))/2
        t_50=min(df_id['Timepoint'][(df_id.int_smooth<=int_50)])
        Movie_ids_sox.append(id)
        time_half_maximal_sox.append(t_50)
    df_sox_metrics=pd.DataFrame(zip(Movie_ids_sox, time_half_maximal_sox), columns=['Movie ID','Time of half-maximal SOX2 expression (h)'])

    #merging eomes metrics with feature manifest
    df_features_addons=pd.merge(df_features, df_eomes_metrics, on=['Movie ID'], how='left').merge(df_cdh_metrics, on=['Movie ID'], how='left').merge(df_sox_metrics, on=['Movie ID'], how='left')

    return df_features_addons

# %% [markdown]
## master function to implement the pipeline


def compute_metrics(path_manifest, save_folder, final_feature_folder):
    print('compiling intensity and z features into a single dataframe')
    df=import_folder(save_folder)

    print(df.head())

    print('computing glass information for normalized z position')
    df_all_z=add_bottom_z(df)

    print('merging the bottom z information with the colony mask path csv')
    # df_z=df_all_z.groupby('id_tag')['z_bottom'].agg('first').reset_index()
    # path_manifest['id_tag'] = [f'{barcode}_{pos}-{well}' for barcode, pos, well in zip(path_manifest['Barcode'].values, path_manifest['Position'].values, path_manifest['Well'].values)]
    df_merged=pd.merge(df_all_z,path_manifest, how='left',on=['Movie ID', 'Timepoint', 'Gene', 'Experimental Condition'])

    print('computing area at the glass (bottom 2 z MIP) and migration time')
    df_mm=add_bottom_mip(df_merged)

    print('merging everything into a single feature manifest')
    df_features=pd.merge(df_all_z,df_mm, on=['Movie ID','Z plane','Timepoint'], suffixes=("","_remove"))
    df_features.drop([i for i in df_features.columns if 'remove' in i], axis=1, inplace=True)
    df_features.drop(['Unnamed: 0_x', 'Unnamed: 0', 'gene_channel_3', 'gene_channel_4', 'fms_id', 'id_tag', 'Mask_path', 'Unnamed: 0_y'], axis=1, inplace=True, errors='ignore')


    print('saving the final feature file')
    df_features.to_csv(rf'{final_feature_folder}/ImmunoPanel_entire_manifest.csv', index=False)
    return df_features


# %% [markdown]
## running the pipeline to generate and save feature manifest
path_manifest=pd.read_csv(r'/allen/aics/users/filip.sluzewski/Public_Repos/emt-data-analysis/EMT_EOMES-new-timelapse/7063/manifest.csv')
save_folder=r'/allen/aics/users/filip.sluzewski/Public_Repos/emt-data-analysis/EMT_EOMES-new-timelapse/7063/feature-extraction/'
final_feature_folder=r'/allen/aics/emt/data_analysis_plots/Colony_Metrics/Resubmission/7063/'
df_features=compute_metrics(path_manifest, save_folder, final_feature_folder)