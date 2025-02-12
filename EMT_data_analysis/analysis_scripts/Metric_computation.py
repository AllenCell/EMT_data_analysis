#%%[markdown]
## importing the required libraries

import numpy as np
import pandas as pd
import os

import warnings
warnings.filterwarnings("ignore")

import scipy.ndimage
from scipy.signal import savgol_filter
from bioio import BioImage
from tqdm import tqdm

import platform
from pathlib import Path


# %% [markdown]
## Defining the required functions



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
    for file in os.listdir(folder_path):
        f1=pd.read_csv(folder_path+'/'+file)
        f1['id_tag'] = Path(file).name.split('_')[-2] + '_' + f1['scene'].values[0]
        df=pd.concat([df,f1])
    return df



######----adding normalized Z-------#####
def add_bottom_z(df):
    """
    This function adds bottom Z defined from the sum of area over all time points vs Z
    
    Parameters
    ----------
    df: DataFrame
        Dataframe obtained from BF colony mask 

    Returns
    -------
    df_merged: DataFrame
        Returns the input DataFrame with 'z_norm' and 'z_bottom' columns"""
    
    df['area']=df['area_pixels']*(0.271*0.271)
    area_time=df.groupby(['id_tag','z'])['area'].agg('sum').reset_index()
    file_id, z_bottom=[],[]
    for id, df_id in tqdm(area_time.groupby('id_tag')):
        file_id.append(id)
        df_id=df_id.reset_index()
    
        raw_values=df_id['area'].values
        dy=np.diff(raw_values)
        max_dy=np.max(dy)
        idx_max= np.where(dy== max_dy)[0]
        
        zo=df_id['z'][idx_max].values[0]+1

        
        z_bottom.append(zo)
    df_bottom_z=pd.DataFrame(zip(file_id,z_bottom), columns=['id_tag','z_bottom'])

    df_merged=pd.merge(df,df_bottom_z, on=['id_tag'])

    df_merged['z_norm']=df_merged.apply(lambda x: x['z']-x['z_bottom'], axis=1)

    return df_merged





def add_bottom_mip(df_merged):
    '''
    This adds area of MIP of bottom 2zs to get area at the glass and compute migration time from that.
    
    Parameters
    ----------
    df_merged: DataFrame
        Dataframe with z_bottom column and colony mask paths for each movie

    Returns
    -------
    df_mm: DataFrame
        Returns the input DataFrame with 'z_norm' and 'z_bottom' columns added to df_merged dataframe'''
     
    df_mm=pd.DataFrame()
    for id, df_id in tqdm(df_merged.groupby('id_tag')):
        z_bottom=df_id.z_bottom.values[0]

        seg_path=df_id['Mask_path'].values[0]
        img_seg=BioImage(seg_path).data.squeeze()
        img_z=img_seg[z_bottom:z_bottom+2]
        z_max_proj = np.max(img_z,axis=0)
        img_fh=scipy.ndimage.binary_fill_holes(z_max_proj).astype(int)
    
        ar2=np.count_nonzero(img_fh)
        
        df_area = df_id.copy(deep=True)
        df_area['Bottom_z_Area_pixels'] = ar2
        df_mm=pd.concat([df_mm,df_area])

    return df_mm
    




# %% [markdown]
## master function to implement the pipeline


def compute_metrics(path_manifest, save_folder, final_feature_folder):
    print('compiling intensity and z features into a single dataframe')
    df=import_folder(save_folder)

    print('computing glass information for normalized z position')
    df_all_z=add_bottom_z(df)

    print('merging the bottom z information with the colony mask path csv')
    df_z=df_all_z.groupby('fms_id')['z_bottom'].agg('first').reset_index()
    path_manifest['id_tag'] = [f'{barcode}_{pos}_{well}' for barcode, pos, well in zip(path_manifest['Barcode'].values, path_manifest['Position'].values, path_manifest['Well'].values)]
    df_features=pd.merge(df_z,path_manifest, how='left',on=['id_tag'])

    # print('computing area at the glass (bottom 2 z MIP) and migration time')
    # df_mm=add_bottom_mip_migration(df_merged)

    # print('merging everything into a single feature manifest')
    # df_features=pd.merge(df_all_z,df_mm, on=['fms_id','Timepoint'], suffixes=("","_remove"))
    # df_features.drop([i for i in df_features.columns if 'remove' in i], axis=1, inplace=True)

    n_movies=df_features.id_tag.nunique()
    print('saving the final feature file')
    df_features.to_csv(rf'{final_feature_folder}/ImmunoPanel_{n_movies}_entire_manifest.csv')
    return df_features


# %% [markdown]
## running the pipeline to generate and save feature manifest
path_manifest=pd.read_csv(r'/allen/aics/assay-dev/users/Sandi/cyto-dl/data/bf_colony_seg/ms_colony_mask_389movies_manifest_v0.csv')
save_folder=r'/allen/aics/assay-dev/users/Filip/Data/EMT-colony-mask-features'
final_feature_folder=r'/allen/aics/assay-dev/users/Nivedita/EMT/EMT_deliverable/BF_colony_mask/Manifests'
df_features=compute_metrics(path_manifest, save_folder, final_feature_folder)
# %%
