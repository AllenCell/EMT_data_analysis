#%%[markdown]
## importing the required libraries

import numpy as np
import pandas as pd
import os

import warnings
warnings.filterwarnings("ignore")

import scipy.ndimage
from aicsimageio import AICSImage
from tqdm import tqdm

from aicsfiles import FileManagementSystem as fms

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
    area_time=df.groupby(['fms_id','z'])['area'].agg('sum').reset_index()
    file_id, z_bottom=[],[]
    for id, df_fms in tqdm(area_time.groupby('fms_id')):
        file_id.append(id)
        df_fms=df_fms.reset_index()
    
        raw_values=df_fms['area'].values
        dy=np.diff(raw_values)
        max_dy=np.max(dy)
        idx_max= np.where(dy== max_dy)[0]
        
        zo=df_fms['z'][idx_max].values[0]+1

        
        z_bottom.append(zo)
    df_bottom_z=pd.DataFrame(zip(file_id,z_bottom), columns=['fms_id','z_bottom'])

    df_merged=pd.merge(df,df_bottom_z, on=['fms_id'])

    df_merged['z_norm']=df_merged.apply(lambda x: x['z']-x['z_bottom'], axis=1)

    return df_merged





def add_bottom_mip_migration(df_merged):
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
    for id, df_id in tqdm(df_merged.groupby('fms_id')):
        ar_v,tp=[],[]
        z_bottom=df_id.z_bottom.values[0]

        if platform.system()!='Windows':
            folder=df_id.colony_mask_path.values[0]
            folder = Path(folder).as_posix()
        else:
            folder=df_id.colony_mask_path.values[0]
        print(folder)
        t, mask_path=[],[]
        for file in os.listdir(folder):
            if 'tif' in file:
                temp=file.split('=')[4]
                frame=int(temp.split('_')[0])
                t.append(frame)
                if platform.system()!='Windows':
                    mask_path.append(folder+'/'+file)
                else:
                    mask_path.append(folder+'\\'+file)
        df_seg=pd.DataFrame(zip(t, mask_path), columns=['Timepoint','Mask_path'])   

        l=len(t)
        if l>97:
            l=97

        for tm in np.arange(l):
            seg_path=df_seg['Mask_path'][df_seg.Timepoint==tm].values[0]
            img_seg=AICSImage(seg_path).data.squeeze()
            img_z=img_seg[z_bottom:z_bottom+2]
            z_max_proj = np.max(img_z,axis=0)
            img_fh=scipy.ndimage.binary_fill_holes(z_max_proj).astype(int)
        
            ar2=np.count_nonzero(img_fh)
            ar_v.append(ar2)
            tp.append(tm)
        df_area=pd.DataFrame(zip(tp,ar_v), columns=['Timepoint','Bottom_z_Area_pixels'])
        print('adding migration timing..')
        raw_values=df_area.Bottom_z_Area_pixels.values
        from scipy.signal import savgol_filter
        df_area['dy2']=savgol_filter(raw_values,polyorder=2, window_length=40, deriv=2)
        d_filt=df_area[(df_area.Timepoint>=35)&(df_area.Timepoint<=80)]
        index_infl=d_filt['dy2'].idxmax()

        x_p=df_area['Timepoint'][index_infl]
        df_area['Migration_hr']=x_p*(30/60)
        df_area['Bottom_z_mip']=df_area['Bottom_z_Area_pixels']*(0.271*0.271)
        df_area['fms_id']=id
        df_merged_area=pd.merge(df_id,df_area, on=['fms_id'])
        df_mm=pd.concat([df_mm,df_merged_area])

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
    df_merged=pd.merge(df_z,path_manifest, how='left',on=['fms_id'])

    print('computing area at the glass (bottom 2 z MIP) and migration time')
    df_mm=add_bottom_mip_migration(df_merged)

    print('merging everything into a single feature manifest')
    df_features=pd.merge(df_all_z,df_mm, on=['fms_id','Timepoint'], suffixes=("","_remove"))
    df_features.drop([i for i in df_features.columns if 'remove' in i], axis=1, inplace=True)

    n_movies=df_features.fms_id.nunique()
    print('saving the final feature file')
    df_features.to_csv(rf'{final_feature_folder}/Final_v3_{n_movies}_entire_manifest.csv')
    return df_features


# %% [markdown]
## running the pipeline to generate and save feature manifest
path_manifest=pd.read_csv(r'/allen/aics/assay-dev/users/Sandi/cyto-dl/data/bf_colony_seg/ms_colony_mask_389movies_manifest_v0.csv')
save_folder=r'/allen/aics/assay-dev/users/Filip/Data/EMT-colony-mask-features'
final_feature_folder=r'/allen/aics/assay-dev/users/Nivedita/EMT/EMT_deliverable/BF_colony_mask/Manifests'
df_features=compute_metrics(path_manifest, save_folder, final_feature_folder)
# %%
