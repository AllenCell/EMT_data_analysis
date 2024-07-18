######---------importing libraries--------#######
import numpy as np
import pandas as pd
import os
import pickle as pkl

import warnings
warnings.filterwarnings("ignore")

import scipy.ndimage
from skimage.transform import SimilarityTransform, warp
from aicsimageio import AICSImage
from tqdm import tqdm



#######---extracting area and intensity values for every z-----####--TAKES THE MOST TIME
from aicsfiles import FileManagementSystem 
fms=FileManagementSystem.from_env('prod')
import platform
from pathlib import Path

def compute_bf_colony_features(df, save_folder, align=True):
    '''
    This function  computes area of the BF colony mask at every z position and also extracts corresponding intensity values from the fluorescence channel. It also adds other features from the BF colony mask.
    Parameters
    ----------
    df: DataFrame
        Dataframe with colony mask paths for each movie

    save_folder: path
        Folder path where feature csv for each movie is stored

    align_image: Bool
        Flag to enable alignment of the image using the barcode of the movie


    Returns
    -------
    saves feature files for each movie in the mentioned folder'''


    for fms_id, df_fms in tqdm(df.groupby('fms_id')):
    #importing raw image
        print(f'FMS_id-{fms_id}')
        print('Getting raw data...')
        file_fms_id=df_fms.fms_id.values[0]
        record = list(fms.find(
            annotations={"File Id":file_fms_id},
            limit=1,
        ))[0]
        
        file_path=record.path
        if platform.system()=='Windows':
            path_w=file_path.replace('/','\\')
            img=AICSImage(repr(path_w)[1:-1])
        else:
            img=AICSImage(file_path)
    
        print('Getting colony mask....')
        if platform.system()!='Windows':
            folder = df_fms.colony_mask_path.values[0]
            folder = Path(folder).as_posix()
        else:
            folder = df_fms.colony_mask_path.values[0]
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
        
        print('Computing features....')
        df_cr=pd.DataFrame()
        l=len(t)
        if l>97:
            l=97
        for time in tqdm(np.arange(l)):
            img_tl=img.get_image_dask_data("ZYX", C=1, T=time)
            img_raw = img_tl.compute() 
            seg_path=df_seg['Mask_path'][df_seg.Timepoint==time].values[0]
                
            img_seg=AICSImage(seg_path).data.squeeze()
            
            if align:
                barcode = list(record.annotations['Plate Barcode'])[0]
                transform = get_alignment_matrix(barcode)
                transform = transform.inverse
            
            s_z=int(img_seg.shape[0])
            z,area,mean_int, total_int, var_int=[],[],[],[],[]
            for i in np.arange(s_z):
                z.append(i)
                seg_z = img_seg[i]
                if align:
                    seg_z = align_image(seg_z, transform)
                
                # select=np.where(seg_z, img_raw[i], 0 )
                mask=np.bool_(seg_z)
                img_int=img_raw[i]
                intensity=np.mean(img_int[mask])
                ar=np.count_nonzero(mask)
                total=np.sum(img_int[mask])
                var=np.var(img_int[mask])

                area.append(ar)
                total_int.append(total)
                mean_int.append(intensity)
                var_int.append(var)
            df_prop=pd.DataFrame(zip(z,area,mean_int,total_int,var_int), columns=['z','area_pixels','mean_intensity','total_intensity','Variance_intensity'])
            z_proj=np.count_nonzero(img_seg, axis=0)
            m_z=np.ma.masked_equal(z_proj,0)
            z_max_proj = np.max(img_seg,axis=0)
            ar2=np.count_nonzero(z_max_proj) 
            df_prop['MIP_area']=ar2
            df_prop['z_median']=np.ma.median(m_z)
            df_prop['z_mean']=np.ma.mean(m_z)
            df_prop['z_max']=np.ma.max(m_z)
            df_prop['Timepoint']=time
            df_cr=pd.concat([df_cr,df_prop])
        df_cr['fms_id']=fms_id
        df_cr['gene']=df_fms.gene.values[0]
        df_cr['Condition']=df_fms.fms_condition.values[0]
        df_cr.to_csv(Path(save_folder) / f'Features_bf_colony_mask_{fms_id}.csv')


#######----compiling all movies in a single dataset----####
def import_folder(folder_path):
    df=pd.DataFrame() 
    for file in os.listdir(folder_path):
        f1=pd.read_csv(folder_path+'/'+file)
        df=pd.concat([df,f1])
    return df

#######----apply alignment matrix to 2D image----####
def align_image(
        img: np.ndarray, 
        transform: SimilarityTransform   
    ):
    '''
    This function aligns an image according to the camera alignment matrix for its barcode.
    
    Parameters
    ----------
    img: np.ndarray
        Image to be aligned. Image assumed to be YX.
    transform: SimilarityTransform
        Transformation matrix to align the image.
    '''
    return warp(img, transform, order=0, preserve_range=True)

def get_alignment_matrix(barcode, alignment_folder='/allen/aics/assay-dev/users/Filip/Data/EMT-alignment-matrices/alignment_info/'):
    '''
    This function returns the alignment matrix for a given barcode.
    
    Parameters
    ----------
    barcode: str
        Barcode of the image.
    alignment_folder: str
        Folder path where alignment matrices are stored.
    '''
    matrix = pkl.load(open(f'{alignment_folder}/{barcode}_alignmentmatrix.pkl', 'rb'))
    return SimilarityTransform(matrix=matrix)


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
    

def compute_all_features(path_manifest, save_folder, intensity_extraction=True, compiling_all=True, bottom_z=True, add_migration=True ):
    '''
    Master function to extract area and intensity from colony masks and corresponding fluorescent images and calculating area MIP and migration timing.

    Parameters
    ----------
    path_manifest : DataFrame
        DataFrame containing fms_ids of the movies and the corresponding folder path where bf colony masks are stored.

        
    save_folder: path
        Folder path to sve the individual feature csvs for each movie

    itensity_extraction: Bool
        Flag to enable extraction of area and intensity per z using BF colony mask

    compiling_all: Bool
        Flag to enable compiling all feature csvs in to a single manifest

    add_bottom_z: Bool
        Flag to add z values corresponding to detection of glass based on BF colony mask

    add_migration:  Bool
        Flag to add area at the glass- area of MIP for bottom 2 Z and migration timing calculated from area at glass metric

    Returns
    -------
    df_features: DataFrame
        Returns a DataFrame with all the computed features- ready for analysis

        '''
    #computing area and intensity features
   
    if intensity_extraction==True:
        print('Beginning intensity and area extraction....')
        compute_bf_colony_features(path_manifest, save_folder)

        print('Individual movie features saved....')

    if compiling_all:
        print('Compiling all movies into a single manifest....')
        df=import_folder(save_folder)

    if bottom_z:
        print('Computing glass information for normalized z position....')
        df_all_z=add_bottom_z(df)

    
    if add_migration:
    #adding MIP area of bottom 2 z and migration timing -----#####

        df_z=df_all_z.groupby('fms_id')['z_bottom'].agg('first').reset_index()


    #merging the bottom z information with the colony mask path csv
        df_merged=pd.merge(path_manifest,df_z, how='left',on=['fms_id'])

        print('Adding bottom 2z area MIP and corresponding migration time....')

        df_mm=add_bottom_mip_migration(df_merged)

        df_features=pd.merge(df_all_z,df_mm, on=['fms_id','Timepoint'], suffixes=("","_remove"))
        df_features.drop([i for i in df_features.columns if 'remove' in i], axis=1, inplace=True)

    return df_features






    