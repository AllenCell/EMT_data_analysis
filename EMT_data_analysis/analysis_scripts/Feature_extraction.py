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








    
