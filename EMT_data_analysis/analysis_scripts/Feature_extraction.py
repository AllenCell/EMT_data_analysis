######---------importing libraries--------#######
import numpy as np
import pandas as pd
import os

import warnings
warnings.filterwarnings("ignore")

from bioio import BioImage
from tqdm import tqdm

from EMT_data_analysis.analysis_scripts.Image_alignment import align_image, get_alignment_matrix, camera_correction

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


    for scene, df_scene in tqdm(df.groupby('Scene Identifier')):
    #importing raw image
        barcode = df_scene['Barcode']

        print(f'Bacode-{barcode}')
        print(f'Scene-{scene}')
        print('Getting raw data...')
                
        file_path=df_scene['Raw path']
        if platform.system()=='Windows':
            path_w=file_path.replace('/','\\')
            img=BioImage(repr(path_w)[1:-1])
        else:
            img=BioImage(file_path)
        img.set_scene(scene)
    
        print('Getting colony mask....')
        if platform.system()!='Windows':
            folder = df_scene.colony_mask_path.values[0]
            folder = Path(folder).as_posix()
        else:
            folder = df_scene.colony_mask_path.values[0]
        print(folder)
        df_seg = pd.DataFrame([df_scene['Timepoint (h)'], df_scene['ACM path']], columns=['Timepoint','Mask_path'])
        
        print('Computing features....')
        df_cr=pd.DataFrame()
        img_tl=img.get_image_dask_data("ZYX", C=1,)
        img_raw = img_tl.compute() 
        
        seg_path=df_seg['Mask_path'].values[0]    
        img_seg=BioImage(seg_path).data.squeeze()
        
        if df_scene['channel 3'] == 'N-cadherin':
            img_seg = camera_correction(img_seg)
            img_raw = camera_correction(img_raw)

        if align:
            transform = get_alignment_matrix(
                barcode=df_scene['Instrument'], 
                alignment_folder='/allen/aics/assay-dev/users/Filip/Projects/camera_alignment/EMT-reprocess/camera-alignment/alignment_info'
            )
            transform = transform.inverse
        
        s_z=int(img_seg.shape[0])
        z,area,mean_int, total_int, var_int=[],[],[],[],[]
        for i in np.arange(s_z):
            z.append(i)
            seg_z = img_seg[i]
            if align:
                seg_z = align_image(seg_z, transform)
            
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
        df_prop['Timepoint']=df_scene['Timepoint (h)']
        df_cr=pd.concat([df_cr,df_prop])

        df_cr['scene']=scene
        df_cr['gene']=df_scene.gene.values[0]
        df_cr['Condition']=df_scene.fms_condition.values[0]
        df_cr.to_csv(Path(save_folder) / f'Features_bf_colony_mask_{barcode}_{scene}.csv')








    
