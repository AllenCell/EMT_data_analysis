######---------importing libraries--------#######
import numpy as np
import pandas as pd
import os

import warnings
warnings.filterwarnings("ignore")

from bioio import BioImage
from tqdm import tqdm

from EMT_data_analysis.analysis_scripts.Image_alignment import align_image, get_alignment_matrix

#######---extracting area and intensity values for every z-----####--TAKES THE MOST TIME
import platform
from pathlib import Path

import argparse

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


    for movie_id, df_movie in tqdm(df.groupby('Movie Unique ID')):
    #importing raw image
        print(f'Movie: {movie_id}')
        print('Getting raw data...')
        
        file_path=df_movie['Raw Converted File Download'].values[0]
        if platform.system()=='Windows':
            path_w=file_path.replace('/','\\')
            img=BioImage(repr(path_w)[1:-1])
        else:
            img=BioImage(file_path)
    
        print('Getting colony mask....')
        if platform.system()!='Windows':
            seg_path = df_movie['All Cells Mask File Download'].values[0]
            seg_path = Path(seg_path).as_posix()
        else:
            seg_path = df_movie['All Cells Mask File Download'].values[0]
        
        seg_img = BioImage(seg_path)
        
        print('Computing features....')
        df_cr=pd.DataFrame()
        l = df_movie['Image Size T'].values[0]
        if l>97:
            l=97
        for time in tqdm(np.arange(l)):
            img_tl = img.get_image_dask_data("ZYX", C=1, T=time)
            img_raw = img_tl.compute() 
                
            img_seg_tl = img_seg.get_image_dask_data("ZYX", T=time)
            img_seg = img_seg_tl.compute()
            
            if align:
                transform = get_alignment_matrix(df_movie['Camera Alignment Matrix'].values[0])
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

                area.append(ar)
                total_int.append(total)
                mean_int.append(intensity)

            df_prop=pd.DataFrame(zip(z,area,mean_int,total_int), columns=['Z plane','Area of all cells mask per Z (pixels)','Mean intensity per Z','Total intensity per Z'])
            z_proj=np.count_nonzero(img_seg, axis=0)
            m_z=np.ma.masked_equal(z_proj,0)
            z_max_proj = np.max(img_seg,axis=0)
            ar2=np.count_nonzero(z_max_proj) 
            df_prop['MIP_area']=ar2
            df_prop['z_median']=np.ma.median(m_z)
            df_prop['z_mean']=np.ma.mean(m_z)
            df_prop['Timepoint']=time
            df_cr=pd.concat([df_cr,df_prop])
            
        df_cr['Movie Unique ID']=movie_id
        df_cr['Gene']=df_movie.gene.values[0]
        df_cr['Experimental Condition']=df_movie['Experimental Condition'].values[0]
        df_cr.to_csv(Path(save_folder) / f'Features_bf_colony_mask_{movie_id}.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    
    args = parser.parse_args()
    compute_bf_colony_features(args.data_path, args.output_path)





    
