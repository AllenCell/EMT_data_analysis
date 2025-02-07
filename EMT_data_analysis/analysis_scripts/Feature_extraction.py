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
from fire import Fire

def main(manifest: str, outdir: str, align=True):
    # run compute features for these parameters is not running the full workflow
    df = pd.read_csv(manifest)
    save_folder = Path(outdir) / 'feature_extraction'
    save_folder.mkdir(parents=True, exist_ok=True)
    compute_bf_colony_features(df, save_folder, True)


def compute_bf_colony_features(df, save_folder, align=True, flourescence_channels=[2,3]):
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


    for _, df_scene in tqdm(df.iterrows(), total=len(df.index)):
    #importing raw image
        barcode = df_scene['Barcode']
        well = df_scene['Well']
        pos = df_scene['Position']
        scene = f'{pos}-{well}'

        outfn = Path(save_folder) / f'Features_bf_colony_mask_{barcode}_{scene}.csv'
        # if outfn.exists():
            # continue

        print(f'Bacode-{barcode}')
        print(f'Scene-{scene}')
        print('Getting raw data...')
                
        file_path=df_scene['Raw File Path']
        if platform.system()=='Windows':
            path_w=file_path.replace('/','\\')
            img=BioImage(repr(path_w)[1:-1])
        else:
            img=BioImage(file_path)
        img.set_scene(scene)
    
        print('Getting colony mask....')
        seg_path = df_scene['Seg File Path']
        if platform.system()!='Windows':
            seg_path = Path(seg_path).as_posix()
        print(seg_path)
        df_seg = pd.DataFrame([[df_scene['Timepoint (h)'], seg_path]], columns=['Timepoint','Mask_path'])
        
        df_channels = []
        for ch in flourescence_channels:
            df_cr=pd.DataFrame()
            img_tl=img.get_image_dask_data("ZYX", C=ch,)
            img_raw = img_tl.compute() 
            
            seg_path=df_seg['Mask_path'].values[0]    
            img_seg=BioImage(seg_path).data.squeeze()
            
            if df_scene['Channel 3'] == 'N-cadherin':
                print('Correcting Camera Issues')
                print('Start size:', img_raw.shape)
                img_seg = camera_correction(img_seg)
                img_raw = camera_correction(img_raw)

            if align:
                print('Performing Camera Alignment')
                transform = get_alignment_matrix(
                    barcode=df_scene['Scope'], 
                    alignment_folder='/allen/aics/assay-dev/users/Filip/Projects/camera_alignment/EMT-reprocess/camera-alignment/alignment_info'
                )
                transform = transform.inverse
            
            print('Raw:', img_raw.shape)
            print('Seg:', img_seg.shape)
            if img_seg.shape[-1] < img_seg.shape[0]:
                img_seg = img_seg.transpose([2,0,1])
                print('Seg trandspose:', img_seg.shape) 

            print('Computing features....')
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
            
            channel = f'channel_{int(ch+1)}'
            df_channels.append(pd.DataFrame(zip(z,area,mean_int,total_int,var_int), columns=['z',f'area_pixels',f'{channel}_mean_intensity',f'{channel}_total_intensity',f'{channel}_variance_intensity']))
        df_prop = pd.concat(df_channels, axis=1)
        df_prop = df_prop.loc[:,~df_prop.columns.duplicated()].copy()
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
        df_cr['gene_channel_3'] = [df_scene['Channel 3'] if df_scene['Channel 3'] is not None else 'Control',]*len(df_cr)
        df_cr['gene_channel_4'] = [df_scene['Channel 4'] if df_scene['Channel 4'] is not None else 'Control',]*len(df_cr)
        df_cr.to_csv(Path(save_folder) / f'Features_bf_colony_mask_{barcode}_{scene}.csv')



if __name__ == '__main__':
    Fire(main)




    
