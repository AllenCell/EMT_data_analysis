import numpy as np
import pyvista as pv
import trimesh
from pathlib import Path
from bioio import BioImage
from skimage.transform import resize
from tqdm import tqdm
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.measure import regionprops_table
import pandas as pd
import argparse
import quilt3 as q3
from typing import Optional

from EMT_data_analysis.tools import alignment, io, const


def main(
        data_id: Optional[str]=None,
        output: Optional[str]=None
    ):
    '''
        Generate three figures for the inside-outside classification of nuclei
        at 0, 16, and 32 hours.
        
        Parameters
        ----------
        data_id: str
            Data ID of the movie.
        output: str
            Path to the output directory where the figures will be saved.
    '''
    # ensure output directory exists

    if data_id is None:
        data_id = const.EXAMPLE_IO_ID

    if output is None:
        output = io.setup_base_directory_name("figures/Inside-Outside/mesh-figures")
    else:
        output = Path(output)
        output.mkdir(exist_ok=True, parents=True)
    
    # load data
    df_meta = io.load_imaging_and_segmentation_dataset()
    df_meta = df_meta[df_meta['Data ID'] == data_id]
    df = io.load_inside_outside_classification()
    df = df[df['Data ID'] == data_id]
    df = df[df['Z']<27]

    tmp_dir = Path("./emt_tmp/nuclei_localization/")
    tmp_dir.mkdir(exist_ok=True, parents=True)
    mesh_path = df_meta['CollagenIV Segmentation Mesh Folder'].values[0].replace('s3://allencell/', '')
    bucket = q3.Bucket("s3://allencell")
    bucket.fetch(
        mesh_path + '/', 
        str(tmp_dir) + '/'
    )

    # load meshes
    mesh_fn = tmp_dir / (mesh_path.split('/')[-1] + '.vtm')
    meshes = pv.read(mesh_fn)
    
    # base filename for output
    outname = Path(mesh_fn).stem.replace('_mesh', '_inside-outside_classification_figure')

    # lighting setup
    light1 = pv.Light(
        position=(0.8, 0, .8),
        focal_point=(0, 0, 0),
        color=[1.0, 1.0, 0.9843, 1.0],  # Color temp. 5400 K
        intensity=.6,
    )
    light2 = pv.Light(
        position=(0.2, 1, .8),
        focal_point=(0, 0, 0),
        color=[1.0, 0.83921, 0.6666, 1.0],  # Color temp. 2850 K
        intensity=.6,
    )
    light3 = pv.Light(
        position=(0.5, 0.5, 1.0),
        focal_point=(0, 0, 0),
        color=[1.0, 0.83921, 0.6666, 1.0],  # Color temp. 2850 K
        intensity=.2,
    )

    # generate figures
    pv.start_xvfb()
    pl = pv.Plotter(off_screen=True, notebook=False, window_size=(1088, 1088))
    pl.zoom_camera(0.8)
    for tp in tqdm([0, 32, 64]):
        pl.clear()
        
        # get the data for the timepoint
        time = tp * 0.5
        df_tp = df[df['Time hr'] == time]
        
        # add mesh to the scene
        pl.add_mesh(meshes[f'{tp}'], color='#66b2b2', opacity=.5, show_edges=False, smooth_shading=True, specular=0.5, specular_power=15)
        
        # add nuclei centroids to the scene
        for _, row in df_tp.iterrows():
            pl.add_mesh(
                create_nucleus_mesh(row), 
                color='yellow', 
                show_edges=False, 
                smooth_shading=True, 
                specular=0.5, 
                specular_power=15
            )
        for _, row in df_tp.query('not Inside').iterrows():
            pl.add_mesh(
                create_nucleus_mesh(row), 
                color='cyan', 
                show_edges=False, 
                smooth_shading=True, 
                specular=0.5, 
                specular_power=15
            )
                
        # add lights to the scene
        pl.add_light(light1)
        pl.add_light(light2)
        pl.add_light(light3)
        
        # set camera position
        pl.camera_position = [(2920.2761459216617, 623.4148868582856, 437.7566674877991),
            (923.5, 623.5, 121.5),
            (-0.15643369450004763, -0.00041208853610704473, 0.9876883766695366)]
        
        # add title and save figure
        pl.add_title(f'Timepoint: {tp}')
        pl.screenshot(output / f'{outname}_hour_{time:.2f}.png')
    pl.close()
    
def create_nucleus_mesh(df_nucleus: pd.DataFrame):
    # create the sphere mesh
    mesh = trimesh.creation.icosphere(3, radius=10)
    
    # move sphere mesh to the nucleus centroid
    mesh.apply_translation(
        (df_nucleus["X"], 
        df_nucleus["Y"], 
        df_nucleus["Z"] * 2.88 / 0.271)
    )
    return mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate figures for inside-outside classification of nuclei.')
    parser.add_argument(
        '--data_id', 
        type=str,
        help='Data ID of the movie.'
    )
    parser.add_argument(
        '--output', 
        type=str,
        help='Path to the output directory where the figures will be saved.'
    )
    
    args = parser.parse_args()
    
    main(args.data_id, args.output)