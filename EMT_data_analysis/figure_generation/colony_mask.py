import numpy as np
import pyvista as pv
import trimesh
from pathlib import Path
from bioio import BioImage
from skimage.transform import resize
from tqdm import tqdm
from skimage.transform import resize
from skimage.filters import gaussian
import pandas as pd
from CGAL import CGAL_Alpha_wrap_3
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from CGAL.CGAL_Kernel import Point_3
from skimage.morphology import remove_small_objects
import argparse
from typing import List


def main(
        dataset_manifest_path: str,
        colony_feature_manifest_path: str,
        movie_id: str,
        out_dir: str,
    ):
    '''
        This function creates a visualization of the colony mask in 3D for 0, 16, 32, and 48 hours.
        
        Parameters
        ----------
        dataset_manifest_path: str
            Path to the csv manifest containing summary data of the entire dataset
        colony_feature_manifest_path: str
            Path to the csv manifest containing results from brightfield colony mask feature extraction.
        movie_id: str
            Movie Unique ID of the movie.
        out_dir: str
            Path to the output directory where the visualization will be saved.
    '''
    
    # get bottom z layer
    df_feature = pd.read_csv(colony_feature_manifest_path)
    zbottom = df_feature.loc[df['Movie Unique ID'] == movie_id, 'z_bottom'].values[0]
    
    # get segmentation and base filename
    df_manifest = pd.read_csv(dataset_manifest_path)
    seg_fn = df_manifest.loc[df_manifest['Movie Unique ID'] == movie_id, 'All Cells Mask File Download'].values[0]
    seg = BioIo(seg_fn)
    outname = Path(seg_fn).stem + '_figure'
    
    # lighting setup
    light1 = pv.Light(
        position=(0, 0.8, .8),
        color=[1.0, 1.0, 0.9843, 1.0],  # Color temp. 5400 K
        intensity=.6,
    )
    light2 = pv.Light(
        position=(1, 0.2, .8),
        color=[1.0, 0.83921, 0.6666, 1.0],  # Color temp. 2850 K
        intensity=.6,
    )
    light3 = pv.Light(
        position=(0.5, 0.5, 1.0),
        color=[1.0, 0.83921, 0.6666, 1.0],  # Color temp. 2850 K
        intensity=.2,
    )

    # process frames for 0, 16, 32, and 48 hours
    pl = pv.Plotter(off_screen=True, notebook=False, window_size=(1088, 1088))
    for tp in tqdm([0, 32, 64, 96]):
        # clear scene
        pl.clear()
        
        # read timepoint and remove small objects
        seg = seg_file.get_image_data("ZYX", T=tp)
        seg = seg.transpose(2,1,0)
        seg = remove_small_objects(seg>0, min_size=5000)

        #cut the bottom z layers
        seg = seg[:,:,zbottom:]

        # create bottom and top segments
        seg_bottom = seg.copy()
        seg_bottom[:,:,2:] = 0
        seg_top = seg.copy()
        seg_top[:,:,:2] = 0

        # resize to isotropic voxels
        seg_bottom = resize(seg_bottom, (seg_bottom.shape[0], seg_bottom.shape[1], int(seg_bottom.shape[2] * 2.88 / 0.271)), order=0, preserve_range=True)
        seg_top = resize(seg_top, (seg_top.shape[0], seg_top.shape[1], int(seg_top.shape[2] * 2.88 / 0.271)), order=0, preserve_range=True)

        # create marching cubes meshes
        mesh_bottom = trimesh.voxel.ops.matrix_to_marching_cubes(seg_bottom, pitch=1)
        nv_bot, nf_top = alpha_wrap(mesh_bottom.vertices, alpha=10)
        mesh_bottom = trimesh.Trimesh(vertices=nv_bot, faces=nf_top)

        mesh_top = trimesh.voxel.ops.matrix_to_marching_cubes(seg_top, pitch=1)
        nv_top, nf_top = alpha_wrap(mesh_top.vertices, alpha=10)
        mesh_top = trimesh.Trimesh(vertices=nv_top, faces=nf_top)
        
        # add meshes to scene
        pl.add_mesh(mesh_bottom, color='magenta', show_edges=False, smooth_shading=True, specular=0.5, specular_power=15)
        pl.add_mesh(mesh_top, color='cyan', opacity=0.5, show_edges=False, smooth_shading=True, specular=0.5, specular_power=15)
        
        # add scale axis
        pl.add_mesh(make_axis([mesh_bottom, mesh_top]), smooth_shading=True)
        
        # set camera position and add lights
        pl.camera_position = [(899.801291895324, -4286.661837821917, 2158.211255367028),
            (913.9024224698779, 613.9024224698779, 111.7505),
            (0.0012619617942305635, 0.38534332696220364, 0.9227724138801128)]
        pl.add_light(light1)
        pl.add_light(light2)
        pl.add_light(light3)
        pl.add_light(
            pv.Light(light_type='HeadLight', intensity=0.5)
        )
        
        # output scene as image
        pl.add_title(f'Timepoint: {tp}')
        pl.show(auto_close=False)
        pl.screenshot(out_dir / f'{outname}_tp_{tp:03d}.png')
    pl.close()
    
def make_axis(
        ref_meshes, 
        length: float=50/0.271, 
        radius: int=10, 
        offset: List[int]=[-100, -100, 100]
    ):
    '''
        This function creates an scale reference axis for the scene.
        
        Parameters
        ----------
        ref_meshes: list
            List of reference meshes to use for lower bound of the axis.
        length: float
            Length of the axis.
        radius: int
            Radius of the axis.
        offset: list[int]
            Offset of the axis from lower bound of meshes.
    '''
    # get the minimum point of the bounding box of the meshes
    point = np.min(
        np.vstack([m.bounds for m in ref_meshes]),
        axis=0
    ) + offset
    
    # return axis object
    return trimesh.creation.axis(
        origin_size = radius,
        axis_length = length, 
        axis_radius = radius,
        transform = trimesh.transformations.translation_matrix(point)
        )
    
def alpha_wrap(points: np.ndarray, alpha: float = 20.0, offset=0.001):
    r"""Runs the Alphawrapping algorithm from CGAL on an input set of points
    provided as a numpy array

    See https://www.cgal.org/2022/05/18/alpha_wrap/ for algorithm details. In
    essence offset is the size that the algorithm considers points to have
    finite mass and alpha is the size of the ball it will do delaunay-style
    triangulation.

    Parameters
    ----------
    points : (N,3) numpy array
        surface points to shrinkwrap into a mesh
    alpha : float
        alpha controls the minimum carving size, and thus the size of straits 
        and holes that cannot be traversed during carving.
    offset : float
        offset is the value of the distance field level-set defining the offset 
        surface. It controls the distance of the mesh vertices to the input, 
        and thus the tightness of the approximation

    Returns
    -------
    nv : (N,3) numpy array
        numpy array of triangle mesh vertices
    nf : (M,3) numpy array
        numpy array of triangle mesh face indices

    """

    def cgal_vertices_faces_triangle_mesh(Q: Polyhedron_3):
        r"""Convert CGAL polyhedron object from the alphawrap library back into 
        numpy vertices and faces to be accessible.

        Parameters
        ----------
        Q : CGAL Polyhedron_3 oject
            surface mesh

        Returns
        -------
        vertices : (N,3) numpy array/allen/aics/assay-dev/users/Filip/forOthers/forAlexandra/EMT-membrane-mesh
            numpy array of mesh vertices
        faces : (M,3) numpy array
            numpy array of triangle face indices

        """
        vertices = np.zeros((Q.size_of_vertices(), 3), dtype=float)
        vertices_packed = {}
        faces = np.zeros((Q.size_of_facets(), 3), dtype=float)
        next_idx_v = 0
        for idx_f, facet in enumerate(Q.facets()):
            he = facet.halfedge()
            for j in range(3):
                p = he.vertex().point()
                v = tuple((p.x(), p.y(), p.z()))
                idx_v = vertices_packed.get(v, -1)
                if idx_v < 0:
                    vertices[next_idx_v, :] = v
                    vertices_packed[v] = next_idx_v
                    idx_v = next_idx_v
                    next_idx_v += 1
                faces[idx_f, j] = idx_v
                he = he.next()
        return vertices, faces

    vertices_cgal = [Point_3(x, y, z) for x, y, z in points.astype(np.double)]
    Q = Polyhedron_3()
    CGAL_Alpha_wrap_3.alpha_wrap_3(vertices_cgal, alpha, offset, Q)
    nv, nf = cgal_vertices_faces_triangle_mesh(Q)
    return nv, nf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate figures for colony mask segmentation.')

    parser.add_argument(
        '--manifest_path',
        type=str,
        required=True,
        help='Path to the csv manifest containing summary data of the entire dataset.'
    )
    parser.add_argument(
        '--feature_path',
        type=str,
        required=True,
        help='Path to the csv manifest containing results from brightfield colony mask feature extraction.'
    )
    parser.add_argument(
        '--movie_id',
        type=str,
        required=True,
        help='Movie Unique ID of the movie.'
    )
    parser.add_argument(
        '--output_directory',
        type=str,
        required=True,
        help='Path to the output directory where the visualization will be saved.'
    )

    args = parser.parse_args()
    main(args.manifest_path, args.feature_path, args.movie_id, args.output_directory)