#####----------Importing Libraries----------#####

import numpy as np
import pandas as pd
from pathlib import Path

# 3d meshing libraries
import pyvista as pv
import trimesh
import point_cloud_utils as pcu

from bioio import BioImage

from skimage.measure import regionprops

from .Image_alignment import align_image, get_alignment_matrix

#####----------Main Analysis Function----------#####

def nuclei_localization(
        manifest_path:str, 
        movie_id:str,
        output_directory:str,
        align_segmentation:bool=True,
    ):
    '''
        This is the main function to localize nuclei inside a 3D mesh.
        
        Parameters
        ----------
        manifest_path: str
            Path to the csv manifest of the full dataset
        movie_id: str
            Movie Unique ID from manifest for data to process
        output_directory: str
            Path to the output directory where the localized nuclei data will be saved.
        align_segmentation: bool
            Flag to enable alignment of the segmentation using the barcode of the movie.
            Default is True.
    '''
    # ensure output directory exists
    out_dir = Path(output_directory)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # load data
    df = pd.read(manifest_path)
    df = df[df['Movie Unique ID'] == movie_id]

    segmentations = BioImage(df['CollagenIV Segmentation Probability File Download'].values[0])
    meshes = pv.load(df['CollagenIV Segmentation Mesh Folder'].values[0])
    
    # localize nuclei for each timepoint
    num_timepoints = int(df['Image Size T'].values[0])
    nuclei = []
    for timepoint in range(num_timepoints):
        # check if mesh exists for this timepoint
        if f'{timepoint}' not in meshes.keys():
            print(f"Mesh for timepoint {timepoint} not found.")
            continue
        
        if align_segmentation:
            alignment_matrix = np.ndarray(eval(df['Camera Alignment Matrix'].values[0]))
        else:
            alignment_matrix = np.zeros((3,3))

        # localize nuclei
        print(f"Localizing nuclei for timepoint {timepoint}...")
        nuclei_tp = localize_for_timepoint(
            mesh=meshes[f'{timepoint}'],
            seg=segmentations.get_image_data("ZYX", T=timepoint).squeeze(),
            align_segmentation=align_segmentation,
            alignment_matrix=alignment_matrix
        )
        
        nuclei_tp['Movie Unique ID'] = movie_id
        nuclei_tp['Time hr'] = timepoint / 0.5
        nuclei.append(nuclei_tp)
        
    # save nuclei data
    nuclei = pd.concat(nuclei)
    cols = nuclei.columns
    nuclei = nuclei[cols[-2:] + cols[:-2]]

    out_fn = out_dir / (Path(segmentation_fn).stem.replace("_segmentation", "_localized_nuclei") + ".csv")
    nuclei.to_csv(out_fn, index=False)
    
#####----------Helper Functions----------#####
    
def localize_for_timepoint(
        mesh:pv.PolyData, 
        seg:np.ndarray, 
        align_segmentation:bool,
        alignment_matrix:np.ndarray
    ):
    '''
        This function localizes nuclei inside a 3D mesh for a given timepoint.
        
        Parameters
        ----------
        mesh: pv.PolyData
            3D mesh for the timepoint.
        seg: np.ndarray
            Nuclei segmentation for the timepoint.
        align_segmentation: bool
            Flag to enable alignment of the segmentation using the barcode of the movie.
        barcode: str
            Barcode of the movie.
    '''
    
    # align segmentation if required
    if align_segmentation:
        transform = get_alignment_matrix(barcode, alignment_folder)
        seg = align_image(seg, transform)

    # convert 2d surface mesh into an enclosed 3d mesh
    vert, faces = mesh.points, mesh.faces.reshape(mesh.n_faces, 4)[:,1:]
    vert_up = np.zeros_like(vert)
    np.copyto(vert_up, vert)
    vert_up[:, 2] = max(vert[:,2])
    face_up = np.zeros_like(faces)
    np.copyto(face_up, faces)

    mesh = trimesh.Trimesh(vertices=vert, faces=faces)
    roof = trimesh.Trimesh(vertices=vert_up, faces=face_up)
    mesh_conc = trimesh.util.concatenate(mesh, roof)

    vert, faces = mesh_conc.vertices, mesh_conc.faces

    vw, fw = pcu.make_mesh_watertight(vert, faces, 10_000)

    mesh = trimesh.Trimesh(vertices=vw, faces=fw)

    # transpose segmentation to XYZ coordinates and set z-scale for isotropic resolution
    seg = seg.transpose(2, 1, 0)
    scale = 2.88 / 0.271

    # initialize ray caster (for checking if a point is inside the mesh)
    rayCaster = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    # initialize nuclei data dictionary
    nucData = {}
    nucData["Label"] = []
    nucData["Inside"] = []
    nucData["X"] = []
    nucData["Y"] = []
    nucData["Z"] = []
    
    # localize nuclei
    props = regionprops(seg.astype(int))
    for prop in props:
        nucData['Label'].append(prop.label)
        nucData["X"].append(int(prop.centroid[0]))
        nucData["Y"].append(int(prop.centroid[1]))
        nucData["Z"].append(int(prop.centroid[2]))
        
        # get nuclei centroid (scales to isotropic resolution)
        centroid = [
            prop.centroid[0],
            prop.centroid[1],
            prop.centroid[2] * scale
        ]
        
        # check if centroid is inside the mesh
        if rayCaster.contains_points([centroid])[0]:
            nucData['Inside'].append(True)
        else:
            nucData['Inside'].append(False)
    
    return pd.DataFrame(nucData)