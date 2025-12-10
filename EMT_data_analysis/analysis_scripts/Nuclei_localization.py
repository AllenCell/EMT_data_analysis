#####----------Importing Libraries----------#####

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import quilt3 as q3
from shutil import rmtree

# 3d meshing libraries
import pyvista as pv
import trimesh
import point_cloud_utils as pcu

from bioio import BioImage

from skimage.measure import regionprops

from EMT_data_analysis.tools import alignment, io

import argparse

#####----------Main Analysis Function----------#####

def nuclei_localization(
        df:pd.DataFrame, 
        data_id:str,
        output_directory:str,
        align_segmentation:bool=True,
    ):
    '''
        This is the main function to localize nuclei inside a 3D mesh.
        
        Parameters
        ----------
        manifest_path: str
            Path to the csv manifest of the full dataset
        data_id: str
            Data ID from manifest for data to process
        output_directory: str
            Path to the output directory where the localized nuclei data will be saved.
        align_segmentation: bool
            Flag to enable alignment of the segmentation using the barcode of the movie.
            Default is True.
    '''
    # ensure output directory exists
    out_dir = Path(output_directory)
    out_dir.mkdir(exist_ok=True, parents=True)

    tmp_dir = Path("./emt_tmp/nuclei_localization/")
    tmp_dir.mkdir(exist_ok=True, parents=True)
    
    # load segmentations and meshes
    # First, check for local ZARR file in the reprocessed directory
    local_zarr_base = Path("/allen/aics/emt/all_cells_masks/ZARR_Conversion/August_24_H2B_reprocess_v2/main")
    local_zarr_path = local_zarr_base / f"{data_id}_H2B_nuclear_segmentation.ome.zarr"

    local_zarr_base_batch2v2 = Path("/allen/aics/emt/nuclear_segmentation/ZARR_Conversion/deliverable_2_v2")
    local_zarr_path_batch2v2 = local_zarr_base_batch2v2 / f"{data_id}_H2B_nuclear_segmentation.ome.zarr"

    if local_zarr_path.exists():
        seg_path = str(local_zarr_path)
        print(f"Using batch1 local ZARR: {seg_path}")
    elif local_zarr_path_batch2v2.exists():
        seg_path = str(local_zarr_path_batch2v2)
        print(f"Using batch2_v2 local ZARR: {seg_path}")
    elif df['Gene'].values[0] == 'HIST1H2BJ':
        seg_path = df['H2B Nuclear Segmentation URL'].values[0]
        print(f"Using H2B segmentation from quilt manifest: {seg_path}")
    else:
        raise ValueError(f"The move {data_id} does not have H2B segmentations")
        
    # import pdb; pdb.set_trace()
    segmentations = BioImage(seg_path)

    # Check for local mesh files first, then fall back to S3 bucket
    # Local mesh directory for resubmission data
    local_mesh_base = Path("//allen/aics/emt/basement_membrane_segmentation/Resubmission/compile")
    local_mesh_folder = local_mesh_base / f"{data_id}_collagenIV_segmentation_mesh"

    mesh_fn = None
    use_local_mesh = False

    # Check if local mesh folder exists with VTM file
    if local_mesh_folder.exists():
        local_vtm_files = list(local_mesh_folder.glob('*.vtm'))
        if local_vtm_files:
            mesh_fn = local_vtm_files[0]
            use_local_mesh = True
            print(f"Using local mesh: {mesh_fn}")

    if not use_local_mesh:
        # Download meshes into temporary directory from s3 bucket
        mesh_path = df['CollagenIV Segmentation Mesh Folder'].values[0].replace('s3://allencell/', '')
        bucket = q3.Bucket("s3://allencell")
        try:
            bucket.fetch(
                mesh_path + '/',
                str(tmp_dir) + '/'
            )
        except Exception as e:
            print(f"Failed to download mesh for {data_id}: {e}")
            rmtree(tmp_dir, ignore_errors=True)
            return

        # load meshes - handle both naming conventions:
        # 1. DataID-prefixed: {data_id}_collagenIV_segmentation_mesh.vtm
        # 2. Generic: collagenIV_segmentation_mesh.vtm
        vtm_files = list(tmp_dir.glob('*.vtm'))
        if not vtm_files:
            print(f"No VTM mesh file found for {data_id} in {tmp_dir}")
            rmtree(tmp_dir, ignore_errors=True)
            return
        mesh_fn = vtm_files[0]  # Use the first (and typically only) VTM file
        print(f"Using S3 mesh: {mesh_fn}")


    # load meshes
    meshes = pv.read(mesh_fn)
    
    # localize nuclei for each timepoint
    num_timepoints = int(df['Image Size T'].values[0])
    nuclei = []
    for timepoint in tqdm(range(num_timepoints), desc=f"Movie {data_id}"):
        # check if mesh exists for this timepoint
        if f'{timepoint}' not in meshes.keys():
            print(f"Mesh for timepoint {timepoint} not found.")
            continue
        
        if align_segmentation:
            alignment_matrix = alignment.parse_rotation_matrix_from_string(df['Dual Camera Alignment Matrix Value'].values[0])
        else:
            alignment_matrix = np.zeros((3,3))

        # localize nuclei
        nuclei_tp = localize_for_timepoint(
            mesh=meshes[f'{timepoint}'],
            seg=segmentations.get_image_data("ZYX", T=timepoint).squeeze(),
            align_segmentation=align_segmentation,
            alignment_matrix=alignment_matrix
        )
        
        nuclei_tp['Data ID'] = data_id
        nuclei_tp['Time hr'] = timepoint / 0.5
        nuclei.append(nuclei_tp)
        
    # save nuclei data
    nuclei = pd.concat(nuclei)
    cols = nuclei.columns.tolist()
    newcols = cols[-2:]
    newcols.extend(cols[:-2])
    nuclei = nuclei[newcols]

    out_fn = out_dir / (data_id + "_localized_nuclei.csv")
    nuclei.to_csv(out_fn, index=False)
    rmtree(tmp_dir)


    
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
        transform = alignment.get_alignment_matrix(alignment_matrix)
        transform = transform.inverse

    mf_holes = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
    outline_verts = mf_holes.points
    top = np.percentile(outline_verts[:,2], 99)
    for i in range(outline_verts.shape[0]):
        vert = outline_verts[i]
        mesh.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
        new_vert = np.array([vert[0], vert[1], max([vert[2], top])])
        
        v_idx = mesh.find_closest_point(vert)
        mesh.points[v_idx] = new_vert

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

    vw, fw = pcu.make_mesh_watertight(vert, faces, 10000)

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

        try:
            contains = rayCaster.contains_points([centroid])
        except:
            continue
        
        # check if centroid is inside the mesh
        if contains[0]:
            nucData['Inside'].append(True)
        else:
            nucData['Inside'].append(False)
    
    return pd.DataFrame(nucData)


#####----------Run Function Call----------#####

def run_nuclei_localization(
        df_manifest:pd.DataFrame,
        output_directory:str,
        align_segmentation:bool=True,
    ):
    '''
        This is the main function to localize nuclei inside a 3D mesh.
        
        Parameters
        ----------
        manifest_path: str
            Path to the csv manifest of the full dataset
        data_id: str
            Data ID from manifest for data to process
        output_directory: str
            Path to the output directory where the localized nuclei data will be saved.
        align_segmentation: bool
            Flag to enable alignment of the segmentation using the barcode of the movie.
            Default is True.
    '''
    # Filter to specific Data IDs for analysis
    ANALYSIS_DATA_IDS = [
        '3500005548_43', '3500005548_46', '3500005548_48',
        '3500005824_35', '3500005824_36', '3500005824_37', '3500005824_38',
        '3500005828_43', '3500005828_45', '3500005828_46', '3500005828_67', '3500005828_70',
        '3500006256_19', '3500006256_21',
        '3500007081_8',
        '3500007213_38',
        '3500007247_5',
        '3500007432_52', '3500007432_57', '3500007432_61', '3500007432_63',
    ]

    df_cond = df_manifest[df_manifest['Data ID'].isin(ANALYSIS_DATA_IDS)]

    print(f"Processing {len(df_cond)} movies with CollagenIV segmentations.")

    for data_id in tqdm(pd.unique(df_cond['Data ID']), desc="Movies"):
        df_id = df_manifest[df_manifest['Data ID'] == data_id]

        # make sure the movie has the required segmentations
        nuclei_localization(
            df=df_id,
            data_id=data_id,
            output_directory=output_directory,
            align_segmentation=align_segmentation
        )

#####----------Argument Parsing----------#####
if __name__ == '__main__':
    manifest = io.load_imaging_and_segmentation_dataset()
    output_dir = io.setup_base_directory_name("nuclei_localization")

    run_nuclei_localization(
        df_manifest=manifest,
        output_directory=output_dir
    )