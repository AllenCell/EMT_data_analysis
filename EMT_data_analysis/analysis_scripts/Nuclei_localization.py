#####----------Importing Libraries----------#####

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import quilt3 as q3
from shutil import rmtree
import gc

# 3d meshing libraries
import pyvista as pv
import trimesh
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

from bioio import BioImage

from skimage.measure import regionprops

from joblib import Parallel, delayed

from EMT_data_analysis.tools import alignment, io

import asyncio
import logging
import argparse

#####----------Main Analysis Function----------#####

def nuclei_localization(
        df:pd.DataFrame,
        data_id:str,
        output_directory:str,
        align_mesh:bool=True,
    ):
    '''
        This is the main function to localize nuclei inside a 3D mesh.

        This version transforms the MESH to match nuclei coordinates,
        keeping nuclei positions fixed.

        Parameters
        ----------
        manifest_path: str
            Path to the csv manifest of the full dataset
        data_id: str
            Data ID from manifest for data to process
        output_directory: str
            Path to the output directory where the localized nuclei data will be saved.
        align_mesh: bool
            Flag to enable alignment of the mesh to match nuclear segmentation coordinates.
            Default is True.
    '''
    # ensure output directory exists
    out_dir = Path(output_directory)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Use data_id-specific tmp_dir to avoid conflicts in parallel processing
    tmp_dir = Path(f"./emt_tmp/nuclei_localization_{data_id}/")
    tmp_dir.mkdir(exist_ok=True, parents=True)

    # load segmentations from manifest URL
    seg_path = df['H2B Nuclear Segmentation URL'].values[0]
    print(f"Loading H2B segmentation: {seg_path}")
    segmentations = BioImage(seg_path)

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

    # Parse alignment matrix once outside the loop (optimization)
    if align_mesh:
        alignment_matrix = alignment.parse_rotation_matrix_from_string(df['Dual Camera Alignment Matrix Value'].values[0])
    else:
        alignment_matrix = None

    # localize nuclei for each timepoint
    num_timepoints = int(df['Image Size T'].values[0])
    nuclei = []
    for timepoint in tqdm(range(num_timepoints), desc=f"Movie {data_id}"):
        # check if mesh exists for this timepoint
        if f'{timepoint}' not in meshes.keys():
            print(f"Mesh for timepoint {timepoint} not found.")
            continue

        # localize nuclei
        nuclei_tp = localize_for_timepoint(
            mesh=meshes[f'{timepoint}'],
            seg=segmentations.get_image_data("ZYX", T=timepoint).squeeze(),
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
    rmtree(tmp_dir, ignore_errors=True)

    # Explicit cleanup to free memory (important for parallel processing)
    del meshes, segmentations, nuclei
    gc.collect()



#####----------Helper Functions----------#####

def transform_mesh_vertices(mesh_pv: pv.PolyData, alignment_matrix: np.ndarray) -> pv.PolyData:
    """
    Transform mesh vertices from Camera 1 (CollagenIV) to Camera 2 (H2B) coordinates.

    The Dual Camera Alignment Matrix transforms Camera 2 → Camera 1.
    To transform mesh (Camera 1) → nuclei space (Camera 2), we use INVERSE.

    Parameters
    ----------
    mesh_pv : pv.PolyData
        PyVista mesh with vertices in Camera 1 coordinates.
    alignment_matrix : np.ndarray
        3x3 alignment matrix from the manifest.

    Returns
    -------
    pv.PolyData
        Mesh with vertices transformed to Camera 2 coordinates.
    """
    # Get the affine transform and its inverse
    transform = alignment.get_alignment_matrix(alignment_matrix)
    # Use INVERSE to go from Camera 1 (mesh) → Camera 2 (nuclei)
    inv_transform = transform.inverse

    # Get the 2D affine matrix components
    # The transform is a skimage AffineTransform with a 3x3 matrix for 2D
    # Matrix format: [[a, b, tx], [c, d, ty], [0, 0, 1]]
    matrix = inv_transform.params

    # Transform each vertex's XY coordinates (Z stays the same)
    vertices = mesh_pv.points.copy()

    # Apply 2D affine transform to X, Y coordinates
    # new_x = a*x + b*y + tx
    # new_y = c*x + d*y + ty
    xy = vertices[:, :2]  # Shape: (N, 2)
    ones = np.ones((len(xy), 1))
    xy_homogeneous = np.hstack([xy, ones])  # Shape: (N, 3)

    # Apply transform
    xy_transformed = xy_homogeneous @ matrix.T  # Shape: (N, 3)

    # Update vertices with transformed XY
    vertices[:, 0] = xy_transformed[:, 0]
    vertices[:, 1] = xy_transformed[:, 1]
    # Z coordinate unchanged

    # Create new mesh with transformed vertices
    transformed_mesh = pv.PolyData(vertices, mesh_pv.faces)

    return transformed_mesh


def fill_holes_flat_cap(mesh_pv: pv.PolyData) -> tuple:
    """
    Fill holes in a mesh by creating a flat cap at the max boundary Z level.
    Creates a closed mesh directly without using pcu.make_mesh_watertight()
    to avoid double-wall artifacts.

    This is an MIT-licensed alternative to PyMeshFix's GPL-licensed repair.

    Parameters
    ----------
    mesh_pv : pv.PolyData
        PyVista mesh with holes to fill.

    Returns
    -------
    tuple
        (vertices, faces) of the closed mesh.
    """
    vert = mesh_pv.points.copy()
    faces = mesh_pv.faces.reshape(-1, 4)[:, 1:].copy()

    # Extract boundary edges (the hole outline)
    boundary = mesh_pv.extract_feature_edges(
        boundary_edges=True, feature_edges=False,
        manifold_edges=False, non_manifold_edges=False
    )
    boundary_points = boundary.points

    if len(boundary_points) == 0:
        # No holes found, return as-is
        return vert, faces

    # Use max Z of boundary as cap level (preserves full biological extent)
    cap_z = np.max(boundary_points[:, 2])

    # Map boundary points to vertex indices in original mesh (vectorized)
    # Use scipy's cdist for efficient pairwise distance computation
    dists = cdist(boundary_points, vert)
    min_indices = np.argmin(dists, axis=1)
    min_dists = dists[np.arange(len(boundary_points)), min_indices]
    valid_mask = min_dists < 0.1
    boundary_indices = np.unique(min_indices[valid_mask])

    # Create cap vertices (same XY as boundary, but at cap_z)
    cap_verts = vert[boundary_indices].copy()
    cap_verts[:, 2] = cap_z

    # Add cap vertices to mesh
    n_orig_verts = len(vert)
    new_vert = np.vstack([vert, cap_verts])
    cap_vert_indices = np.arange(n_orig_verts, n_orig_verts + len(cap_verts))
    boundary_to_cap = dict(zip(boundary_indices, cap_vert_indices))

    # Create side faces connecting boundary to cap (vectorized)
    boundary_edges = boundary.lines.reshape(-1, 3)[:, 1:]

    # Pre-compute vertex indices for all boundary points at once
    edge_points_flat = boundary_points[boundary_edges.flatten()]
    edge_dists = cdist(edge_points_flat, vert)
    edge_vert_indices = np.argmin(edge_dists, axis=1).reshape(-1, 2)

    side_faces = []
    for i, (v1, v2) in enumerate(edge_vert_indices):
        if v1 in boundary_to_cap and v2 in boundary_to_cap:
            c1, c2 = boundary_to_cap[v1], boundary_to_cap[v2]
            side_faces.append([v1, v2, c2])
            side_faces.append([v1, c2, c1])
    side_faces = np.array(side_faces) if side_faces else np.empty((0, 3), dtype=int)

    # Create cap faces using Delaunay triangulation
    cap_xy = cap_verts[:, :2]
    tri = Delaunay(cap_xy)
    cap_faces = cap_vert_indices[tri.simplices]

    # Combine all faces
    if len(side_faces) > 0:
        new_faces = np.vstack([faces, side_faces, cap_faces])
    else:
        new_faces = np.vstack([faces, cap_faces])

    return new_vert, new_faces


def localize_for_timepoint(
        mesh:pv.PolyData,
        seg:np.ndarray,
        alignment_matrix:np.ndarray = None
    ):
    '''
        This function localizes nuclei inside a 3D mesh for a given timepoint.

        This version transforms the MESH to match nuclei coordinates,
        keeping nuclei positions fixed.

        Parameters
        ----------
        mesh: pv.PolyData
            3D mesh for the timepoint (in Camera 1 coordinates).
        seg: np.ndarray
            Nuclei segmentation for the timepoint (in Camera 2 coordinates).
        alignment_matrix: np.ndarray or None
            Parsed alignment matrix for dual camera alignment.
            If None, no alignment is performed.
    '''

    # Transform mesh to match nuclei coordinates if alignment is enabled
    # Mesh is in Camera 1 space (CollagenIV), nuclei are in Camera 2 space (H2B)
    # We transform mesh → Camera 2 using INVERSE of the alignment matrix
    if alignment_matrix is not None:
        mesh = transform_mesh_vertices(mesh, alignment_matrix)

    # Extend boundary vertices to the top Z level to help close mesh holes
    mf_holes = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
    outline_verts = mf_holes.points
    top = np.percentile(outline_verts[:,2], 99)
    for i in range(outline_verts.shape[0]):
        vert = outline_verts[i]
        new_vert = np.array([vert[0], vert[1], max([vert[2], top])])

        v_idx = mesh.find_closest_point(vert)
        mesh.points[v_idx] = new_vert

    # transpose segmentation to XYZ coordinates and set z-scale for isotropic resolution
    seg = seg.transpose(2, 1, 0)
    scale = 2.88 / 0.271

    # Fill holes and create watertight mesh using custom flat cap approach
    vw, fw = fill_holes_flat_cap(mesh)
    mesh = trimesh.Trimesh(vertices=vw, faces=fw)

    # initialize ray caster (for checking if a point is inside the mesh)
    rayCaster = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    # Extract nuclei properties using regionprops
    props = regionprops(seg.astype(int))

    if len(props) == 0:
        return pd.DataFrame({
            "Label": [], "Inside": [], "X": [], "Y": [], "Z": []
        })

    # Collect all nuclei data at once (vectorized)
    labels = [prop.label for prop in props]
    x_coords = [int(prop.centroid[0]) for prop in props]
    y_coords = [int(prop.centroid[1]) for prop in props]
    z_coords = [int(prop.centroid[2]) for prop in props]

    # Build centroids array for batch contains_points check
    # Nuclei coordinates are NOT transformed - they stay in original Camera 2 space
    centroids = np.array([
        [prop.centroid[0], prop.centroid[1], prop.centroid[2] * scale]
        for prop in props
    ])

    # Batch check all centroids at once (much faster than one-by-one)
    try:
        inside_flags = rayCaster.contains_points(centroids)
    except Exception:
        # Fallback to all False if ray casting fails
        inside_flags = np.zeros(len(centroids), dtype=bool)

    # Build dataframe directly from arrays
    nucData = {
        "Label": labels,
        "Inside": inside_flags.tolist(),
        "X": x_coords,
        "Y": y_coords,
        "Z": z_coords
    }

    return pd.DataFrame(nucData)


#####----------Run Function Call----------#####

def _suppress_zarr_async_warnings():
    """Suppress noisy zarr v3 asyncio shutdown warnings in joblib workers."""
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(lambda loop, context: None)
    except RuntimeError:
        pass


def _process_single_movie(data_id, df_manifest, output_directory, align_mesh):
    """
    Helper function to process a single movie for parallel execution.
    Returns the data_id and success status for logging.
    """
    _suppress_zarr_async_warnings()
    try:
        df_id = df_manifest[df_manifest['Data ID'] == data_id]
        nuclei_localization(
            df=df_id,
            data_id=data_id,
            output_directory=output_directory,
            align_mesh=align_mesh
        )
        return data_id, True, None
    except Exception as e:
        return data_id, False, str(e)


def run_nuclei_localization(
        output_directory:str,
        align_mesh:bool=True,
        n_jobs:int=4,
        parallel:bool=True,
        load_from_aws:bool=True,
        local_imaging_csv:str=None,
    ):
    '''
        This is the main function to localize nuclei inside a 3D mesh.

        Parameters
        ----------
        output_directory: str
            Path to the output directory where the localized nuclei data will be saved.
        align_mesh: bool
            Flag to enable alignment of the mesh to match nuclear segmentation.
            Default is True.
        n_jobs: int
            Number of parallel jobs. Default is 4 to balance speed vs memory.
            Each worker loads ~500MB-1GB of data (segmentation + mesh).
            Reduce to 2 if encountering memory issues (SIGKILL).
        parallel: bool
            Whether to use parallel processing. Set to False for debugging.
            Default is True.
        load_from_aws: bool
            Whether to load the imaging and segmentation manifest from AWS.
            Default is True.
        local_imaging_csv: str or None
            Path to local imaging_and_segmentation_data.csv.
            Only used when load_from_aws is False.
    '''
    # Load manifest
    df_manifest = io.load_imaging_and_segmentation_dataset(
        load_from_aws=load_from_aws,
        local_path=local_imaging_csv
    )

    # Filter to movies used in nucleus-BM workflow
    df_cond = df_manifest[df_manifest['Used For'].str.contains("nucleus-BM workflow", na=False)]
    data_ids = pd.unique(df_cond['Data ID'])

    print(f"Processing {len(data_ids)} movies with CollagenIV segmentations.")
    print(f"Alignment mode: Transform MESH to match nuclei coordinates")

    if parallel:
        # Parallel processing using joblib
        print(f"Running in parallel mode with n_jobs={n_jobs}")
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_process_single_movie)(
                data_id, df_manifest, output_directory, align_mesh
            )
            for data_id in data_ids
        )

        # Report results
        successful = [r[0] for r in results if r[1]]
        failed = [(r[0], r[2]) for r in results if not r[1]]

        print(f"\nCompleted: {len(successful)}/{len(data_ids)} movies processed successfully.")
        if failed:
            print("Failed movies:")
            for data_id, error in failed:
                print(f"  - {data_id}: {error}")
    else:
        # Sequential processing (for debugging)
        print("Running in sequential mode")
        for data_id in tqdm(data_ids, desc="Movies"):
            df_id = df_manifest[df_manifest['Data ID'] == data_id]
            nuclei_localization(
                df=df_id,
                data_id=data_id,
                output_directory=output_directory,
                align_mesh=align_mesh
            )

#####----------Argument Parsing----------#####
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Localize nuclei inside 3D meshes (mesh transform version).')
    parser.add_argument('--n_jobs', type=int, default=4,
                        help='Number of parallel jobs (default: 4). Reduce if memory issues occur.')
    parser.add_argument('--sequential', action='store_true',
                        help='Run in sequential mode (for debugging)')
    parser.add_argument('--no-align', action='store_true',
                        help='Disable mesh alignment')
    parser.add_argument('--local', action='store_true',
                        help='Load imaging_and_segmentation_data from local file instead of AWS')
    parser.add_argument('--local-csv', type=str, default=None,
                        help='Path to local imaging_and_segmentation_data.csv (only used with --local)')
    args = parser.parse_args()

    output_dir = io.setup_base_directory_name("nuclei_localization")

    run_nuclei_localization(
        output_directory=output_dir,
        align_mesh=not args.no_align,
        n_jobs=args.n_jobs,
        parallel=not args.sequential,
        load_from_aws=not args.local,
        local_imaging_csv=args.local_csv
    )
