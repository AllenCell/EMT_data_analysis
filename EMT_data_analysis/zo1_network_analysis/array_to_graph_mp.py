## Import packages:
## For working with filepaths and directories
from pathlib import Path, WindowsPath
## Regular expressions for searching filenames for time and channel number
import re
## For parallel number-crunching
from multiprocessing import Pool
import psutil
## For numerical work
import numpy as np
## For working with DataFrames
import pandas as pd
## For working with images
from aicsimageio import AICSImage
from skimage import filters
from skimage import morphology
from skimage import measure



def save_img_no_rw(img, out_dir, fname=None):

    fname = fname if fname else 'temp'

    fig_count = 0
    out_fpath = Path.joinpath(out_dir, Path(fname+f'_{fig_count}.tif'))
    while out_fpath.exists():
        fig_count += 1
        out_fpath = Path.joinpath(out_dir, Path(fname+f'_{fig_count}.tif'))

    AICSImage(img).save(out_fpath)


def arr2graph(arr):
    """Will take an array and return the nodes and edges \
    as well as their connections. """
    # arr = zo1_lab.astype(bool)

    ## Make sure that the array is either 2D or 3D
    try:
        assert(arr.ndim == 2 or arr.ndim == 3)
    except AssertionError:
        print('Input array must be 2D or 3D.')

    ## Fill any tiny holes
    arr_filled = morphology.binary_closing(arr, footprint=morphology.cube(3))
    skel = morphology.skeletonize(arr_filled).astype(bool)
    ## skeletonize above will make your array int8 dtype, and
    ## will make True == 255, but I want it to be 1, so I will
    ## force it to be bool, hence the .astype above.

    ## Converting the bool to int now does not make 
    ## True -> 255, instead True -> 1 (which is what I want):
    ## Transform the skeletonized array into one where each
    ## pixel has a value equal to the number of non-zero 
    ## immediate neighbors plus itself
    ## the * skel is to re-skeletonize the rank sum
    if arr.ndim == 2:
        conn = filters.rank.sum(skel.astype(np.uint8), 
                                footprint=morphology.square(3),
                                mask=skel) * skel
    elif arr.ndim == 3:
        conn = filters.rank.pop(skel.astype(np.uint8),
                                footprint=morphology.cube(3),
                                mask=skel) * skel
    # This produces an array with the following values
    # (which is why I insisted on having the skeletonized array
    # have only 0s and 1s as values):
    # conn == 1,2 -> node (isolated point)
    # conn == 2 -> node (end point)
    # conn == 3 -> edge
    # conn >= 4 -> node (branch point)

    ## Label those endpoints, edges, and branchpoints (this is
    ## to get the connections between edges and nodes later on):
    edges_arr = (conn == 3)
    nodes_arr = ((conn == 1) + (conn == 2) + (conn >= 4))

    ## There can be both isolated nodes (a single pixel in space)
    ## and isolated edges (a closed loop in space)
    ## how do you uniquely define such a graph?
    ## Both edges and nodes need their own labels.
    nodes_lab = morphology.label(nodes_arr, connectivity=3)
    edges_lab = morphology.label(edges_arr, connectivity=3)

    return nodes_lab, edges_lab, skel


def get_neighboring_labels(home_img, labeled_neighbors_img, bad_neighbors=None):
    """home_img will be made binary (can be an image where only a particular label was
    chosen by home_img == lab)
    bad_neighbors argument lets you choose labels in labeled_neighbors_img to exclude
    from result (e.g. 0 is often background, so may want to exclude 0)"""
    neighbors = [*np.unique(morphology.binary_dilation(home_img, footprint=morphology.cube(3)) * labeled_neighbors_img)]
    if bad_neighbors:
        neighbors = [n for n in neighbors if n not in np.unique(bad_neighbors)]
    return tuple(neighbors)


def expand_bbox(bbox, ndim):
    big_bbox = (tuple((np.array(bbox[0:ndim]) - 0.5).astype(int)), tuple(np.array(bbox[ndim:2*ndim]) + 1))
    return big_bbox


def get_windows(img_lab): #labeled_img
    img_lab_props = measure.regionprops(img_lab)
    ndim = img_lab.ndim

    ## Create a list of labels and their associated bounding box:
    lab_labs, lab_bbox = zip(*[(lab.label, lab.bbox) for lab in img_lab_props])

    ## Apparently Python now allows your upper slice range to exceed bounds, and instead
    ## will just return the values within range.
    ## Grab a bbox that is 1 pixel wider on each edge of each axis:
    lab_bbox_big = [expand_bbox(bbox, ndim) for bbox in lab_bbox]   

    ## Create slicing windows of these expanded bboxes:
    windows = [[slice(*i) for i in list(zip(*bb))] for bb in lab_bbox_big]

    ## zip the labels and windows together
    lab_windows = zip(lab_labs, windows)

    return lab_windows


def main(fp_lab, fp_img=None):

    barcode = fp_lab.parts[-3]
    position_well = fp_lab.parts[-2]
    fname = fp_lab.parts[-1].split('.', 1)[0]

    ## Make output directories for valuable images of nodes, edges, labeled networks/skeletons, and the network tables
    nodes_outdir = Path.joinpath(out_dir, Path('imgs_nodes'), Path(barcode), Path(position_well))
    Path.mkdir(nodes_outdir, parents=True, exist_ok=True)

    edges_outdir = Path.joinpath(out_dir, Path('imgs_edges'), Path(barcode), Path(position_well))
    Path.mkdir(edges_outdir, parents=True, exist_ok=True)

    skel_outdir = Path.joinpath(out_dir, Path('imgs_skel'), Path(barcode), Path(position_well))
    Path.mkdir(skel_outdir, parents=True, exist_ok=True)


    network_out_subdir = Path.joinpath(out_dir, Path('network_tables'), Path(barcode), Path(position_well))
    Path.mkdir(network_out_subdir, parents=True, exist_ok=True)

    ## Extract the timepoint and channel of the segmentation from the filename:
    t = re.search('T=[0-9]+', fp_lab.parts[-1])
    if t:
        t = int(t.group(0).split('=')[-1]) 
    else:
        t = 0 
        print("""No 'T=[0-9]+' found in filename. Assuming both label and intensity 
              images contain only 1 timepoint and setting t = 0.""")

    ## Let me know where you are in the timelapse...
    print(f'T={t}... Seg_File={fp_lab.parts[-1]}')

    ## Open the predicted segmentation file and the original image file:
    lab = AICSImage(fp_lab)

    ## Load the stacks of interest into memory:
    dim_order = "ZYX"
    zo1_lab = lab.get_image_data(dim_order)
    zo1_lab = zo1_lab.astype(int)

    nodes_lab, edges_lab, skel = arr2graph(zo1_lab)
    nodes_arr, edges_arr = nodes_lab.astype(bool), edges_lab.astype(bool)

    nodes_lab_windows = get_windows(nodes_lab)
    edges_lab_windows = get_windows(edges_lab)


    ## Find connected networks in the skeleton by labeling skel
    skel_lab = morphology.label(skel, connectivity=3)

    ## Find which nodes neighbor which edges, and which edges neighbor which nodes:
    ## NOTE
    ## we are ensuring that only one node shows up when querying for neighbors in a
    ## window by setting nodes_lab == l (i.e. only show pixels that equal the label
    ## associated with the window)
    node_neighbors_edgelabs = [(l, get_neighboring_labels(nodes_lab[*w]==l, edges_lab[*w], bad_neighbors=[0])) for l,w in nodes_lab_windows]
    edge_neighbors_nodelabs = [(l, get_neighboring_labels(edges_lab[*w]==l, nodes_lab[*w], bad_neighbors=[0])) for l,w in edges_lab_windows]


    ## Use the combination of node_neighbors_edgelabs and edge_neighbors_nodelabs to
    ## find which nodes neighbor each other:
    nodes_lab_props = measure.regionprops(nodes_lab, intensity_image=skel_lab)

    node_neighbors_nodelabs = []
    for x in node_neighbors_edgelabs:
        ## Get which edges are connected to a particular node:
        node, edges = x
        ## Iterate through the edge_neighbors and look for connected nodes
        ## in the edge_neighbors_nodelabs list:
        node_neighbors_nodelabs.append((node, [n for e,n in edge_neighbors_nodelabs if e in edges]))
    ## Clean up the node list with node neighbors so that  there are no repeating node labels
    node_neighbors_nodelabs = [(node, tuple(np.unique([n for ns in n_neighbors for n in ns]))) for node, n_neighbors in node_neighbors_nodelabs]
    ## and also remove the "home node" from the node neighbors list to get the final cleaned up list:
    node_neighbors_nodelabs = [(node, tuple([n for n in n_neighbors if n != node])) for node, n_neighbors in node_neighbors_nodelabs]

    ## Collect information for the node-based network table:
    node_label, neighboring_nodes = zip(*node_neighbors_nodelabs)
    node_label_e, neighboring_edges = zip(*node_neighbors_edgelabs)

    node_locs = []
    node_centroids_subpx = []
    node_centroids = []
    node_ntwrk_lab = []
    for x in nodes_lab_props:
        loc_norm = [np.linalg.norm(loc) for loc in x.coords.tolist()]
        cntrd_norm = np.linalg.norm(x.centroid)
        centermost_loc = x.coords.tolist()[np.argmin(np.abs(loc_norm - cntrd_norm))]
        ntwrk_lab = int(*np.unique(x.image_intensity[x.image])) if len(np.unique(x.image_intensity[x.image]))==1 else np.nan

        node_locs.append(x.coords.tolist())
        node_centroids_subpx.append(x.centroid)
        node_centroids.append(centermost_loc)
        node_ntwrk_lab.append(ntwrk_lab)

    ## Also collect some information for the edge-based network table
    edges_lab_props = measure.regionprops(edges_lab, intensity_image=skel_lab)
    edge_label, edge_nodes = zip(*edge_neighbors_nodelabs)
    edge_num_px = [x.num_pixels for x in edges_lab_props]
    edge_ntwrk_lab = [int(*np.unique(x.image_intensity[x.image])) if len(np.unique(x.image_intensity[x.image]))==1 else np.nan for x in edges_lab_props]

    if fp_img:
        ## create an output folder for the intensity images
        intens_outdir = Path.joinpath(out_dir, Path('imgs_intens'), Path(barcode), Path(position_well))
        Path.mkdir(intens_outdir, parents=True, exist_ok=True)
        
        ## Add some fluorescence metrics if an image was provided:
        ## ZO-1 is channel 1 in the microscopy data
        CHAN = 1

        ## Open the raw image at the associated timepoint and channel
        img = AICSImage(fp_img)
        print(f'Trying to load into dask: T={t}...')
        ## this part usually takes a couple of minutes per timepoint
        img_dask = img.get_image_dask_data(dim_order, T=t, C=CHAN)
        print(f'Loaded into dask: T={t}, trying to compute: T={t}...')
        img_arr = img_dask.compute()
        print(f'Computed T={t}.')

        nodes_lab_props_fluor = measure.regionprops(nodes_lab, intensity_image=img_arr)
        nodes_fluor = {x.label:x.image_intensity[x.image] for x in nodes_lab_props_fluor}
        nodes_fluor_median = {x:np.median(nodes_fluor[x]) for x in nodes_fluor}
        nodes_fluor_mean = {x:np.mean(nodes_fluor[x]) for x in nodes_fluor}
        nodes_fluor_max = {x:np.max(nodes_fluor[x]) for x in nodes_fluor}
        nodes_fluor_min = {x:np.min(nodes_fluor[x]) for x in nodes_fluor}
        nodes_fluor_std = {x:np.std(nodes_fluor[x]) for x in nodes_fluor}

        edges_lab_props_fluor = measure.regionprops(edges_lab, intensity_image=img_arr)
        edges_fluor = {x.label:x.image_intensity[x.image] for x in edges_lab_props_fluor}
        edges_fluor_median = {x:np.median(edges_fluor[x]) for x in edges_fluor}
        edges_fluor_mean = {x:np.mean(edges_fluor[x]) for x in edges_fluor}
        edges_fluor_max = {x:np.max(edges_fluor[x]) for x in edges_fluor}
        edges_fluor_min = {x:np.min(edges_fluor[x]) for x in edges_fluor}
        edges_fluor_std = {x:np.std(edges_fluor[x]) for x in edges_fluor}

        skel_lab_props_fluor = measure.regionprops(skel_lab, intensity_image=img_arr)
        skel_fluor = {x.label:x.image_intensity[x.image] for x in skel_lab_props_fluor}
        skel_fluor_median = {x:np.median(skel_fluor[x]) for x in skel_fluor}
        skel_fluor_mean = {x:np.mean(skel_fluor[x]) for x in skel_fluor}
        skel_fluor_max = {x:np.max(skel_fluor[x]) for x in skel_fluor}
        skel_fluor_min = {x:np.min(skel_fluor[x]) for x in skel_fluor}
        skel_fluor_std = {x:np.std(skel_fluor[x]) for x in skel_fluor}

        img_min = np.min(img_arr)
        img_max = np.max(img_arr)
        img_median = np.median(img_arr)
        img_mean = np.mean(img_arr)
        img_std = np.std(img_arr)
        otsu_thresh = filters.threshold_otsu(img_arr)
        skel_median = np.median(img_arr[skel])
        skel_mean = np.mean(img_arr[skel])
        skel_std = np.std(img_arr[skel])

        fname_out = Path.joinpath(intens_outdir, Path(fname+'_intens.tif'))
        if not fname_out.exists():
            AICSImage(img_dask).save(fname_out)
        else:
            pass

    else:
        CHAN = np.nan

        nodes_lab_props_fluor = measure.regionprops(nodes_lab)
        nodes_fluor = {x.label:np.array([np.nan]) for x in nodes_lab_props_fluor}
        nodes_fluor_median = {x:np.median(nodes_fluor[x]) for x in nodes_fluor}
        nodes_fluor_mean = {x:np.mean(nodes_fluor[x]) for x in nodes_fluor}
        nodes_fluor_max = {x:np.max(nodes_fluor[x]) for x in nodes_fluor}
        nodes_fluor_min = {x:np.min(nodes_fluor[x]) for x in nodes_fluor}
        nodes_fluor_std = {x:np.std(nodes_fluor[x]) for x in nodes_fluor}

        edges_lab_props_fluor = measure.regionprops(edges_lab)
        edges_fluor = {x.label:np.array([np.nan]) for x in edges_lab_props_fluor}
        edges_fluor_median = {x:np.median(edges_fluor[x]) for x in edges_fluor}
        edges_fluor_mean = {x:np.mean(edges_fluor[x]) for x in edges_fluor}
        edges_fluor_max = {x:np.max(edges_fluor[x]) for x in edges_fluor}
        edges_fluor_min = {x:np.min(edges_fluor[x]) for x in edges_fluor}
        edges_fluor_std = {x:np.std(edges_fluor[x]) for x in edges_fluor}

        skel_lab_props_fluor = measure.regionprops(skel_lab)
        skel_fluor = {x.label:np.array([np.nan]) for x in skel_lab_props_fluor}
        skel_fluor_median = {x:np.median(skel_fluor[x]) for x in skel_fluor}
        skel_fluor_mean = {x:np.mean(skel_fluor[x]) for x in skel_fluor}
        skel_fluor_max = {x:np.max(skel_fluor[x]) for x in skel_fluor}
        skel_fluor_min = {x:np.min(skel_fluor[x]) for x in skel_fluor}
        skel_fluor_std = {x:np.std(skel_fluor[x]) for x in skel_fluor}

        img_min = np.nan
        img_max = np.nan
        img_median = np.nan
        img_mean = np.nan
        img_std = np.nan
        otsu_thresh = np.nan
        skel_median = np.nan
        skel_mean = np.nan
        skel_std = np.nan


    ## Output some tables describing the network:
    ## Table for classically describing network (each line is a node):
    nodes_df = pd.DataFrame(data={'node_label':node_label,
                                  'network_label':node_ntwrk_lab, 
                                  'node_centroids_subpx':node_centroids_subpx, 
                                  'node_centroids':node_centroids, 
                                  f'node_locs_{dim_order}':node_locs, 
                                  'neighboring_nodes':neighboring_nodes,
                                  'neighboring_edges':neighboring_edges,
                                  'filename':fname,
                                  'barcode':barcode,
                                  'position_well':position_well,
                                  'timeframe':t,
                                  'img_intensity_channel':CHAN,
                                  'filepath_img_labeled':fp_lab,
                                  'filepath_img_intensity':fp_img,
                                  'img_min':img_min,
                                  'img_max':img_max,
                                  'img_median':img_median,
                                  'img_mean':img_mean,
                                  'img_std':img_std,
                                  'otsu_thresh':otsu_thresh,
                                  'total_ntwrk_fluor_median':skel_median,
                                  'total_ntwrk_fluor_mean':skel_mean,
                                  'total_ntwrk_fluor_std':skel_std
                                  })
    nodes_df['nodes_fluor'] = nodes_df.node_label.transform(lambda x: nodes_fluor[x].tolist())
    nodes_df['nodes_fluor_median'] = nodes_df.node_label.transform(lambda x: nodes_fluor_median[x])
    nodes_df['nodes_fluor_mean'] = nodes_df.node_label.transform(lambda x: nodes_fluor_mean[x])
    nodes_df['nodes_fluor_max'] = nodes_df.node_label.transform(lambda x: nodes_fluor_max[x])
    nodes_df['nodes_fluor_min'] = nodes_df.node_label.transform(lambda x: nodes_fluor_min[x])
    nodes_df['nodes_fluor_std'] = nodes_df.node_label.transform(lambda x: nodes_fluor_std[x])

    nodes_df.to_csv(Path.joinpath(network_out_subdir, Path(fname+'_nodes.tsv')),
                    sep='\t', index=False)

    ## Table describing the network as a combination of edges (each line is an edge);
    ## P.S. (There is both a table to describe nodes and edges because both isolated
    ## nodes and isolated edges can exist):
    edges_df = pd.DataFrame(data={'edge_label':edge_label,
                                  'network_label':edge_ntwrk_lab, 
                                  'neighboring_nodes':edge_nodes,
                                  'num_pixels':edge_num_px,
                                  'filename':fname,
                                  'barcode':barcode,
                                  'position_well':position_well,
                                  'timeframe':t,
                                  'img_intensity_channel':CHAN,
                                  'filepath_img_labeled':fp_lab,
                                  'filepath_img_intensity':fp_img,
                                  'img_min':img_min,
                                  'img_max':img_max,
                                  'img_median':img_median,
                                  'img_mean':img_mean,
                                  'img_std':img_std,
                                  'otsu_thresh':otsu_thresh,
                                  'total_ntwrk_fluor_median':skel_median,
                                  'total_ntwrk_fluor_mean':skel_mean,
                                  'total_ntwrk_fluor_std':skel_std
                                  })
    edges_df['edges_fluor'] = edges_df.edge_label.transform(lambda x: edges_fluor[x].tolist())
    edges_df['edges_fluor_median'] = edges_df.edge_label.transform(lambda x: edges_fluor_median[x])
    edges_df['edges_fluor_mean'] = edges_df.edge_label.transform(lambda x: edges_fluor_mean[x])
    edges_df['edges_fluor_max'] = edges_df.edge_label.transform(lambda x: edges_fluor_max[x])
    edges_df['edges_fluor_min'] = edges_df.edge_label.transform(lambda x: edges_fluor_min[x])
    edges_df['edges_fluor_std'] = edges_df.edge_label.transform(lambda x: edges_fluor_std[x])
    
    edges_df.to_csv(Path.joinpath(network_out_subdir, Path(fname+'_edges.tsv')),
                    sep='\t', index=False)

    ## Make some groupby objects to merge on:
    lf_gb = nodes_df.groupby(['filename', 'barcode', 'position_well', 'timeframe', 'network_label',
                              'img_intensity_channel', 'filepath_img_labeled', 'filepath_img_intensity',
                              'img_min', 'img_max', 'img_median', 'img_mean', 'img_std', 'otsu_thresh',
                              'total_ntwrk_fluor_median', 'total_ntwrk_fluor_mean', 'total_ntwrk_fluor_std'
                              ])
    rt_gb = edges_df.groupby(['filename', 'barcode', 'position_well', 'timeframe', 'network_label',
                              'img_intensity_channel', 'filepath_img_labeled', 'filepath_img_intensity',
                              'img_min', 'img_max', 'img_median', 'img_mean', 'img_std', 'otsu_thresh',
                              'total_ntwrk_fluor_median', 'total_ntwrk_fluor_mean', 'total_ntwrk_fluor_std'
                              ])

    d_lf = {('node_label', 'nodes_fluor', 'nodes_fluor_median', 'nodes_fluor_mean',
             'nodes_fluor_max', 'nodes_fluor_min', 'nodes_fluor_std'): list}
    d_rt = {('edge_label', 'num_pixels', 'edges_fluor', 'edges_fluor_median', 'edges_fluor_mean',
             'edges_fluor_max', 'edges_fluor_min', 'edges_fluor_std'): list}
    lf = lf_gb.agg({c: fn for cols, fn in d_lf.items() for c in cols}).reset_index()
    rt = rt_gb.agg({c: fn for cols, fn in d_rt.items() for c in cols}).reset_index()

    ## Combine the nodes dataframe and edges dataframe into a single dataframe of network metrics 
    netmet_df = pd.merge(left=lf, right=rt, how='outer', suffixes=("_nodes","_edges"),
                         on=['filename', 'barcode', 'position_well', 'timeframe', 'network_label',
                             'img_intensity_channel', 'filepath_img_labeled', 'filepath_img_intensity',
                             'img_min', 'img_max', 'img_median', 'img_mean', 'img_std', 'otsu_thresh',
                             'total_ntwrk_fluor_median', 'total_ntwrk_fluor_mean', 'total_ntwrk_fluor_std'
                             ])

    ## when merging with the 'outer' method you introduce NaN values if a network has no edges or no nodes, so replace
    ## those with empty lists
    netmet_df.num_pixels = netmet_df.num_pixels.apply(lambda x: x if isinstance(x, list) else [])

    netmet_df['total_edge_len'] = netmet_df['num_pixels'].transform(lambda x: sum(x) if x else np.nan)
    netmet_df['median_edge_len'] = netmet_df['num_pixels'].transform(lambda x: np.median(x) if x else np.nan)
    netmet_df['mean_edge_len'] = netmet_df['num_pixels'].transform(lambda x: np.mean(x) if x else np.nan)
    netmet_df['max_edge_len'] = netmet_df['num_pixels'].transform(lambda x: max(x) if x else np.nan)
    netmet_df['min_edge_len'] = netmet_df['num_pixels'].transform(lambda x: min(x) if x else np.nan)
    netmet_df['std_edge_len'] = netmet_df['num_pixels'].transform(lambda x: np.std(x) if x else np.nan)

    netmet_df['ntwrk_fluor'] = netmet_df.network_label.transform(lambda x: skel_fluor[x].tolist())
    netmet_df['ntwrk_fluor_median'] = netmet_df.network_label.transform(lambda x: skel_fluor_median[x])
    netmet_df['ntwrk_fluor_mean'] = netmet_df.network_label.transform(lambda x: skel_fluor_mean[x])
    netmet_df['ntwrk_fluor_max'] = netmet_df.network_label.transform(lambda x: skel_fluor_max[x])
    netmet_df['ntwrk_fluor_min'] = netmet_df.network_label.transform(lambda x: skel_fluor_min[x])
    netmet_df['ntwrk_fluor_std'] = netmet_df.network_label.transform(lambda x: skel_fluor_std[x])

    netmet_df.rename(columns={'num_pixels':'edges_num_pixels'}, inplace=True)

    netmet_df.to_csv(Path.joinpath(network_out_subdir, Path(fname+'_ntwrk__metrics.tsv')),
                     sep='\t', index=False)

    AICSImage(nodes_lab).save(Path.joinpath(nodes_outdir, Path(fname+'_nodes.tif')))
    AICSImage(edges_lab).save(Path.joinpath(edges_outdir, Path(fname+'_edges.tif')))
    AICSImage(skel_lab).save(Path.joinpath(skel_outdir, Path(fname+'_skel.tif')))

    return



# Get some local filename and directory locations:
sct_fpath = Path(__file__)
cwd = sct_fpath.parent
sct_fstem = sct_fpath.stem

# Create a folder for image output if it doesn't already exist:
out_dir = Path.joinpath(cwd, sct_fstem + '_out')
Path.mkdir(out_dir, exist_ok=True)


# Get list of z-stacks of interest
lab_dir = Path.joinpath(cwd, 'label_segmentations_mp_out')

fp_lab_list = [p for p in lab_dir.glob('*/*/*')]

## couple the labeled image filepaths to their corresponding raw image filepaths found in
## the barcode_annotations file
barcode_annotations = pd.read_csv(Path.joinpath(cwd, 'annotations/zo1_to_seg_done.csv'))
fname_list = [fp_lab.parts[-1].split('.', 1)[0] for fp_lab in fp_lab_list]
annot = pd.concat([barcode_annotations[barcode_annotations.file_name.apply(lambda x: x.split('.')[0] in fname)] for fname in fname_list])


## Get the raw image filepath from annot
fp_img_list = annot.file_path.transform(lambda x: Path(x)).to_list()
## If this is a WindowsPath then 
fp_img_list = ['\\' + str(fp_img) if isinstance(fp_img, WindowsPath) else fp_img for fp_img in fp_img_list]

br_dir = Path.joinpath(cwd, 'annotations/br_medians')
fp_br_list = [p for p in br_dir.glob('**/*.tiff')]

fp_lab_img_list = list(zip(fp_lab_list, fp_img_list))


if __name__ == '__main__':

    max_cpu_count = psutil.cpu_count()
    n_proc = int(input(f'How many processors do you want to use? (max cpu count is {max_cpu_count})\n> ')) or max_cpu_count
    print(f'Using {n_proc} processors...')

    with Pool(processes=n_proc) as pool:
        pool.starmap(main, fp_lab_img_list)
        pool.close()
        pool.join()

    print('Done.')
