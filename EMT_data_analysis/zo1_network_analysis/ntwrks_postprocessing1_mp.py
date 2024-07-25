## Import packages:
## For working with filepaths and directories
from pathlib import Path
## For parallel number-crunching
from multiprocessing import Pool, Manager
import psutil
## for numerical work
import numpy as np
## For working with DataFrames
import pandas as pd
pd.options.mode.copy_on_write = True
## For working with images
from aicsimageio import AICSImage
## for working with strings
import re
## For creating a progress bar
from tqdm import tqdm


## slice_XYZT: output cropped images and corresponding network table with cropped timepoints and networks filtered out
## postprocessing 1: create table suitable for next steps

## postprocessing 2: output filtered images and major component images
##  requires data table with fluroescences and such


def stringified_intlist_to_intlist(ls):
    """Converts a list that is saved as a string back to a list object."""
    # ls = ntwrk_df_cln['node_label'].iloc[0]

    strints = ls if isinstance(ls, list) else ls.strip('[]')
    ls_cln = ls if isinstance(strints, list) else [int(x) for x in strints.split(',') if strints]

    return ls_cln


def extract_T(fp_as_string):
    t = re.search('T=[0-9]+', fp_as_string)
    if t:
        t = int(t.group(0).split('=')[-1]) 
    else:
        t = 0 
        print("""No 'T=[0-9]+' found in filename. Assuming both label and intensity 
              images contain only 1 timepoint and setting t = 0.""")
        
    return t


def crop_networks(groupby_object):

    nm, df = groupby_object

    barcode, position_well, timeframe = nm
    print(f'Working on barcode {barcode}: {position_well} -- T={timeframe}...')

    # get image paths
    skels_fpaths = list(skels_dir.glob('*/*/*'))
    edges_fpaths = list(edges_dir.glob('*/*/*'))
    nodes_fpaths = list(nodes_dir.glob('*/*/*'))

    ## find the filepath in skels_dir that corresponds to this dataframe
    skels_bc_pw_tf_fn_fp = [(*fp.parts[-3:-1], extract_T(fp.parts[-1]), fp.parts[-1], fp) for fp in skels_fpaths]
    edges_bc_pw_tf_fn_fp = [(*fp.parts[-3:-1], extract_T(fp.parts[-1]), fp.parts[-1], fp) for fp in edges_fpaths]
    nodes_bc_pw_tf_fn_fp = [(*fp.parts[-3:-1], extract_T(fp.parts[-1]), fp.parts[-1], fp) for fp in nodes_fpaths]

    skels_info = [x for x in skels_bc_pw_tf_fn_fp if nm == x[:3]]
    skels_bc, skels_pw, skels_tf, skels_fn, skels_fp = skels_info[0] if len(skels_info)==1 else [None,]*5

    edges_info = [x for x in edges_bc_pw_tf_fn_fp if nm == x[:3]]
    edges_bc, edges_pw, edges_tf, edges_fn, edges_fp = edges_info[0] if len(edges_info)==1 else [None,]*5

    nodes_info = [x for x in nodes_bc_pw_tf_fn_fp if nm == x[:3]]
    nodes_bc, nodes_pw, nodes_tf, nodes_fn, nodes_fp = nodes_info[0] if len(nodes_info)==1 else [None,]*5

    ## Get the corresponding barcode annotation entry
    annot = barcode_annotations_cln.query('`Plate Barcode` == @barcode').query('position_well == @position_well')

    ## Get the x and y slice points and the T stopping point
    x_crop = slice(*np.ravel(annot[['x_slice_start', 'x_slice_stop']].values))
    y_crop = slice(*np.ravel(annot[['y_slice_start', 'y_slice_stop']].values))
    t_crop = annot['time at cell mixing (timeframe)']
    t_crop = t_crop.iloc[0] if np.isfinite(t_crop.values) else np.inf


    dim_order = "ZYX"

    ## If the timeframe that we are looking at is still acceptable...
    if timeframe < t_crop:
        ## ...crop images for skeletons, edges and nodes:
        if isinstance(skels_fp, Path):
            ## open image
            img = AICSImage(skels_fp)
            img_arr = img.get_image_data(dim_order)

            ntwrk_labs_in_crop = np.unique(img_arr[:, y_crop, x_crop])

            skel_crop = img_arr * np.isin(img_arr, ntwrk_labs_in_crop)

            ## Save images that show only the major component
            out_subdir = Path.joinpath(out_dir_skels_crop, f'{barcode}/{position_well}')
            Path.mkdir(out_subdir, parents=True, exist_ok=True)
            AICSImage(skel_crop).save(Path.joinpath(out_subdir, Path(skels_fn).stem + '_crop.tif'))
        else:
            ntwrk_labs_in_crop = np.array([np.nan,])
            print(f'skels_fp is not a Path for {nm}')

        if isinstance(edges_fp, Path):
            ## open image
            img = AICSImage(edges_fp)
            img_arr = img.get_image_data(dim_order)

            edge_labs_in_crop = np.unique(img_arr[:, y_crop, x_crop])

            edges_crop = img_arr * np.isin(img_arr, edge_labs_in_crop)

            ## Save images that show only the major component
            out_subdir = Path.joinpath(out_dir_edges_crop, f'{barcode}/{position_well}')
            Path.mkdir(out_subdir, parents=True, exist_ok=True)
            AICSImage(edges_crop).save(Path.joinpath(out_subdir, Path(edges_fn).stem + '_crop.tif'))
        else:
            edge_labs_in_crop = np.array([np.nan,])
            print(f'edges_fp is not a Path for {nm}')

        if isinstance(nodes_fp, Path):
            ## open image
            img = AICSImage(nodes_fp)
            img_arr = img.get_image_data(dim_order)

            node_labs_in_crop = np.unique(img_arr[:, y_crop, x_crop])

            nodes_crop = img_arr * np.isin(img_arr, node_labs_in_crop)

            ## Save images that show only the major component
            out_subdir = Path.joinpath(out_dir_nodes_crop, f'{barcode}/{position_well}')
            Path.mkdir(out_subdir, parents=True, exist_ok=True)
            AICSImage(nodes_crop).save(Path.joinpath(out_subdir, Path(nodes_fn).stem + '_crop.tif'))
        else:
            node_labs_in_crop = np.array([np.nan,])
            print(f'nodes_fp is not a Path for {nm}')
    else:
        ntwrk_labs_in_crop = np.array([])
        edge_labs_in_crop = np.array([])
        node_labs_in_crop = np.array([])

    ## Append these labels to a list
    t_crop_list.append((nm[:2], t_crop))
    ntwrk_labs_in_crop_list.append((nm, ntwrk_labs_in_crop.tolist()))
    edge_labs_in_crop_list.append((nm, edge_labs_in_crop.tolist()))
    node_labs_in_crop_list.append((nm, node_labs_in_crop.tolist()))

    return



## code for other parts of the manuscript label the conditions differently,
## so map the nomenclature used in the manual annotations to the new
## nomenclature for consistency
## see: https://github.com/aics-int/Analysis_deliverable/blob/main/Analysis_scripts/orientation_analysis.py
SEP_to_Nivi_expt_cond_map = {'2D MG EMT 1:60MG':'2D-MG-EMT-1-60-MG',
                             '2D PLF EMT 1:60MG':'2D-PLF-EMT-1-60-MG',
                             '3D MG EMT 1:60MG':'3D-MG-EMT-1-60-MG',
                             '3D MG EMT no MG':'3D-MG-EMT-no-MG'}

## Get some local filename and directory locations:
sct_fpath = Path(__file__)
cwd = sct_fpath.parent
sct_fstem = sct_fpath.stem
barcode_annotations = pd.read_csv(Path.joinpath(cwd, 'annotations/zo1_to_seg_done.csv'))#, keep_default_na=False)
ntwrk_dir = Path.joinpath(cwd, 'concat_tables_out')
edges_dir = Path.joinpath(cwd, 'array_to_graph_mp_out/imgs_edges')
nodes_dir = Path.joinpath(cwd, 'array_to_graph_mp_out/imgs_nodes')
skels_dir = Path.joinpath(cwd, 'array_to_graph_mp_out/imgs_skel')

## Create a folder for image output if it doesn't already exist:
out_dir = Path.joinpath(cwd, sct_fstem + '_out')
out_dir_ntwrk_table = Path.joinpath(out_dir, Path('network_tables'))
out_dir_edges_crop = Path.joinpath(out_dir, 'imgs/edges_crop')
out_dir_nodes_crop = Path.joinpath(out_dir, 'imgs/nodes_crop')
out_dir_skels_crop = Path.joinpath(out_dir, 'imgs/skels_crop')

out_paths = [out_dir, out_dir_ntwrk_table,
             out_dir_edges_crop, out_dir_nodes_crop, out_dir_skels_crop]

for out in out_paths:
    Path.mkdir(out, parents=True, exist_ok=True)


## read the dataset in
ntwrk_df = pd.read_csv(Path.joinpath(ntwrk_dir, 'ntwrk_dataset.tsv'), sep='\t', low_memory=False)

## convert the barcodes columns to strings
ntwrk_df['barcode'] = ntwrk_df.barcode.astype(str)

## remove the barcode that was an initial test (if present)
ntwrk_df.query("barcode != 'initial_test'", inplace=True)

## Need to ensure that slice_start and slice_stop columns have None Type objects and not NaNs
barcode_annotations.x_slice_start = barcode_annotations.x_slice_start.astype('object').transform(lambda x: int(x) if np.isfinite(x) else None, convert_dtype=False)
barcode_annotations.x_slice_stop = barcode_annotations.x_slice_stop.astype('object').transform(lambda x: int(x) if np.isfinite(x) else None, convert_dtype=False)
barcode_annotations.y_slice_start = barcode_annotations.y_slice_start.astype('object').transform(lambda x: int(x) if np.isfinite(x) else None, convert_dtype=False)
barcode_annotations.y_slice_stop = barcode_annotations.y_slice_stop.astype('object').transform(lambda x: int(x) if np.isfinite(x) else None, convert_dtype=False)

## Assign experimental condition labels from barcode_annotations to nodes_df and edges_df
## according to the plate barcode, position index, and well label
barcode_annotations['position_well'] = barcode_annotations.apply(lambda x: 'P'+'-'.join(x[['Position Index', 'Well Label']].astype(str)), axis=1)

poswell_exptcond_map = {x[0]:x[1] for x in barcode_annotations[['position_well', 'expt_condition']].values}

## create the experimental condition column
ntwrk_df['expt_condition'] = ntwrk_df['position_well'].apply(lambda x: poswell_exptcond_map[x])

## convert my experimental condition labels to be consistent with Nivi's
ntwrk_df['expt_condition'] = ntwrk_df['expt_condition'].apply(lambda x: SEP_to_Nivi_expt_cond_map[x] if x in SEP_to_Nivi_expt_cond_map else x)

## use the 'timeframe' column to generate the 'Time (hours)' column
## (our timesteps are 0.5 hours per timeframe)
ntwrk_df['Time (hours)'] = ntwrk_df['timeframe'].apply(lambda x: x*0.5)

## Replace any NaNs in columns that are supposed to have list objects with string representations of empty lists
ntwrk_df.fillna({'node_label':'[]'}, inplace=True)
ntwrk_df.fillna({'edge_label':'[]'}, inplace=True)
ntwrk_df.fillna({'edges_num_pixels':'[]'}, inplace=True)
ntwrk_df.fillna({'ntwrk_fluor':'[]'}, inplace=True)

## Restore string representations of lists to list objects
ntwrk_df['node_label'] = ntwrk_df.node_label.transform(stringified_intlist_to_intlist)
ntwrk_df['edge_label'] = ntwrk_df.edge_label.transform(stringified_intlist_to_intlist)
ntwrk_df['edges_num_pixels'] = ntwrk_df.edges_num_pixels.transform(stringified_intlist_to_intlist)
ntwrk_df['ntwrk_fluor'] = ntwrk_df.ntwrk_fluor.transform(stringified_intlist_to_intlist)

## Filter out videos with a quantitative_analysis_tractability less than 4
barcode_annotations_cln = barcode_annotations.query('`quantitative_analysis_tractability (5 = high, 1 = low, 0 = unusable)` >= 4')
barcode_annotations_cln['Plate Barcode'] = barcode_annotations_cln['Plate Barcode'].astype(str)

## Keep only these highly tractable files in ntwrk_df
filename_stems = barcode_annotations_cln.file_name.apply(lambda x: x.split('.')[0])
ntwrk_df = ntwrk_df[ntwrk_df.filename.apply(lambda x: any([fnstem in x for fnstem in filename_stems]))]


## Splits up ntwrk_df into each timelapses individual timeframe and
## process those timeframes in parallel
cols = ['barcode', 'position_well', 'timeframe']
grps = ntwrk_df.groupby(cols)


## Pass analysis function to multiprocessing
if __name__ == '__main__':

    max_cpu_count = psutil.cpu_count()
    n_proc = int(input(f'How many processors do you want to use? (max cpu count is {max_cpu_count})\n> ')) or max_cpu_count
    print(f'Using {n_proc} processors...')

    manager = Manager()

    t_crop_list = manager.list()
    ntwrk_labs_in_crop_list = manager.list()
    edge_labs_in_crop_list = manager.list()
    node_labs_in_crop_list = manager.list()

    print('Starting multiprocessing...')
    with Pool(processes=n_proc) as pool:
        list(tqdm(pool.imap(crop_networks, grps, chunksize=10), total=len(grps)))
        pool.close()
        pool.join()

    print('Done multiprocessing.')

    ## Turn manager lists into regular lists
    t_crop_list = list(t_crop_list)
    ntwrk_labs_in_crop_list = list(ntwrk_labs_in_crop_list)
    edge_labs_in_crop_list = list(edge_labs_in_crop_list)
    node_labs_in_crop_list = list(node_labs_in_crop_list)


    ## Clean up the data tables
    t_crop_dict = dict(t_crop_list)
    ntwrk_labs_in_crop_dict = dict(ntwrk_labs_in_crop_list)
    edge_labs_in_crop_dict = dict(edge_labs_in_crop_list)
    node_labs_in_crop_dict = dict(node_labs_in_crop_list)

    ntwrk_df['t_crop'] = ntwrk_df[['barcode', 'position_well']].apply(lambda x: t_crop_dict[*x], axis=1)
    ntwrk_df = ntwrk_df.query('timeframe < t_crop')

    ntwrk_df['ntwrk_labs_in_crop'] = ntwrk_df[['barcode', 'position_well', 'timeframe']].apply(lambda x: ntwrk_labs_in_crop_dict[*x], axis=1)
    ntwrk_df['edges_labs_in_crop'] = ntwrk_df[['barcode', 'position_well', 'timeframe']].apply(lambda x: edge_labs_in_crop_dict[*x], axis=1)
    ntwrk_df['nodes_labs_in_crop'] = ntwrk_df[['barcode', 'position_well', 'timeframe']].apply(lambda x: node_labs_in_crop_dict[*x], axis=1)

    ntwrk_df = ntwrk_df[ntwrk_df.apply(lambda x: x.network_label in x.ntwrk_labs_in_crop, axis=1)]


    ## Save the cleaned up tables into too large and too small tables:
    ## the .json files include the edge and network fluorescence intensity columns but these
    ## have been removed from the .tsv version of the data because when opening the files
    ## with Excel there is a hard cap on the number of characters that fit in a single cell
    ## and these columns occasionally have a cell that exceeds that limit and it looks like
    ## something is wrong with the data output when it is opened in Excel (even though the
    ## .tsv is actually correctly saved and can be correctly opened as a DataFrame in Pandas)
    ntwrk_df.to_json(Path.joinpath(out_dir_ntwrk_table, f'ntwrk_dataset_crop.json'), index=False)
    ntwrk_df.drop(columns=['edges_fluor', 'ntwrk_fluor'], inplace=True)
    ntwrk_df.to_csv(Path.joinpath(out_dir_ntwrk_table, f'ntwrk_dataset_crop.tsv'), sep='\t', index=False)

    print('Done.')
