## Import packages:
## For working with filepaths and directories
from pathlib import Path
# For parallel number-crunching
from multiprocessing import Pool, Manager
import psutil
## numpy
import numpy as np
## For working with DataFrames
import pandas as pd
pd.options.mode.copy_on_write = True
## for working with labeled images
## For working with images
from aicsimageio import AICSImage
## for working with text strings
import re
## For creating a progress bar
from tqdm import tqdm


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


def label_filt_img(img_arr, lab_val_list, save=False, out_path=None):

    img_filt = np.zeros(img_arr.shape)

    for (lab, val) in lab_val_list:
        img_filt += (img_arr == lab) * val


    if save==True:
        AICSImage(img_filt).save(out_path)
        img_filt = None
    else:
        pass

    return img_filt


def filter_networks(groupby_object):

    nm, df = groupby_object

    barcode, position_well, timeframe = nm
    print(f' Working on barcode {barcode}: {position_well} -- T={timeframe}...')

    ## get image paths
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
    df_at_T = df.query('timeframe == @timeframe')

    ## num_nodes and num_edges should/are defined for ntwrk_df (ie. globally):
    # df_at_T['num_nodes'] = df_at_T.node_label.transform(lambda x: len(x))
    # df_at_T['num_edges'] = df_at_T.edge_label.transform(lambda x: len(x))
    

    ## The following are some tests for filtering out incorrect/undesirable networks:
    ## remove any networks that consist of 2 or fewer edges or 3 or fewer nodes
    ## and are also dim:
    ## THIS WAS USED FOR THE DATA USED IN THE FIG ON MAR 26 2024:
    df_at_T_filt = df_at_T.query('((num_nodes > 11) and (mean_edge_len >= ntwrk_edge_len_mean)) or (ntwrk_fluor_median >= (init_6hrs_total_ntwrk_fluor_median_median + 0*init_6hrs_total_ntwrk_fluor_median_std))')
    df_at_T_filt = df_at_T.query('((num_edges > 10) and (mean_edge_len >= ntwrk_edge_len_mean)) or (ntwrk_fluor_median >= (init_6hrs_total_ntwrk_fluor_median_median + 0*init_6hrs_total_ntwrk_fluor_median_std))')

    ## Get the label of the biggest network
    biggest_labels = np.unique([np.array(x).tolist() for x in df_at_T.ntwrk_edge_len_max_label.values])

    ## Get the labels that exceed the fluorescence threshold
    vals_big = list(zip(biggest_labels, [True]*len(biggest_labels)))
    vals_median = df_at_T[['network_label', 'ntwrk_fluor_median']].values.tolist()
    vals_max = df_at_T[['network_label', 'ntwrk_fluor_max']].values.tolist()
    vals_mean = df_at_T[['network_label', 'ntwrk_fluor_mean']].values.tolist()
    vals_filt = list(zip(df_at_T_filt['network_label'].values, [True]*len(df_at_T_filt['network_label'].values)))


    dim_order = "ZYX"

    ## ...crop images for skeletons, edges and nodes:
    if isinstance(skels_fp, Path):
        ## open image
        img = AICSImage(skels_fp)
        img_arr = img.get_image_data(dim_order)

        ## create output directories
        out_subdir = Path.joinpath(out_dir_skels_majcomp, f'{barcode}/{position_well}')
        out_subdir_med = Path.joinpath(out_dir_skels_med, f'{barcode}/{position_well}')
        out_subdir_max = Path.joinpath(out_dir_skels_max, f'{barcode}/{position_well}')
        out_subdir_mean = Path.joinpath(out_dir_skels_mean, f'{barcode}/{position_well}')
        out_subdir_filt = Path.joinpath(out_dir_skels_filt, f'{barcode}/{position_well}')

        [Path.mkdir(dir, parents=True, exist_ok=True) for dir in [out_subdir,
                                                                  out_subdir_med,
                                                                  out_subdir_max,
                                                                  out_subdir_mean,
                                                                  out_subdir_filt]]

        label_filt_img(img_arr, vals_big, save=True, out_path=Path.joinpath(out_subdir, Path(skels_fn).stem + '_majcomp.tif'))
        label_filt_img(img_arr, vals_median, save=True, out_path=Path.joinpath(out_subdir_med, Path(skels_fn).stem + '_ntwrk_median.tif'))
        label_filt_img(img_arr, vals_max, save=True, out_path=Path.joinpath(out_subdir_max, Path(skels_fn).stem + '_ntwrk_max.tif'))
        label_filt_img(img_arr, vals_mean, save=True, out_path=Path.joinpath(out_subdir_mean, Path(skels_fn).stem + '_ntwrk_mean.tif'))
        label_filt_img(img_arr, vals_filt, save=True, out_path=Path.joinpath(out_subdir_filt, Path(skels_fn).stem + '_ntwrk_filt.tif'))

    else:
        print(f'skels_fp is not a Path for {nm}')


    if isinstance(edges_fp, Path):
        ## open image
        img = AICSImage(edges_fp)
        img_arr = img.get_image_data(dim_order)

        ## create output directories
        out_subdir = Path.joinpath(out_dir_edges_majcomp, f'{barcode}/{position_well}')
        Path.mkdir(out_subdir, parents=True, exist_ok=True)

        ## Save images that show only the major component
        label_filt_img(img_arr, vals_big, save=True, out_path=Path.joinpath(out_subdir, Path(edges_fn).stem + '_majcomp.tif'))

    else:
        print(f'edges_fp is not a Path for {nm}')


    if isinstance(nodes_fp, Path):
        ## open image
        img = AICSImage(nodes_fp)
        img_arr = img.get_image_data(dim_order)

        ## create output directories
        out_subdir = Path.joinpath(out_dir_nodes_majcomp, f'{barcode}/{position_well}')
        Path.mkdir(out_subdir, parents=True, exist_ok=True)

        ## Save images that show only the major component
        label_filt_img(img_arr, vals_big, save=True, out_path=Path.joinpath(out_subdir, Path(nodes_fn).stem + '_majcomp.tif'))

    else:
        print(f'nodes_fp is not a Path for {nm}')

    ## Append these labels to a list
    biggest_labels_list.append((nm, biggest_labels.tolist()))

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

ntwrk_dir = Path.joinpath(cwd, 'ntwrks_postprocessing1_mp_out/network_tables')
edges_dir = Path.joinpath(cwd, 'ntwrks_postprocessing1_mp_out/imgs/edges_crop')
nodes_dir = Path.joinpath(cwd, 'ntwrks_postprocessing1_mp_out/imgs/nodes_crop')
skels_dir = Path.joinpath(cwd, 'ntwrks_postprocessing1_mp_out/imgs/skels_crop')

## Create a folder for image output if it doesn't already exist:
out_dir = Path.joinpath(cwd, sct_fstem + '_out')

out_dir_ntwrk_table = Path.joinpath(out_dir, Path('network_tables'))

out_dir_skels_filt = Path.joinpath(out_dir, Path('imgs/skels_filt'))

out_dir_edges_majcomp = Path.joinpath(out_dir, 'imgs/edges_majcomp')
out_dir_nodes_majcomp = Path.joinpath(out_dir, 'imgs/nodes_majcomp')
out_dir_skels_majcomp = Path.joinpath(out_dir, 'imgs/skels_majcomp')

out_dir_edges_med = Path.joinpath(out_dir, 'imgs/edges_ntwrk_median')
out_dir_nodes_med = Path.joinpath(out_dir, 'imgs/nodes_ntwrk_median')
out_dir_skels_med = Path.joinpath(out_dir, 'imgs/skels_ntwrk_median')

out_dir_edges_max = Path.joinpath(out_dir, 'imgs/edges_ntwrk_max')
out_dir_nodes_max = Path.joinpath(out_dir, 'imgs/nodes_ntwrk_max')
out_dir_skels_max = Path.joinpath(out_dir, 'imgs/skels_ntwrk_max')

out_dir_edges_mean = Path.joinpath(out_dir, 'imgs/edges_ntwrk_mean')
out_dir_nodes_mean = Path.joinpath(out_dir, 'imgs/nodes_ntwrk_mean')
out_dir_skels_mean = Path.joinpath(out_dir, 'imgs/skels_ntwrk_mean')


out_paths = [out_dir, out_dir_ntwrk_table, out_dir_skels_filt,
             out_dir_skels_majcomp, out_dir_edges_majcomp, out_dir_nodes_majcomp,
             out_dir_skels_med, out_dir_skels_max, out_dir_skels_mean,
             ]

for out in out_paths:
    Path.mkdir(out, parents=True, exist_ok=True)


## read the dataset in
## NOTE the .tsv file does not have all fluorescence intensity values in it,
## but the .json file does
ntwrk_df = pd.read_json(Path.joinpath(ntwrk_dir, 'ntwrk_dataset_crop.json'))

## convert the barcodes columns to strings
ntwrk_df['barcode'] = ntwrk_df.barcode.astype(str)

## remove the barcode that was an initial test (if present)
ntwrk_df.query("barcode != 'initial_test'", inplace=True)

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
## TODO remove this chunk below?
ntwrk_df.fillna({'node_label':'[]'}, inplace=True)
ntwrk_df.fillna({'edge_label':'[]'}, inplace=True)
ntwrk_df.fillna({'edges_num_pixels':'[]'}, inplace=True)
ntwrk_df.fillna({'ntwrk_fluor':'[]'}, inplace=True)

## Restore string representations of lists to list objects
ntwrk_df['node_label'] = ntwrk_df.node_label.transform(stringified_intlist_to_intlist)
ntwrk_df['edge_label'] = ntwrk_df.edge_label.transform(stringified_intlist_to_intlist)
ntwrk_df['edges_num_pixels'] = ntwrk_df.edges_num_pixels.transform(stringified_intlist_to_intlist)
ntwrk_df['ntwrk_fluor'] = ntwrk_df.ntwrk_fluor.transform(stringified_intlist_to_intlist)

## get the number of nodes and number of edges
ntwrk_df['num_nodes'] = ntwrk_df.node_label.transform(lambda x: len(x))
ntwrk_df['num_edges'] = ntwrk_df.edge_label.transform(lambda x: len(x))


## Find the network with the largest edge_len and the size of that edge_len for each timepoint as a way to
## find the major component of the graph which will then be saved as an image
cols = ['expt_condition', 'barcode', 'position_well', 'timeframe']
ntwrk_sizes_max = ntwrk_df.groupby(cols).apply(lambda df: df.total_edge_len.max())
d = ntwrk_sizes_max.to_dict()
ntwrk_df['ntwrk_edge_len_max'] = ntwrk_df[cols].apply(lambda df: d[*df], axis=1)

ntwrk_df['total_ntwrk_fluor_max'] = ntwrk_df.groupby(cols).ntwrk_fluor_max.transform(lambda s: s.max())

ntwrk_df['ntwrk_edge_len_mean'] = ntwrk_df.groupby(cols).mean_edge_len.transform(lambda s: s.mean())


## NOTE that the max ntwrk label number can change over time depending on how pieces break off
## (eg. if a piece breaks off in the upper left then networks further down could get a network
## label with a larger value, since labeling is done in a rasterized fashion).
## Also, there can be 2 major components if there is a tie at a particular timepoint...
## Therefore they are not very useful right now for grouping and plotting data.
## Despite this, we will need these labels to output the major component images:
ntwrk_sizes_max_label = ntwrk_df.groupby(cols).apply(lambda df: df[df.total_edge_len == df.total_edge_len.max()].network_label.values)
d = ntwrk_sizes_max_label.to_dict()
ntwrk_df['ntwrk_edge_len_max_label'] = ntwrk_df[cols].apply(lambda df: d[*df], axis=1)
ntwrk_df['ntwrk_edge_len_max_label_num'] = ntwrk_df['ntwrk_edge_len_max_label'].transform(lambda x: len(x))

ntwrk_df.query('ntwrk_edge_len_max_label_num > 1').ntwrk_edge_len_max_label

ntwrk_df['ntwrk_edge_len_max_label'] = ntwrk_df['ntwrk_edge_len_max_label'].transform(lambda x: int(x[-1]) if len(x) == 1 else x)


## get the mean and median fluorescence of the first 6 hours as an estimate of initial brightness
cols = ['expt_condition', 'barcode', 'position_well']

init_6hrs_total_ntwrk_fluor_mean = ntwrk_df.query('timeframe < 12').groupby(cols).apply(lambda df: df.total_ntwrk_fluor_mean.mean())
d = init_6hrs_total_ntwrk_fluor_mean.to_dict()
ntwrk_df['init_6hrs_total_ntwrk_fluor_mean_mean'] = ntwrk_df[cols].apply(lambda df: d[*df], axis=1)

init_6hrs_total_ntwrk_fluor_mean_std = ntwrk_df.query('timeframe < 12').groupby(cols).apply(lambda df: df.total_ntwrk_fluor_mean.std())
d = init_6hrs_total_ntwrk_fluor_mean_std.to_dict()
ntwrk_df['init_6hrs_total_ntwrk_fluor_mean_std'] = ntwrk_df[cols].apply(lambda df: d[*df], axis=1)

init_6hrs_total_ntwrk_fluor_mean_max = ntwrk_df.query('timeframe < 12').groupby(cols).apply(lambda df: df.total_ntwrk_fluor_mean.max())
d = init_6hrs_total_ntwrk_fluor_mean_max.to_dict()
ntwrk_df['init_6hrs_total_ntwrk_fluor_mean_max'] = ntwrk_df[cols].apply(lambda df: d[*df], axis=1)


init_6hrs_total_ntwrk_fluor_median = ntwrk_df.query('timeframe < 12').groupby(cols).apply(lambda df: df.total_ntwrk_fluor_median.median())
d = init_6hrs_total_ntwrk_fluor_median.to_dict()
ntwrk_df['init_6hrs_total_ntwrk_fluor_median_median'] = ntwrk_df[cols].apply(lambda df: d[*df], axis=1)

init_6hrs_total_ntwrk_fluor_median_std = ntwrk_df.query('timeframe < 12').groupby(cols).apply(lambda df: df.total_ntwrk_fluor_median.std())
d = init_6hrs_total_ntwrk_fluor_median_std.to_dict()
ntwrk_df['init_6hrs_total_ntwrk_fluor_median_std'] = ntwrk_df[cols].apply(lambda df: d[*df], axis=1)

init_6hrs_total_ntwrk_fluor_median_max = ntwrk_df.query('timeframe < 12').groupby(cols).apply(lambda df: df.total_ntwrk_fluor_median.max())
d = init_6hrs_total_ntwrk_fluor_median_max.to_dict()
ntwrk_df['init_6hrs_total_ntwrk_fluor_median_max'] = ntwrk_df[cols].apply(lambda df: d[*df], axis=1)


## group the data into timeframes per timelapse
cols = ['barcode', 'position_well', 'timeframe']

grps = ntwrk_df.groupby(cols)


## Pass analysis function to multiprocessing
if __name__ == '__main__':

    max_cpu_count = psutil.cpu_count()
    n_proc = int(input(f'How many processors do you want to use? (max cpu count is {max_cpu_count})\n> ')) or max_cpu_count
    print(f'Using {n_proc} processors...')


    manager = Manager()
    biggest_labels_list = manager.list()

    print('Starting multiprocessing...')
    with Pool(processes=n_proc, maxtasksperchild=10) as pool:
        list(tqdm(pool.imap(filter_networks, grps, chunksize=10), total=len(grps)))
        pool.close()
        pool.join()

    print('Done multiprocessing.')

    ## Turn manager lists into regular lists
    biggest_labels_list = list(biggest_labels_list)

    ## TODO should also clean up and save the edges and nodes tables
    ## Clean up the data tables
    biggest_labels_dict = dict(biggest_labels_list)

    ntwrk_df['biggest_labels'] = ntwrk_df[['barcode', 'position_well', 'timeframe']].apply(lambda x: biggest_labels_dict[*x], axis=1)

    ## THIS WAS USED FOR THE DATA USED IN THE FIG ON MAR 26 2024:
    ntwrk_df_filt = ntwrk_df.query('((num_nodes > 11) and (mean_edge_len >= ntwrk_edge_len_mean)) or (ntwrk_fluor_median >= (init_6hrs_total_ntwrk_fluor_median_median + 0*init_6hrs_total_ntwrk_fluor_median_std))')
    ntwrk_df_filt = ntwrk_df.query('((num_edges > 10) and (mean_edge_len >= ntwrk_edge_len_mean)) or (ntwrk_fluor_median >= (init_6hrs_total_ntwrk_fluor_median_median + 0*init_6hrs_total_ntwrk_fluor_median_std))')

    ntwrk_df_majcomp = ntwrk_df[ntwrk_df.apply(lambda x: x.network_label in x.biggest_labels, axis=1)]

    ## Save the cleaned up tables into too large and too small tables:
    ## the .json files include the edge and network fluorescence intensity columns but these
    ## have been removed from the .tsv version of the data because when opening the files
    ## with Excel there is a hard cap on the number of characters that fit in a single cell
    ## and these columns occasionally have a cell that exceeds that limit and it looks like
    ## something is wrong with the data output when it is opened in Excel (even though the
    ## .tsv is actually correctly saved and can be correctly opened as a DataFrame in Pandas)
    ntwrk_df.to_json(Path.joinpath(out_dir_ntwrk_table, 'ntwrk_unfilt.json'), index=False)
    ntwrk_df_filt.to_json(Path.joinpath(out_dir_ntwrk_table, 'ntwrk_filt.json'), index=False)
    ntwrk_df_majcomp.to_json(Path.joinpath(out_dir_ntwrk_table, 'ntwrk_majcomp.json'), index=False)

    ntwrk_df.drop(columns=['edges_fluor', 'ntwrk_fluor'], inplace=True)
    ntwrk_df_filt.drop(columns=['edges_fluor', 'ntwrk_fluor'], inplace=True)
    ntwrk_df_majcomp.drop(columns=['edges_fluor', 'ntwrk_fluor'], inplace=True)

    ntwrk_df.to_csv(Path.joinpath(out_dir_ntwrk_table, 'ntwrk_unfilt.tsv'), sep='\t', index=False)
    ntwrk_df_filt.to_csv(Path.joinpath(out_dir_ntwrk_table, 'ntwrk_filt.tsv'), sep='\t', index=False)
    ntwrk_df_majcomp.to_csv(Path.joinpath(out_dir_ntwrk_table, 'ntwrk_majcomp.tsv'), sep='\t', index=False)

    print('Done.')
