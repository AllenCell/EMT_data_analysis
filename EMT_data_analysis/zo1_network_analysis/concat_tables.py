## For working with filepaths and directories
from pathlib import Path
## For working with DataFrames
import pandas as pd



## Get some local filename and directory locations:
sct_fpath = Path(__file__)
cwd = sct_fpath.parent
sct_fstem = sct_fpath.stem
barcode_annotations = pd.read_csv(Path.joinpath(cwd, 'annotations/zo1_to_seg_done.csv'))
data_dir = Path.joinpath(cwd, 'array_to_graph_mp_out/network_tables')

## Create a folder for image output if it doesn't already exist:
out_dir = Path.joinpath(cwd, sct_fstem + '_out')
Path.mkdir(out_dir, exist_ok=True)

## Compile network data tables into a single dataframe:
nodes_df_paths = []
edges_df_paths = []
ntwrk_df_paths = []
for fpath in data_dir.glob('*/*/*'):
    fname = fpath.parts[-1]
    posit_well = fpath.parts[-2]
    barcode = fpath.parts[-3]
    if '_nodes' in fname:
        nodes_df_paths.append(fpath)
    elif '_edges' in fname:
        edges_df_paths.append(fpath)
    elif '_ntwrk' in fname:
        ntwrk_df_paths.append(fpath)

## Construct the dataset
nodes_df = pd.concat([pd.read_csv(fpath, sep='\t') for fpath in nodes_df_paths])
edges_df = pd.concat([pd.read_csv(fpath, sep='\t') for fpath in edges_df_paths])
ntwrk_df = pd.concat([pd.read_csv(fpath, sep='\t') for fpath in ntwrk_df_paths])

## and then save the dataset
nodes_df.to_csv(Path.joinpath(out_dir, 'nodes_dataset.tsv'), sep='\t', index=False)
edges_df.to_csv(Path.joinpath(out_dir, 'edges_dataset.tsv'), sep='\t', index=False)
ntwrk_df.to_csv(Path.joinpath(out_dir, 'ntwrk_dataset.tsv'), sep='\t', index=False)
