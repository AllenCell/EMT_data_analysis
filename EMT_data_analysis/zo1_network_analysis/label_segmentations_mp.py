## Import packages:
# For working with images
from aicsimageio import AICSImage
from skimage import measure
## For parallel number-crunching
from multiprocessing import Pool
import psutil
## For working with DataFrames
import pandas as pd
## For working with filepaths and directories
from pathlib import Path, WindowsPath
## Regular expressions for searching filenames for time and channel number
import re
## For creating a progress bar
from tqdm import tqdm



def main(fp_pred):

    barcode = fp_pred.parts[-3]
    position_well = fp_pred.parts[-2]
    fname = fp_pred.parts[-1].split('.', 1)[0]

    ## Extract the timepoint and channel of the segmentation from the filename:
    t = re.search('T=[0-9]+', fp_pred.parts[-1])
    t = int(t.group(0).split('=')[-1])

    chan = re.search('C=[0-9]+', fp_pred.parts[-1])
    chan = int(chan.group(0).split('=')[-1])

    ## Let me know where you are in the timelapse...
    print('T=%d... Seg_File=%s'%(t, fp_pred.parts[-1]))

    ## Get the filepaths to the original split position files
    fp_img = [Path(str(r.file_path)) for i,r in barcode_annotations.iterrows() if fname in str(r.file_name)]
    if fp_img:
        ## If the fp_img is a single item (it _should_ be...) then convert the list to the item
        fp_img = fp_img[-1] if len(fp_img)==1 else fp_img
        ## If fp_img has an empty drive (e.g. this happens with paths to the Isilon "/allen/",
        ## then you will need to append a forward slash "\" to make the path recognizable on Windows) 
        fp_img = Path('\\' + str(fp_img)) if not fp_img.drive and (type(fp_img) == WindowsPath) else fp_img

        ## Open the predicted segmentation file and the original image file:
        pred = AICSImage(fp_pred)

        ## Load the stacks of interest into memory:
        zo1_pred = pred.get_image_data("ZYX")

        ## Label the ZO-1 segmentations:
        zo1_lab = measure.label(zo1_pred, background=0, connectivity=3)


        ## Save the labeled stack for possible later usage
        out_subdir = Path.joinpath(out_dir, Path(barcode), Path(position_well))
        Path.mkdir(out_subdir, parents=True, exist_ok=True)
        
        AICSImage(zo1_lab.astype(float)).save(Path.joinpath(out_subdir, Path('%s_T=%d_C=%d_labeled'%(fname, t, chan) + '.tif')))

    return



## Get some local filename and directory locations:
sct_fpath = Path(__file__)
cwd = sct_fpath.parent
sct_fstem = sct_fpath.stem
barcode_annotations = pd.read_csv(Path.joinpath(cwd, 'annotations/zo1_to_seg_done.csv'))

## Create a folder for image output if it doesn't already exist:
out_dir = Path.joinpath(cwd, sct_fstem + '_out')

Path.mkdir(out_dir, exist_ok=True)

## Get list of Z-stacks of interest
# pred_dir = Path(r'\\allen\aics\assay-dev\users\Suraj\EMT_Work\TJP\dl_test_movies_v3_postprocessed_100_255_fixed')
pred_dir = Path('/allen/aics/assay-dev/users/Suraj/EMT_Work/TJP/dl_test_movies_v3_postprocessed_100_255_fixed')

fp_pred_list = [p for p in pred_dir.glob('*/*/*')]

## Request number of CPUs to use
max_cpu_count = psutil.cpu_count()
n_proc = int(input('How many processors do you want to use? (max cpu count is %d)\n> '%max_cpu_count))
print('Using %d processors...'%n_proc)



if __name__ == "__main__":
    with Pool(processes=n_proc) as pool:
        list(tqdm(pool.imap(main, fp_pred_list, chunksize=10), total=len(fp_pred_list)))
        pool.close()
        pool.join()

