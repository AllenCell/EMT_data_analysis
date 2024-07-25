## PROJECT TITLE

**ZO1_NETWORK_ANALYSIS**

## PURPOSE
To analyze and compare the changes in the organization of the tight junction protein ZO-1 over time during the process of epithelial to mesenchymal transition for "colony EMT" (2D MG colonies), "flat colony EMT" (2D PLF colonies), and "lumenoid EMT" (3D lumenoid-forming colonies).

## WORKFLOW
The Python scripts that were run to analyze the ZO-1 segmentation masks and produce the figures used in the initial submission are presented below in sequential order. 
The nomenclature is as follows:

#. <tt>script_name.py</tt>:
input -> output

1. <tt>label_segmentation_mp.py</tt>:
unlabeled ZO-1 segmentations -> labeled versions of the segmentations

2. <tt>array_to_graph_mp.py</tt>:
labeled versions of the segmentations from "1." -> skeleton, edge, and node image representations of labeled segmentations and .tsv files with network, node, and edge information

3. <tt>concat_tables.py</tt>: all .tsv files from "2." -> a single large .tsv file with network, node, and edge information

4. <tt>ntwrks_postprocessing1_mp.py</tt>: the single large .tsv from "3." -> a filtered version of the .tsv from "4." that only includes timepoints before the "t_crop" column defined in <tt>annotations/zo1_to_seg_done.csv</tt> and the network labels found within the cropped region

5. <tt>ntwrks_postprocessing2_mp.py</tt>: the .tsv from "4." -> filtered .tsv that only includes the networks with a median fluorescence above some threshold and output that table as well as the images with filtered labels

6. <tt>network_plots.py</tt>:
the .tsv produced by "3." and "5." above -> various plots

**NOTE: BEWARE -- The output of this workflow will produce about 136 GB of data.**

## USAGE
The workflow was run sequentially in a virtual environment containing the packages and package versions found in the requirements.txt file (located in the same location as this README).
The Python version used was Python 3.11.5.
The file containing my annotations of movies (called <tt>zo1_to_seg_done.csv</tt> found in the <tt>annotations</tt> folder) is required for multiple scripts to execute and contains information on filepaths, plate barcodes, imaging positions, and manually determined migration timings, in addition to other information.
1. create and activate a virtual environment with the packages and package versions specified in <tt>requirements.txt</tt> *
2. run the files found in the "WORKFLOW" section in their specified order
3. the images and plots used in the manuscript are... **TODO**

\* packages can be installed from the <tt>requirements.txt</tt> file by making the folder with the <tt>requirements.txt</tt> file your current working directory and running

            python -m pip install -r requirements.txt

## TO-DO
- replace the <tt>AICSImageIO</tt> package with the <tt>BIOIO</tt> package
- save files in <tt>.OME.TIFF</tt> format instead of <tt>.TIFF</tt>
- if this workflow is included in the initial submission, then update the filepaths currently pointing to VAST folders (e.g. the ones found in "/annotations/zo1_to_seg_done.csv" and in "label_segmentations.py") to point to the data uploaded to the S3 bucket
