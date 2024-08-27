# Maintainer

Allen Institute for Cell Science ([cells@alleninstitute.org](mailto:cells@alleninstitute.org))

# Overview

The Allen Institute for Cell Science is providing the datasets associated with the manuscript “A human induced pluripotent stem (hiPS) cell model for the holistic study of epithelial to mesenchymal transitions (EMTs)” (Hookway et al. 2024, bioRxiv: https://www.biorxiv.org/content/10.1101/2024.08.16.608353v1)

# Contents & Usage:

This dataset can be accessed on quilt [here](https://open.quiltdata.com/b/allencell/tree/aics/emt_timelapse_dataset/).

The `emt_timelapse_dataset` directory is organized in the following structure.

<img src="https://github.com/AllenCell/EMT_data_analysis/blob/bioRxiv-v1/docs/quilt/dataset_structure.svg" width="500px" />

## data

The `data` directory contains all the data used in the study “A human induced pluripotent stem (hiPS) cell model for the holistic study of epithelial to mesenchymal transitions (EMTs)”. Each movie has a unique movie ID. For convenience, each specific OME-Zarr file-type associated with a single movie has been assigned a specific filename extension (shown in the Table below). Hence, each file in the `data` directory has a naming convention: `movie-id_file-extension.ome.zarr`.

File type | Filename Extension
----------|-----------------
Microscope images | `raw_converted.ome.zarr`
All cells mask | `all_cells_mask.ome.zarr`
H2B nuclei segmentation mask | `H2B_nuclear_segmentation.ome.zarr`
EOMES nuclei segmentation mask | `EOMES_nuclear_segmentation.ome.zarr`
CytoGFP ground truth segmentation mask | `cytoGFP_ground_truth_segmentation.ome.zarr`
CollagenIV ground truth segmentation mask | `collagenIV_ground_truth_segmentation.ome.zarr`
CollagenIV segmentation probability | `collagenIV_segmentation_probability.ome.zarr`
CollagenIV segmentaion mesh | `collagenIV_segmentation_mesh`

For improved data accessibility we provide further information in the `manifests` section below.

## manifests

The `manifests` directory contains 6 manifests for streamlined data accessibility. Each manifest file and its contents are explained below.

* `imaging_and_segmentation_data.csv`: contains both the links to download and links to visualize without download, all the imaging data, segmentation data, and mesh data. This csv also provides columns that contain all accompanying metadata from this study.

* `Imaging_and_segmentation_data_column_description.csv`: A complete description of each column label present in the `imaging_and_segmentation_data.csv` is provided here.

* `Image_analysis_extracted_features.csv`: contains all the features extracted from the `raw_converted.ome.zarr` files using the corresponding `all_cells_mask.ome.zarr` segmentation files. Further, the metrics computed from the intensity trajectories over time for each movie is also provided. Features and metrics are given for each time point and Z-plane combination, respectively for each movie.

* `Image_analysis_extracted_features_column_description.csv`: A complete description of each column label present in the `Image_analysis_extracted_features.csv` is provided here.

* `Migration_timing_through_mesh_extracted_features.csv`: contains class annotations for the centroids of individual nuclei, indicating whether each centroid is inside or outside the `CollagenIV_segmentation_mesh`.

* `Migration_timing_through_mesh_extracted_features_column_description.csv`: A complete description of each column label present in the `Migration_timing_through_mesh_extracted_features.csv` is provided here.


## supplemental_files

The `supplemental_files` directory contains model checkpoints (`cytodl_checkpoints`) and CytoDL configuration files (`cytodl_configuration_files`) used for all cells mask and collagenIV mask generation.

# Usage directions

## Download
Each file present in the `data` directory (`movie-id_file-extension.ome.zarr`) can be downloaded using the s3 URI and URL provided in the `imaging_and_segmentation_data.csv`.

## Visualization
To visualize the data in 3D without the need for download, we recommend [using the BioFile Finder app](https://biofile-finder.allencell.org/app?group=Experimental+Condition&group=Gene&source=%7B%22name%22%3A%22imaging_and_segmentation_data.csv+%288%2F15%2F2024+4%3A26%3A03+PM%29%22%2C%22type%22%3A%22csv%22%2C%22uri%22%3A%22https%3A%2F%2Fallencell.s3.amazonaws.com%2Faics%2Femt_timelapse_dataset%2Fmanifests%2Fimaging_and_segmentation_data.csv%3FversionId%3DWmTjARBNL4rNJhV4N7YFYr2dKHWlCHwc%22%7D&sourceMetadata=%7B%22name%22%3A%22Imaging_and_segmentation_data_column_description.csv+%288%2F15%2F2024+4%3A26%3A02+PM%29%22%2C%22type%22%3A%22csv%22%2C%22uri%22%3A%22https%3A%2F%2Fallencell.s3.amazonaws.com%2Faics%2Femt_timelapse_dataset%2Fmanifests%2FImaging_and_segmentation_data_column_description.csv%3FversionId%3D.bmbr.UUT06F9nupeuwxVBYuTMyKyYu6%22%7D), pre-loaded with all the images and segmentations used in this study

# Licensing
For questions on licensing please refer to [https://www.allencell.org/terms-of-use.html](https://www.allencell.org/terms-of-use.html). 

# Feedback
Feedback on benefits and issues you discovered while using this data is greatly appreciated via the [Allen Cell discussion forum](https://forum.allencell.org/).
