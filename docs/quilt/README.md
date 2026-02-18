# Maintainer

Allen Institute for Cell Science ([cells@alleninstitute.org](mailto:cells@alleninstitute.org))

# Overview

The Allen Institute for Cell Science is providing the datasets associated with the manuscript “A human induced pluripotent stem (hiPS) cell model for the holistic study of epithelial to mesenchymal transitions (EMTs)” (Hookway et al. 2024, bioRxiv: https://www.biorxiv.org/content/10.1101/2024.08.16.608353v1)

# Contents & Usage:

This dataset can be accessed on quilt [here](https://open.quiltdata.com/b/allencell/tree/aics/emt_timelapse_dataset/).

The `emt_timelapse_dataset` directory is organized in the following structure.

```
emt_timelapse_dataset
├── data
├── manifests
└── supplemental_files
```

## data

The `data` directory contains all the data used in the study “A human induced pluripotent stem (hiPS) cell model for the holistic study of epithelial to mesenchymal transitions (EMTs)”. Each image (timelapse/fixed) has a unique `Data ID`. 
The `Data ID` contains the unique barcode and the unique scene information associated with each image. **Further, for each fixed immunofluorescent image, the `Data ID` also includes the well label information.** 

For convenience, each specific OME-Zarr file-type associated with a single image has been assigned a specific filename extension (shown in the Table below). 

Hence, each file in the `data` directory has a naming convention: `data-id_file-extension.ome.zarr`.

File type | Filename Extension
----------|-----------------
Microscope images | `raw_converted.ome.zarr`
All cells mask | `all_cells_mask.ome.zarr`
H2B nuclei segmentation mask | `H2B_nuclear_segmentation.ome.zarr`
H2B nuclei segmentation mask | `H2B_nuclear_segmentation.ome.zarr`
CytoGFP ground truth segmentation mask | `cytoGFP_ground_truth_segmentation.ome.zarr`
CollagenIV ground truth segmentation mask | `collagenIV_ground_truth_segmentation.ome.zarr`
CollagenIV segmentation probability | `collagenIV_segmentation_probability.ome.zarr`
CollagenIV segmentaion mesh | `collagenIV_segmentation_mesh`

**For fixed immunofluorescent images - filename extension also contains antibody information**

For improved data accessibility we provide further information in the `manifests` section below.

## manifests

The `manifests` directory contains 6 manifests for streamlined data accessibility. Each manifest file and its contents are explained below.

* `imaging_and_segmentation_data.csv`: contains both the links to download and links to visualize without download, all the imaging data, segmentation data, and mesh data. This csv also provides columns that contain all accompanying metadata from this study.

* `Imaging_and_segmentation_data_column_description.csv`: A complete description of each column label present in the `imaging_and_segmentation_data.csv` is provided here.

* `Image_analysis_extracted_features.csv`: contains all the features extracted from the `raw_converted.ome.zarr` files using the corresponding `all_cells_mask.ome.zarr` segmentation files. Further, the metrics computed from the intensity trajectories over time for each movie is also provided. Features and metrics are given for each time point and Z-plane combination, respectively for each movie.

* `Image_analysis_extracted_features_column_description.csv`: A complete description of each column label present in the `Image_analysis_extracted_features.csv` is provided here.

* `Migration_timing_through_mesh_extracted_features.csv`: contains class annotations for the centroids of individual nuclei, indicating whether each centroid is inside or outside the `CollagenIV_segmentation_mesh`.

* `Migration_timing_through_mesh_extracted_features_column_description.csv`: A complete description of each column label present in the `Migration_timing_through_mesh_extracted_features.csv` is provided here.

* `ddPCR_SNAI1_HPRT1_Ratios.csv` & `ddPCR_TBXT_HPRT1_Ratios.csv`: [Non-imaged-based data] Ratios of positive droplets for SNAI1- and TBXT-targeting CRISPRi cell lines and controls, normalized to the housekeeping gene HPRT1, are provided for each individual lysate used in ddPCR-based validation of CRISPRi-mediated knockdown. This represents the only dataset in the study derived from non–image-based measurements 


## supplemental_files

The `supplemental_files` directory contains model checkpoints (`cytodl_checkpoints`) and CytoDL configuration files (`cytodl_configuration_files`) used for all cells mask and collagenIV mask generation.

# Usage directions

## Download
Each file present in the `data` directory (`movie-id_file-extension.ome.zarr`) can be downloaded using the s3 URI and URL provided in the `imaging_and_segmentation_data.csv`.

## Visualization
To visualize the data in 3D without the need for download, we recommend [using the BioFile Finder app](https://bff.allencell.org/app?c=Gene%3A0.25%2CMicroscope+Modality%3A0.25%2CTimelapse+Duration%3A0.25%2CTimelapse+Interval%3A0.25%2CUsed+For%3A0.25&group=Perturbation&group=Experimental+Condition&group=Gene&filter=%7B%22name%22%3A%22Fixation+Status%22%2C%22value%22%3A%22Live+Cells%22%2C%22type%22%3A%22default%22%7D&openFolder=%5B%22No+perturbation%22%5D&openFolder=%5B%22No+perturbation%22%2C%223D+lumenoid+EMT%22%5D&openFolder=%5B%22No+perturbation%22%2C%223D+lumenoid+EMT%22%2C%22HIST1H2BJ%22%5D&source=%7B%22name%22%3A%22imaging_and_segmentation_data.csv+%2815%2F08%2F2025+13%3A09%3A25%29%22%2C%22type%22%3A%22csv%22%2C%22uri%22%3A%22https%3A%2F%2Fallencell.s3.amazonaws.com%2Faics%2Femt_timelapse_dataset%2Fmanifests%2Fimaging_and_segmentation_data.csv%22%7D&sourceMetadata=%7B%22name%22%3A%22Imaging_and_segmentation_data_column_description.csv+%2815%2F08%2F2025+13%3A09%3A32%29%22%2C%22type%22%3A%22csv%22%2C%22uri%22%3A%22https%3A%2F%2Fallencell.s3.amazonaws.com%2Faics%2Femt_timelapse_dataset%2Fmanifests%2FImaging_and_segmentation_data_column_description.csv%22%7D), pre-loaded with all the images and segmentations used in this study

# Licensing
For questions on licensing please refer to [https://www.allencell.org/terms-of-use.html](https://www.allencell.org/terms-of-use.html). 

# Feedback
Feedback on benefits and issues you discovered while using this data is greatly appreciated via the [Allen Cell discussion forum](https://forum.allencell.org/).
