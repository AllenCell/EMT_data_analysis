import warnings
import time
import numpy as np
import pandas as pd
from pathlib import Path
from bioio import BioImage
from joblib import Parallel, delayed
from aiohttp.client_exceptions import ServerDisconnectedError, ClientError
from bioio_base.exceptions import UnsupportedFileFormatError
from EMT_data_analysis.tools import io, alignment

warnings.filterwarnings("ignore")


def get_channel_camera(channel_num, bf_ch, ch_488, ch_561, ch_638):
    """
    Determine which camera a channel is on based on wavelength mapping.

    Camera assignment rules:
    - Camera 1: Brightfield, 638nm wavelength
    - Camera 2: 488nm, 561nm wavelength

    Parameters
    ----------
    channel_num : int
        Channel number to check
    bf_ch : int or float
        Brightfield channel number from manifest
    ch_488 : int or float
        488nm wavelength channel number from manifest
    ch_561 : int or float
        561nm wavelength channel number from manifest
    ch_638 : int or float
        638nm wavelength channel number from manifest

    Returns
    -------
    int or None
        1 for Camera 1, 2 for Camera 2, None if unknown
    """
    # Handle NaN values by converting to -1 (impossible channel number)
    bf_ch = -1 if pd.isna(bf_ch) else int(bf_ch)
    ch_488 = -1 if pd.isna(ch_488) else int(ch_488)
    ch_561 = -1 if pd.isna(ch_561) else int(ch_561)
    ch_638 = -1 if pd.isna(ch_638) else int(ch_638)

    if channel_num == bf_ch or channel_num == ch_638:
        return 1  # Camera 1: Brightfield, 638nm
    elif channel_num == ch_488 or channel_num == ch_561:
        return 2  # Camera 2: 488nm, 561nm
    return None  # Unknown


def load_image_with_retry(dask_array, max_retries=5):
    """Load dask array with retry logic for transient network errors."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return dask_array.compute()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 30)
                time.sleep(wait_time)
            continue
    raise last_error


def open_bioimage_with_retry(path, max_retries=10):
    """Open BioImage with retry logic for transient network errors."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return BioImage(path)
        except (ServerDisconnectedError, ClientError, ConnectionError, TimeoutError, UnsupportedFileFormatError) as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 60)
                time.sleep(wait_time)
            continue
    raise last_error


def _parse_timelapse_interval_minutes(interval_str):
    """
    Parse timelapse interval string to minutes.

    Parameters
    ----------
    interval_str : str or None
        Timelapse interval string (e.g., '30 minutes', '1 hour', '3 minutes')

    Returns
    -------
    float
        Interval in minutes, defaults to 30 if not parseable
    """
    if pd.isna(interval_str) or interval_str is None:
        return 30.0  # Default to 30 minutes (most common)

    interval_str = str(interval_str).lower().strip()

    if 'minute' in interval_str:
        try:
            return float(''.join(filter(lambda c: c.isdigit() or c == '.', interval_str)))
        except ValueError:
            return 30.0
    elif 'hour' in interval_str:
        try:
            hours = float(''.join(filter(lambda c: c.isdigit() or c == '.', interval_str)))
            return hours * 60
        except ValueError:
            return 60.0
    else:
        return 30.0  # Default


def _process_single_movie(movie_id, raw_path, seg_path, matrix_string, gene, experimental_condition,
                          output_folder, align=True, fixation_status=None, fixation_time_hours=None,
                          timelapse_interval=None, wavelength_channels=None):
    """
    Process a single movie to extract features. Designed for parallel execution.

    Parameters
    ----------
    movie_id : str
        Data ID of the movie
    raw_path : str
        URL/path to raw image data
    seg_path : str
        URL/path to segmentation mask
    matrix_string : str
        Dual camera alignment matrix string
    gene : str
        Gene name
    experimental_condition : str
        Experimental condition
    output_folder : Path
        Output folder for CSV files
    align : bool
        Whether to apply alignment
    fixation_status : str or None
        'Fixed Cells' or 'Live Cells' or None
    fixation_time_hours : float or None
        Time of fixation post EMT induction (in hours) for fixed cells
    timelapse_interval : str or None
        Timelapse interval string from metadata (e.g., '30 minutes')
    wavelength_channels : dict or None
        Dictionary mapping wavelength to channel number:
        {'bf': int, '488': int, '561': int, '638': int}
        Used to determine which camera each channel is on.

    Returns
    -------
    tuple
        (movie_id, success, error_message)
    """
    # Determine if this is a fixed cell (immunostaining) experiment
    is_fixed_cell = (fixation_status == 'Fixed Cells')

    # Parse timelapse interval for timepoint calculation
    # For fixed cells, use 30 min standard interval to match live timelapse frame numbering
    interval_minutes = _parse_timelapse_interval_minutes(timelapse_interval) if not is_fixed_cell else 30.0
    out_fn = Path(output_folder) / f"Features_bf_colony_mask_{movie_id}.csv"

    # Skip if already processed
    if out_fn.exists():
        return (movie_id, True, "already exists")

    # Skip if missing data
    if pd.isna(raw_path) or pd.isna(seg_path):
        return (movie_id, False, "missing raw data or segmentation mask")

    try:
        # Load images
        raw_reader = open_bioimage_with_retry(raw_path)
        seg_reader = open_bioimage_with_retry(seg_path)

        # Determine which channels to extract based on available channels
        num_channels = raw_reader.dims.C
        if num_channels == 1:
            # Single channel image: use Channel 0
            channels_to_extract = [0]
            main_channel = 0
        else:
            # Multi-channel image: use Channel 1 as main, add 2 and 3 if available
            channels_to_extract = [1]  # Channel 1 (main fluorescence)
            main_channel = 1
            if num_channels > 2:
                channels_to_extract.append(2)  # Channel 2
            if num_channels > 3:
                channels_to_extract.append(3)  # Channel 3

        # Compute alignment transform once per movie (will be applied per-channel as needed)
        # Segmentation (All Cells Mask) is from Brightfield = Camera 1
        # Alignment is only needed when raw channel is on Camera 2 (488nm, 561nm)
        base_transform = None
        if align and not pd.isna(matrix_string):
            matrix = alignment.parse_rotation_matrix_from_string(matrix_string)
            base_transform = alignment.get_alignment_matrix(matrix)

        # Determine which channels need alignment based on camera assignment
        # Segmentation is Camera 1, so we only align when raw channel is Camera 2
        channels_need_alignment = {}
        if wavelength_channels is not None and base_transform is not None:
            for ch in channels_to_extract:
                ch_camera = get_channel_camera(
                    ch,
                    wavelength_channels.get('bf'),
                    wavelength_channels.get('488'),
                    wavelength_channels.get('561'),
                    wavelength_channels.get('638')
                )
                # Segmentation is Camera 1, raw channel is Camera 2 -> need alignment
                # Use INVERSE transform: Camera 1 (seg) -> Camera 2 (raw)
                if ch_camera == 2:
                    channels_need_alignment[ch] = base_transform.inverse
                else:
                    # Same camera (Camera 1) or unknown -> no alignment
                    channels_need_alignment[ch] = None
        else:
            # Fallback: if no wavelength info, apply alignment to all channels (old behavior)
            for ch in channels_to_extract:
                channels_need_alignment[ch] = base_transform.inverse if base_transform else None

        # Determine timepoints to process (first 48 hours = 98 timepoints)
        num_timepoints = raw_reader.dims.T
        max_timepoint = min(num_timepoints, 98)

        df_result = []

        for frame in range(max_timepoint):
            # Load segmentation for this frame
            seg_img = load_image_with_retry(seg_reader.get_image_dask_data("ZYX", T=frame))

            # Load raw images for all channels we need
            raw_imgs = {}
            for ch in channels_to_extract:
                raw_imgs[ch] = load_image_with_retry(raw_reader.get_image_dask_data("ZYX", C=ch, T=frame))

            for z, seg in enumerate(seg_img):
                # Get the transform for the main channel
                main_ch_transform = channels_need_alignment.get(main_channel)

                # Apply alignment if needed for main channel
                if main_ch_transform is not None:
                    seg_aligned = alignment.align_image(seg, main_ch_transform)
                else:
                    seg_aligned = seg

                mask = np.bool_(seg_aligned)
                area = np.count_nonzero(mask)

                # Extract intensity for main channel (Channel 0 for single-channel, Channel 1 otherwise)
                masked_values = raw_imgs[main_channel][z][mask]
                mean_intensity = np.mean(masked_values) if area > 0 else np.nan
                total_intensity = np.sum(masked_values) if area > 0 else np.nan

                # For fixed cells, use fixation time to calculate timepoint
                if is_fixed_cell and fixation_time_hours is not None and not pd.isna(fixation_time_hours):
                    timepoint = int(fixation_time_hours * 60 / interval_minutes)
                else:
                    timepoint = frame

                row = {
                    "Z plane": z,
                    "Timepoint": timepoint,
                    "Data ID": movie_id,
                    "Mean intensity per Z": mean_intensity,
                    "Total intensity per Z": total_intensity,
                    "Area of all cells mask per Z (pixels)": area
                }

                # Extract intensity for Channel 2 if available
                if 2 in raw_imgs:
                    # Check if Channel 2 needs different alignment than main channel
                    ch2_transform = channels_need_alignment.get(2)
                    if ch2_transform != main_ch_transform:
                        # Need different alignment for this channel
                        if ch2_transform is not None:
                            seg_ch2 = alignment.align_image(seg, ch2_transform)
                        else:
                            seg_ch2 = seg
                        mask_ch2 = np.bool_(seg_ch2)
                        area_ch2 = np.count_nonzero(mask_ch2)
                        masked_values_ch2 = raw_imgs[2][z][mask_ch2]
                        row["Mean intensity per Z (Channel 2)"] = np.mean(masked_values_ch2) if area_ch2 > 0 else np.nan
                        row["Total intensity per Z (Channel 2)"] = np.sum(masked_values_ch2) if area_ch2 > 0 else np.nan
                    else:
                        # Same alignment as main channel, reuse mask
                        masked_values_ch2 = raw_imgs[2][z][mask]
                        row["Mean intensity per Z (Channel 2)"] = np.mean(masked_values_ch2) if area > 0 else np.nan
                        row["Total intensity per Z (Channel 2)"] = np.sum(masked_values_ch2) if area > 0 else np.nan

                # Extract intensity for Channel 3 if available
                if 3 in raw_imgs:
                    # Check if Channel 3 needs different alignment than main channel
                    ch3_transform = channels_need_alignment.get(3)
                    if ch3_transform != main_ch_transform:
                        # Need different alignment for this channel
                        if ch3_transform is not None:
                            seg_ch3 = alignment.align_image(seg, ch3_transform)
                        else:
                            seg_ch3 = seg
                        mask_ch3 = np.bool_(seg_ch3)
                        area_ch3 = np.count_nonzero(mask_ch3)
                        masked_values_ch3 = raw_imgs[3][z][mask_ch3]
                        row["Mean intensity per Z (Channel 3)"] = np.mean(masked_values_ch3) if area_ch3 > 0 else np.nan
                        row["Total intensity per Z (Channel 3)"] = np.sum(masked_values_ch3) if area_ch3 > 0 else np.nan
                    else:
                        # Same alignment as main channel, reuse mask
                        masked_values_ch3 = raw_imgs[3][z][mask]
                        row["Mean intensity per Z (Channel 3)"] = np.mean(masked_values_ch3) if area > 0 else np.nan
                        row["Total intensity per Z (Channel 3)"] = np.sum(masked_values_ch3) if area > 0 else np.nan

                df_result.append(row)

        # Create DataFrame and add metadata
        df_result = pd.DataFrame(df_result)
        df_result["Gene"] = gene
        df_result["Experimental Condition"] = experimental_condition
        df_result.to_csv(out_fn, index=False)

        return (movie_id, True, "success")

    except Exception as e:
        return (movie_id, False, f"{type(e).__name__}: {str(e)}")

def compute_bf_colony_features_all_movies(output_folder, align=True, n_jobs=32):
    '''
    Computes area of the bright field colony mask at every z position
    and extracts corresponding intensity values from the fluorescence
    channels. Extracts Channel 1 (main), and Channel 2/3 if available.

    Parameters
    ----------
    output_folder : path
        Folder path where feature csv for each movie is stored
    align : bool
        Enable alignment of the image using the barcode of the movie
    n_jobs : int
        Number of parallel jobs. Default: 32.
    '''

    df = io.load_imaging_and_segmentation_dataset()
    print(f"Dataset loaded. Shape: {df.shape}.")

    # Prepare arguments for parallel processing
    movie_args = []
    for movie_id, df_movie in df.groupby('Data ID'):
        raw_path = df_movie["Raw File URL"].values[0]
        seg_path = df_movie["All Cells Mask URL"].values[0]
        matrix_string = df_movie["Dual Camera Alignment Matrix Value"].values[0]
        gene = df_movie["Gene"].values[0]
        experimental_condition = df_movie["Experimental Condition"].values[0]

        # Get fixation info for immunostaining experiments
        fixation_status = df_movie["Fixation Status"].values[0] if "Fixation Status" in df_movie.columns else None
        fixation_time_hours = df_movie["Time Of Fixation Post EMT Induction (In Hours)"].values[0] if "Time Of Fixation Post EMT Induction (In Hours)" in df_movie.columns else None
        timelapse_interval = df_movie["Timelapse Interval"].values[0] if "Timelapse Interval" in df_movie.columns else None

        # Get wavelength-to-channel mapping for camera assignment
        # Camera 1: Brightfield, 638nm | Camera 2: 488nm, 561nm
        wavelength_channels = {
            'bf': df_movie["Brightfield Channel Number In The raw File"].values[0] if "Brightfield Channel Number In The raw File" in df_movie.columns else None,
            '488': df_movie["488 wavelength Channel Number In The raw File"].values[0] if "488 wavelength Channel Number In The raw File" in df_movie.columns else None,
            '561': df_movie["561 Wavelength Channel Number In The raw File"].values[0] if "561 Wavelength Channel Number In The raw File" in df_movie.columns else None,
            '638': df_movie["638 wavelength Channel Number In The raw File"].values[0] if "638 wavelength Channel Number In The raw File" in df_movie.columns else None,
        }

        movie_args.append((
            movie_id, raw_path, seg_path, matrix_string,
            gene, experimental_condition, output_folder, align,
            fixation_status, fixation_time_hours, timelapse_interval,
            wavelength_channels
        ))

    print(f"Processing {len(movie_args)} movies with {n_jobs} parallel jobs...")

    # Process movies in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_single_movie)(
            movie_id, raw_path, seg_path, matrix_string,
            gene, experimental_condition, output_folder, align,
            fixation_status, fixation_time_hours, timelapse_interval,
            wavelength_channels
        )
        for movie_id, raw_path, seg_path, matrix_string,
            gene, experimental_condition, output_folder, align,
            fixation_status, fixation_time_hours, timelapse_interval,
            wavelength_channels in movie_args
    )

    # Report results
    success_count = sum(1 for _, success, _ in results if success)
    failed = [(movie_id, msg) for movie_id, success, msg in results if not success and msg != "already exists"]
    skipped = [(movie_id, msg) for movie_id, success, msg in results if success and msg == "already exists"]

    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Successful: {success_count}")
    print(f"Skipped (already exists): {len(skipped)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed movies:")
        for movie_id, msg in failed:
            print(f"  - {movie_id}: {msg}")

if __name__ == '__main__':
    base_results_dir = io.setup_base_directory_name("feature_extraction")
    compute_bf_colony_features_all_movies(output_folder=base_results_dir)
