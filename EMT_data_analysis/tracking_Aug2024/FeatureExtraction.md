 # Summary of Major Changes to Feature_extraction.py Since Sept 2024

  Timeline of Changes

  | Date       | Commit  | Summary                                   |
  |------------|---------|-------------------------------------------|
  | 2025-04-03 | de39940 | Modified for CSV input                    |
  | 2025-04-03 | ce792dc | Corrections to segmentation image loading |
  | 2025-04-11 | bfa002a | Bugfix                                    |
  | 2025-04-14 | e25cb77 | Path updates                              |
  | 2025-04-29 | 77d8cab | Processing resubmission data              |
  | 2025-07-30 | 6fc7597 | Restored to using Quilt data              |
  | 2025-08-21 | 61904ba | Column name updates                       |
  | 2025-08-21 | afd6cdd | Feature column name updates               |
  | 2025-08-26 | 68e846c | Fix raw file path column                  |
  | 2025-11-10 | f802b64 | Fix column name in feature extraction     |

  ---
  Major Changes

  1. Data Source Changes (April → July 2025)

  - April 2025: Code was modified to accept a CSV manifest as input parameter,
  with hardcoded paths to Filip's local directories for processing new EOMES
  timelapse data
  - July 2025: Reverted back to using Quilt data via
  io.load_imaging_and_segmentation_dataset()

  2. Column Name Updates (August 2025)

  Changed column names to match the new manifest schema:
  - 'Movie ID' → 'Data ID' (for grouping and output)
  - 'File Path' → 'Raw File URL' (for raw data access)

  3. Processing Logic Changes

  - Before July 2025: Processed data row-by-row from manifest with
  per-timepoint segmentation files
  for _, df_tp in df_movie.iterrows():
      seg_reader = BioImage(df_tp["All Cells Mask URL"])
  - After July 2025: Restored to processing by timepoint index with single
  segmentation file per movie
  seg_reader = BioImage(seg_path)  # loaded once per movie
  for frame in range(max_timepoint):
      seg_img = seg_reader.get_image_dask_data("ZYX", T=frame)

  4. Timepoint Handling

  - Before: Used df_movie['Timepoint'].max() from manifest
  - After: Fixed to first 48 hours (98 timepoints): min(98, df_movie['Image 
  Size T'].values[0])

  5. Entry Point Changes

  - April 2025: Used fire library for CLI with barcode parameter
  - July 2025+: Restored to using io.setup_base_directory_name() for output
  directory

  ---

  # Analysis of Changes

  Core Algorithm (UNCHANGED)

  The actual feature computation logic remained the same throughout:
  ```
  for z in range(seg_img.shape[0]):
      mask = seg_img[z] > 0
      area = np.sum(mask)
      mean_intensity = np.mean(raw_img[z][mask]) if area > 0 else 0
      total_intensity = np.sum(raw_img[z][mask]) if area > 0 else 0
  ```

  This Z-plane-by-Z-plane computation of:
  - Area of all cells mask
  - Mean intensity per Z
  - Total intensity per Z

  Has not changed.

  What Changed (All IO/Infrastructure)

  | Change Type         | Description                                  | Breaking?                   |
  |---------------------|----------------------------------------------|-----------------------------|
  | Data source         | CSV manifest → Quilt → back to Quilt         | No (reverted)               |
  | Column names        | Movie ID → Data ID, File Path → Raw File URL | No (manifest schema change) |
  | Timepoint iteration | Row-by-row vs index-based loop               | No (same result)            |
  | Entry point         | fire CLI → standard __main__                 | No                          |
  | Output columns      | Movie ID → Data ID in results                | Minor                       |

  Only Potentially Significant Change

  The timepoint limit logic:
  - Before (April): max_timepoint = df_movie['Timepoint'].max() (process all available)
  - After (July): max_timepoint = min(98, df_movie['Image Size T'].values[0]) (cap at 48 hours)

  This is a data filtering change, not a logic change. If a movie has >98 timepoints, timepoints beyond 98 will not be
   processed in the current version. But this was the original behavior that was restored, not a new change.

  Conclusion

  No breaking or significant logic changes. All changes were:
  1. Adapting to manifest schema changes (Movie ID → Data ID)
  2. Adapting to URL column name changes
  3. Temporary modifications for processing specific datasets (later reverted)
  4. Restoring original Quilt-based data loading

  The feature extraction algorithm itself is identical.
  Current State (Nov 2025)

  The script now:
  1. Loads data from Quilt via io.load_imaging_and_segmentation_dataset()
  2. Groups by 'Data ID'
  3. Uses 'Raw File URL' for raw image data
  4. Uses 'All Cells Mask URL' for segmentation
  5. Processes first 98 timepoints (48 hours)
  6. Outputs features per Z-plane including intensity and area measurements
