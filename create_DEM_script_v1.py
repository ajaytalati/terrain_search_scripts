#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlined script to download and prepare a projected Digital Elevation Model (DEM)
for a specified Area of Interest.

Downloads DEM tiles covering the area, merges them, and reprojects the result
to the target Coordinate Reference System (CRS).

Formerly part of calculate_regional_steepness.py, now focused solely on DEM prep.
"""

import os
import sys
import warnings
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
import elevation # For downloading DEM
import numpy as np # Used in reprojection function

# --- Configuration ---

# Define paths RELATIVE to the script's location for clarity
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Get directory script is in

# 1. Define Area of Interest Bounds (USER MUST PROVIDE/VERIFY)
# Format: (longitude_min, latitude_min, longitude_max, latitude_max) in WGS84 (EPSG:4326)
# Example bounds for the United Kingdom (including Scotland, NI, etc.) - ADJUST AS NEEDED
AREA_OF_INTEREST_BOUNDS = (-11.0, 49.5, 2.0, 61.0)
AREA_NAME = "united_kingdom" # Used for naming output files/directory

# 2. Define Target Projected CRS (e.g., British National Grid)
TARGET_CRS = "EPSG:27700" # <- UPDATE if a different projection is needed

# 3. Output File Paths - CONSTRUCT ABSOLUTE PATHS
OUTPUT_DIR_NAME = f"outputs_{AREA_NAME}_dem" # e.g., outputs_united_kingdom_dem
OUTPUT_DIR = os.path.join(SCRIPT_DIR, OUTPUT_DIR_NAME) # Absolute path to output dir

RAW_DEM_FILE = os.path.join(OUTPUT_DIR, f"{AREA_NAME}_dem_raw_4326.tif")
PROJECTED_DEM_FILE = os.path.join(OUTPUT_DIR, f"{AREA_NAME}_dem_proj_{TARGET_CRS.split(':')[-1]}.tif") # e.g., uk_dem_proj_27700.tif

# 4. DEM Download Settings
MAX_DOWNLOAD_TILES = 600 # Increase limit slightly for potentially larger areas like UK
# Available products: SRTM1 (approx 30m), SRTM3 (approx 90m), GMTED2010, ETOPO1, GLO30 (Copernicus DEM) etc.
# Check `elevation` library documentation for full list and availability.
DEM_PRODUCT = 'SRTM1' # SRTM 30m (~1 arc-second) - Good balance for many regions

# --- Ensure Output Directory Exists ---
print(f"Ensuring output directory exists: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Add a check to be absolutely sure after creating it
if not os.path.isdir(OUTPUT_DIR):
    print(f"FATAL ERROR: Failed to create or find output directory: {OUTPUT_DIR}")
    sys.exit(1) # Use sys.exit to indicate error

# --- Functions ---

def download_and_prepare_dem(bounds, product, max_tiles, target_crs, raw_dem_path, proj_dem_path):
    """
    Downloads DEM tiles for the specified bounds using the elevation package,
    merges them into a single raw DEM file (if needed), and then reprojects
    that raw DEM to the target CRS.

    Args:
        bounds (tuple): (lon_min, lat_min, lon_max, lat_max) in WGS84.
        product (str): DEM product name recognized by the 'elevation' library (e.g., 'SRTM1').
        max_tiles (int): Maximum number of DEM tiles allowed for download to prevent excessive requests.
        target_crs (str): The target Coordinate Reference System (e.g., 'EPSG:27700').
        raw_dem_path (str): Absolute path where the downloaded and merged raw DEM (in WGS84) will be saved.
        proj_dem_path (str): Absolute path where the final reprojected DEM will be saved.

    Returns:
        str: The path to the projected DEM file if successful, None otherwise.
    """
    # --- Step 1: Download and Merge Raw DEM ---
    if not os.path.exists(raw_dem_path):
        print(f"\n--- Downloading and Merging DEM ---")
        print(f"Area Bounds: {bounds}")
        print(f"DEM Product: {product}")
        print(f"Output Raw DEM: {raw_dem_path}")
        print(f"Max Download Tiles: {max_tiles}")
        print("This may take a significant amount of time and disk space depending on the area size...")
        try:
            # The elevation.clip command handles both downloading and merging tiles within the bounds
            elevation.clip(bounds=bounds, output=raw_dem_path, product=product, max_download_tiles=max_tiles)
            print("Raw DEM download and merge complete.")
            if not os.path.exists(raw_dem_path) or os.path.getsize(raw_dem_path) == 0:
                 print(f"ERROR: Raw DEM file was not created or is empty after download: {raw_dem_path}")
                 return None
        except Exception as e:
            print(f"ERROR during DEM download/merge: {e}")
            print("Check internet connection, disk space, permissions, product availability, and tile limit.")
            # Attempt to clean up potentially incomplete file
            if os.path.exists(raw_dem_path):
                try:
                    os.remove(raw_dem_path)
                except OSError:
                    pass # Ignore cleanup error
            return None
    else:
        print(f"\nRaw DEM file already exists, skipping download: {raw_dem_path}")
        # Basic check if existing file is valid raster
        try:
            with rasterio.open(raw_dem_path) as src:
                print(f"Existing raw DEM CRS: {src.crs}, Size: {src.width}x{src.height}")
        except rasterio.RasterioIOError:
             print(f"ERROR: Existing raw DEM file is corrupted or not a valid raster: {raw_dem_path}")
             print("Please delete it and run the script again to redownload.")
             return None


    # --- Step 2: Reproject Raw DEM ---
    if not os.path.exists(proj_dem_path):
        print(f"\n--- Reprojecting DEM ---")
        print(f"Source Raw DEM: {raw_dem_path}")
        print(f"Target CRS: {target_crs}")
        print(f"Output Projected DEM: {proj_dem_path}")
        print("WARNING: This step can be memory-intensive and may take time for large areas!")
        try:
            with rasterio.open(raw_dem_path) as src:
                src_crs = src.crs
                if not src_crs:
                     print("WARNING: Source DEM CRS is not defined. Assuming EPSG:4326 (WGS84).")
                     src_crs = 'EPSG:4326'

                # Calculate the transform and dimensions for the reprojected raster
                transform, width, height = rasterio.warp.calculate_default_transform(
                    src_crs, target_crs, src.width, src.height, *src.bounds)

                # Copy metadata from the source and update for the target CRS and transform
                kwargs = src.meta.copy()
                # Ensure nodata value is handled correctly
                src_nodata = src.nodata
                # If source has no nodata, pick a default reasonable one for elevation float data
                # Using a large negative number typical for elevation nodata
                nodata_val = src_nodata if src_nodata is not None else -32767.0

                kwargs.update({
                    'crs': target_crs,
                    'transform': transform,
                    'width': width,
                    'height': height,
                    'nodata': nodata_val,
                    'dtype': 'float32' # Ensure output is float for elevation
                })

                print(f"Projected DEM size: {width}x{height}")
                print(f"Projected NoData value: {nodata_val}")

                # Open the destination file and reproject
                with rasterio.open(proj_dem_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1): # Iterate through bands (usually 1 for DEM)
                        rasterio.warp.reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src_crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=Resampling.bilinear # Bilinear is good for continuous data like elevation
                        )
            print("DEM reprojection complete.")
            if not os.path.exists(proj_dem_path) or os.path.getsize(proj_dem_path) == 0:
                 print(f"ERROR: Projected DEM file was not created or is empty after reprojection: {proj_dem_path}")
                 return None

        except MemoryError:
             print("ERROR: Ran out of memory during DEM reprojection.")
             print("Consider using a machine with more RAM or exploring tiled processing / VRT approaches for very large areas.")
             # Attempt cleanup
             if os.path.exists(proj_dem_path):
                 try: os.remove(proj_dem_path)
                 except OSError: pass
             return None
        except Exception as e:
            print(f"ERROR during DEM reprojection: {e}")
            # Attempt cleanup
            if os.path.exists(proj_dem_path):
                try: os.remove(proj_dem_path)
                except OSError: pass
            return None
    else:
        print(f"\nProjected DEM file already exists, skipping reprojection: {proj_dem_path}")
        # Basic check if existing file is valid raster
        try:
            with rasterio.open(proj_dem_path) as src:
                print(f"Existing projected DEM CRS: {src.crs}, Size: {src.width}x{src.height}")
                if str(src.crs).upper() != str(target_crs).upper():
                     print(f"WARNING: Existing projected DEM CRS ({src.crs}) does not match target CRS ({target_crs}).")
                     print("If this is incorrect, delete the file and run again.")
        except rasterio.RasterioIOError:
             print(f"ERROR: Existing projected DEM file is corrupted or not a valid raster: {proj_dem_path}")
             print("Please delete it and run the script again.")
             return None

    return proj_dem_path


# --- Main Execution ---

if __name__ == "__main__":
    print(f"--- Streamlined DEM Preparation for {AREA_NAME} ---")
    print(f"Area of Interest Bounds (Lon/Lat): {AREA_OF_INTEREST_BOUNDS}")
    print(f"Target CRS: {TARGET_CRS}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Projected DEM Output File: {PROJECTED_DEM_FILE}")

    # Step 1: Download and Reproject DEM for the specified area
    projected_dem_output_path = download_and_prepare_dem(
        bounds=AREA_OF_INTEREST_BOUNDS,
        product=DEM_PRODUCT,
        max_tiles=MAX_DOWNLOAD_TILES,
        target_crs=TARGET_CRS,
        raw_dem_path=RAW_DEM_FILE,
        proj_dem_path=PROJECTED_DEM_FILE
    )

    if projected_dem_output_path and os.path.exists(projected_dem_output_path):
        print(f"\n--- Success ---")
        print(f"Projected DEM successfully created at:")
        print(projected_dem_output_path)
    else:
        print("\n--- Failed ---")
        print("DEM preparation failed. Please check the logs above for errors.")
        sys.exit(1) # Indicate error exit

    print("\n--- Script Finished ---")
