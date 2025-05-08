#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlined script to calculate and combine terrain steepness and
land cover suitability for barefoot running potential within a defined radius.

Version 3.1 - Target Slope Feature:
- Adds a target-based slope suitability calculation.
- User can define a TARGET_SLOPE_DEGREES and SLOPE_PENALTY_TYPE ('quadratic' or 'linear').
- Generates additional maps for target slope suitability and the combined score using this method.
- Original "steeper is better" slope scoring and its combined map are preserved for comparison.

Version 3.0 + Basemap Plot:
- Generates the 4 combined raster+basemap plots.
- Adds a 5th plot showing ONLY the contextily basemap for the AOI.
- Uses user-provided verbose LAND_COVER_SUITABILITY dictionary.
- Added explicit zorder to imshow calls to ensure raster plots above basemap.
- Contextily basemap and raster transparency are enabled.
- Uses numpy.gradient for slope calculation.

Uses a projected Digital Elevation Model (DEM) and a UKCEH Land Cover Map (LCM).
Calculates slope and reclassifies LCM based on user suitability within the AOI.

Requires: rasterio, numpy, matplotlib, os, geopy, math, shapely, contextily
"""

import os
import sys
import rasterio
import rasterio.warp
import rasterio.mask
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from shapely.geometry import box, Point
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import math
import warnings # To suppress UserWarnings if needed
import traceback # For detailed error printing

# Attempt to import contextily
try:
    import contextily as cx
except ImportError:
    print("ERROR: Missing required library 'contextily'.")
    print("Please install it, e.g., using 'pip install contextily'")
    sys.exit(1)

# =============================================================================
# --- Configuration ---
# =============================================================================

# --- 1. Area of Interest Definition ---
#PLACE_NAME = "Burnley, UK"
#PLACE_NAME = "Edale, UK"
PLACE_NAME = "Ladybower Reservoir, UK"
#PLACE_NAME = "Sedbergh, UK"
#PLACE_NAME = "Harrogate, UK"
#PLACE_NAME = "Hawick, UK"
#PLACE_NAME = "Borrowdale, UK"
#PLACE_NAME = "Peak District National Park"
#PLACE_NAME = "Lake District National Park"
#PLACE_NAME = "Hope, UK"
RADIUS_KM = 10 # Kilometers

# --- 2. Input Data Paths ---
# Ensure these paths are correct for your system
PROJECTED_DEM_PATH = "/home/ajay/Python_Projects/steepest_trails_search/outputs_united_kingdom_dem/united_kingdom_dem_proj_27700.tif"
LCM_PATH = "/home/ajay/Python_Projects/steepest_trails_search/LCM_data_and_docs/data/gblcm2023_10m.tif"

# --- 3. Coordinate Reference System (CRS) ---
TARGET_CRS = "EPSG:27700" # OS National Grid (British National Grid)
SOURCE_CRS_GPS = "EPSG:4326" # WGS84 for geocoding

# --- 4. Land Cover Suitability & Class Info ---
# Suitability scores (0.0 to 1.0) for different land cover types
LAND_COVER_SUITABILITY = {
    # Codes based on LCM2021/2023 documentation (verify if needed)
    1: 1.0,  # Broadleaved Woodland
    2: 1.0,  # Coniferous Woodland
    3: 1.0,  # Arable
    4: 1.0,  # Improved Grassland
    5: 1.0,  # Neutral Grassland
    6: 1.0,  # Calcareous Grassland
    7: 1.0,  # Acid Grassland
    8: 0.0,  # Fen, Marsh and Swamp (generally unsuitable)
    9: 0.0,  # Heather (can be tough/woody)
    10: 0.0, # Heather Grassland (mix, can be tough)
    11: 0.0, # Bog (too wet/unstable)
    12: 0.0, # Inland Rock (unsuitable)
    13: 0.0, # Saltwater (unsuitable)
    14: 0.0, # Freshwater (unsuitable)
    15: 0.0, # Supralittoral Rock (unsuitable)
    16: 0.0, # Supralittoral Sediment (can be good if sand, but variable)
    17: 0.0, # Littoral Rock (unsuitable)
    18: 0.0, # Littoral Sediment (variable, avoid mud)
    19: 0.0, # Saltmarsh (unsuitable)
    20: 0.0, # Urban (unsuitable)
    21: 0.0, # Suburban (unsuitable)
}
DEFAULT_SUITABILITY_SCORE = 0.0 # Score for LCM classes not in the dictionary

# --- Class Names and Colors for Legend (using tab20 colormap) ---
cmap_tab20 = plt.get_cmap('tab20', 21)
LCM_CLASS_INFO = {
    # Code: (Name, Color)
    1: ('Broadleaved Woodland', cmap_tab20(0)), 2: ('Coniferous Woodland', cmap_tab20(1)),
    3: ('Arable', cmap_tab20(2)), 4: ('Improved Grassland', cmap_tab20(3)),
    5: ('Neutral Grassland', cmap_tab20(4)), 6: ('Calcareous Grassland', cmap_tab20(5)),
    7: ('Acid Grassland', cmap_tab20(6)), 8: ('Fen, Marsh & Swamp', cmap_tab20(7)),
    9: ('Heather', cmap_tab20(8)), 10: ('Heather Grassland', cmap_tab20(9)),
    11: ('Bog', cmap_tab20(10)), 12: ('Inland Rock', cmap_tab20(11)),
    13: ('Saltwater', cmap_tab20(12)), 14: ('Freshwater', cmap_tab20(13)),
    15: ('Supralittoral Rock', cmap_tab20(14)), 16: ('Supralittoral Sediment', cmap_tab20(15)),
    17: ('Littoral Rock', cmap_tab20(16)), 18: ('Littoral Sediment', cmap_tab20(17)),
    19: ('Saltmarsh', cmap_tab20(18)), 20: ('Urban', cmap_tab20(19)),
    21: ('Suburban', cmap_tab20(20)), 0: ('Nodata / Unclassified', (1, 1, 1, 0)) # Transparent white for nodata
}
LCM_NODATA_VAL = 0 # Nodata value in the LCM raster

# --- 5. Output Configuration ---
OUTPUT_BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Saves outputs in script's directory
MAX_SLOPE_CONSIDERED = 45.0 # Degrees. Upper limit for "steeper is better" normalization & target penalty range.

# --- 5a. NEW: Target Slope Configuration ---
TARGET_SLOPE_DEGREES = 15.0  # Desired optimal slope in degrees.
SLOPE_PENALTY_TYPE = 'linear' # 'quadratic' or 'linear'. Defines how deviation from target slope is penalized.

# --- 6. Visualization ---
VIS_CMAP_SCORE = 'viridis' # Colormap for combined score maps
VIS_CMAP_SLOPE = 'magma' # Colormap for raw slope map
VIS_CMAP_TARGET_SLOPE_SCORE = 'cividis' # Colormap for target slope suitability score map
CTX_PROVIDER = cx.providers.OpenStreetMap.Mapnik # Basemap provider
CTX_ALPHA = 0.7 # Transparency of the main raster overlay on basemap

# =============================================================================
# --- END OF Configuration ---
# =============================================================================

print(f"--- Barefoot Suitability Mapping v3.1 (Target Slope: {TARGET_SLOPE_DEGREES}° {SLOPE_PENALTY_TYPE}, Max Slope: {MAX_SLOPE_CONSIDERED}°) ---")

# --- Helper Functions ---

def get_bounding_box_wgs84(latitude, longitude, radius_km):
    """ Calculates bounding box (min_lon, min_lat, max_lon, max_lat) in WGS84."""
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180): raise ValueError("Invalid latitude or longitude.")
    if radius_km <= 0: raise ValueError("Radius must be positive.")
    earth_radius_km = 6371.0
    lat_delta_deg = math.degrees(radius_km / earth_radius_km)
    # Clamp latitude for cosine calculation to avoid issues near poles
    clamped_lat_rad = math.radians(max(-89.999, min(89.999, latitude)))
    # Handle potential division by zero if cos(clamped_lat_rad) is too small (near poles)
    lon_delta_deg = 180.0 if abs(math.cos(clamped_lat_rad)) < 1e-9 else math.degrees(radius_km / (earth_radius_km * math.cos(clamped_lat_rad)))
    min_lat = max(-90.0, latitude - lat_delta_deg)
    max_lat = min(90.0, latitude + lat_delta_deg)
    # Normalize longitudes to be within -180 to 180
    min_lon = (longitude - lon_delta_deg + 540) % 360 - 180
    max_lon = (longitude + lon_delta_deg + 540) % 360 - 180
    # Ensure min_lon is less than max_lon, especially if crossing antimeridian
    if min_lon > max_lon: min_lon, max_lon = max_lon, min_lon # This might need more robust handling for dateline crossing
    return (min_lon, min_lat, max_lon, max_lat)

def plot_raster_with_contextily(raster_path, output_png_path, title, cmap='viridis',
                                vmin=None, vmax=None, is_discrete=False,
                                class_info=None, nodata_val=None, ctx_alpha=0.8):
    """Visualizes a raster with contextily basemap, saving it to a PNG file."""
    print(f"  Visualizing: {title} -> {os.path.basename(output_png_path)}")
    try:
        with rasterio.open(raster_path) as src:
            if src.crs and TARGET_CRS and src.crs.to_string().upper() != TARGET_CRS.upper():
                 print(f"    WARNING: Raster CRS ({src.crs}) does not match TARGET_CRS ({TARGET_CRS}) for plotting!")

            data = src.read(1, masked=True) # Read data as a masked array
            
            # Explicitly apply nodata mask if nodata_val is provided
            # This ensures that pixels equal to nodata_val are also masked
            if nodata_val is not None:
                # Check for NaN nodata_val (common for float rasters not explicitly set)
                if np.isnan(nodata_val):
                    data.mask = data.mask | np.isnan(data.data)
                else:
                    data.mask = data.mask | (data.data == nodata_val)
            
            # Check if all data is masked
            if data.mask.all():
                print(f"    WARNING: All data is masked for '{title}'. Skipping plot.")
                plt.close('all') # Close any potentially open figures
                return

            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            plot_kwargs = {'alpha': ctx_alpha, 'zorder': 10} # Ensure raster is on top
            raster_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

            if is_discrete and class_info:
                codes = sorted(c for c in class_info.keys() if c != nodata_val) # Exclude nodata from discrete legend codes
                if not codes: # Handle case where only nodata exists or class_info is empty
                    print(f"    WARNING: No valid discrete classes to plot for '{title}'. Plotting as continuous or skipping.")
                    if data.count() == 0: # If no unmasked data points
                         plt.close(fig)
                         return
                    img = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=raster_extent, **plot_kwargs)
                else:
                    colors = [class_info[code][1] for code in codes]
                    names = [class_info[code][0] for code in codes]
                    
                    # Create a colormap and norm for discrete classes
                    # Add nodata representation if it was in class_info and had a color
                    all_codes_for_norm = sorted(class_info.keys())
                    all_colors_for_norm = [class_info[c][1] for c in all_codes_for_norm]
                    if nodata_val is not None and nodata_val in all_codes_for_norm:
                        nodata_idx_in_all = all_codes_for_norm.index(nodata_val)
                        all_colors_for_norm[nodata_idx_in_all] = (1,1,1,0) # Transparent for nodata

                    discrete_cmap = ListedColormap(all_colors_for_norm)
                    bounds = [c - 0.5 for c in all_codes_for_norm] + [all_codes_for_norm[-1] + 0.5]
                    norm = BoundaryNorm(bounds, discrete_cmap.N)
                    
                    img = ax.imshow(data, cmap=discrete_cmap, norm=norm, extent=raster_extent, **plot_kwargs)
                    legend_patches = [Patch(facecolor=class_info[code][1], edgecolor='black', label=f'{code}: {class_info[code][0]}')
                                      for code in codes] # Legend only for valid, non-nodata codes
                    ax.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="LCM Classes", fontsize='small')
            else: # Continuous data
                img = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=raster_extent, **plot_kwargs)
                fig.colorbar(img, ax=ax, label='Score/Value', shrink=0.7)

            # Add Contextily Basemap
            try:
                print(f"    Adding Contextily basemap (CRS: {src.crs.to_string() if src.crs else 'Unknown'})...")
                cx.add_basemap(ax, crs=src.crs.to_string() if src.crs else TARGET_CRS, source=CTX_PROVIDER, zoom='auto') # Fallback to TARGET_CRS if src.crs is None
                print("    Basemap added successfully.")
            except Exception as ctx_e:
                print(f"    ERROR adding contextily basemap: {ctx_e}")
                print(f"    Check network connection and if provider {CTX_PROVIDER} is available.")

            ax.set_title(title, fontsize=14)
            ax.set_axis_off()
            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
            plt.savefig(output_png_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"    Visualization saved: {output_png_path}")

    except FileNotFoundError:
         print(f"    Error: Raster file not found for visualization: {raster_path}")
    except Exception as e:
        print(f"    Error visualizing {raster_path}: {e}")
        traceback.print_exc()
        plt.close('all') # Ensure figure is closed on error

def plot_contextily_basemap_standalone(target_bounds_proj, target_crs_str, output_png_path, title):
    """Plots ONLY the contextily basemap for the given bounds (in target_crs_str) and CRS."""
    print(f"  Visualizing Basemap Only: {title} -> {os.path.basename(output_png_path)}")
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        minx, miny, maxx, maxy = target_bounds_proj
        if not all(np.isfinite([minx, miny, maxx, maxy])):
             print("    ERROR: Invalid target bounds for basemap plot (non-finite values).")
             plt.close(fig); return
        ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)

        print(f"    Adding Contextily basemap (CRS: {target_crs_str})...")
        cx.add_basemap(ax, crs=target_crs_str, source=CTX_PROVIDER, zoom='auto')
        print("    Basemap added successfully.")

        ax.set_title(title, fontsize=14); ax.set_axis_off(); plt.tight_layout()
        plt.savefig(output_png_path, dpi=200, bbox_inches='tight'); plt.close(fig)
        print(f"    Basemap visualization saved: {output_png_path}")
    except Exception as e:
        print(f"    Error visualizing basemap: {e}")
        traceback.print_exc()
        plt.close('all')

def calculate_slope_degrees(dem_data, x_res, y_res, dem_nodata_val):
    """Calculates slope in degrees using numpy.gradient, handling nodata from DEM."""
    dem_data_float = dem_data.astype(np.float32)
    nodata_mask = np.isnan(dem_data_float) # Initialize mask with NaNs
    if dem_nodata_val is not None: # If DEM has a specific nodata value, include it in the mask
        if np.isnan(dem_nodata_val): # Handle NaN nodata value
             nodata_mask = nodata_mask | np.isnan(dem_data_float)
        else:
             nodata_mask = nodata_mask | np.isclose(dem_data_float, dem_nodata_val)
    
    dem_data_float[nodata_mask] = np.nan # Set all identified nodata pixels to NaN for gradient calculation

    gy, gx = np.gradient(dem_data_float, y_res, x_res) # y_res is pixel height, x_res is pixel width
    
    # Slope calculation: arctan(sqrt((dz/dx)^2 + (dz/dy)^2))
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    slope_deg = np.degrees(slope_rad)
    
    # Output nodata for slope raster will be -9999.0
    # Apply this where the original DEM had nodata (or where gradient resulted in NaN, though less likely with NaN input)
    slope_deg[nodata_mask | np.isnan(slope_deg)] = -9999.0
    return slope_deg.astype(np.float32)

# --- NEW: Function to calculate target-based slope score ---
def calculate_target_slope_score(slope_degrees_data, target_slope_deg, max_considered_slope_deg,
                                 penalty_type, nodata_val_slope=-9999.0, output_nodata_val=-1.0):
    """
    Calculates a slope suitability score (0-1) based on deviation from a target slope.
    Score is 1 at target_slope_deg, decreasing based on penalty_type.
    Slopes are normalized using max_considered_slope_deg.
    """
    print(f"    Calculating target slope score (Target: {target_slope_deg}°, Max Considered: {max_considered_slope_deg}°, Penalty: {penalty_type})")
    
    if max_considered_slope_deg <= 0:
        print("    ERROR in calculate_target_slope_score: max_considered_slope_deg must be positive.")
        return np.full(slope_degrees_data.shape, output_nodata_val, dtype=np.float32)

    # Create a masked array to handle nodata in slope_degrees_data
    # Also mask negative slopes as they are invalid.
    slope_masked = np.ma.array(slope_degrees_data,
                               mask=((slope_degrees_data == nodata_val_slope) | \
                                     np.isnan(slope_degrees_data) | \
                                     (slope_degrees_data < 0)),
                               fill_value=output_nodata_val, # Value to fill masked areas if .filled() is used
                               dtype=np.float32)

    # Normalize actual slope: clip to [0, max_considered_slope_deg] then divide by max_considered_slope_deg
    # This ensures normalized slope is within [0, 1] for valid slopes.
    actual_slope_normalized = np.ma.clip(slope_masked, 0, max_considered_slope_deg) / max_considered_slope_deg

    # Normalize target slope: clip to [0, max_considered_slope_deg] then divide by max_considered_slope_deg
    # Target slope is a scalar.
    target_slope_normalized = np.clip(float(target_slope_deg), 0, max_considered_slope_deg) / max_considered_slope_deg

    difference = actual_slope_normalized - target_slope_normalized # Difference is in range [-1, 1]

    if penalty_type.lower() == 'quadratic':
        # Score = 1.0 - (difference^2). Since difference is [-1, 1], difference^2 is [0, 1]. Score is [0, 1].
        score = 1.0 - np.ma.power(difference, 2)
    elif penalty_type.lower() == 'linear':
        # Score = 1.0 - |difference|. Since |difference| is [0, 1], score is [0, 1].
        score = 1.0 - np.ma.abs(difference)
    else:
        print(f"    WARNING: Unknown penalty_type '{penalty_type}'. Defaulting to quadratic.")
        score = 1.0 - np.ma.power(difference, 2) # Default to quadratic

    # Ensure score is strictly within [0, 1], though theoretically it should be.
    score_clipped = np.ma.clip(score, 0, 1)
    
    # Fill masked values (where input slope was nodata) with output_nodata_val
    return score_clipped.filled(output_nodata_val).astype(np.float32)
# --- END NEW FUNCTION ---

# --- Dynamic Path Generation ---
place_slug = PLACE_NAME.replace(' ', '_').replace(',', '').lower()
# Updated subdirectory name to be more descriptive and include version/feature
OUTPUT_SUBDIR_NAME = f"outputs_{place_slug}_r{RADIUS_KM}km_barefoot_v3.1_ts{TARGET_SLOPE_DEGREES}{SLOPE_PENALTY_TYPE[0]}"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, OUTPUT_SUBDIR_NAME)
print(f"Creating output directory: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
if not os.path.isdir(OUTPUT_DIR):
    print(f"FATAL ERROR: Failed to create output directory: {OUTPUT_DIR}"); sys.exit(1)

# --- Define Output Paths ---
# Aligned base rasters
ALIGNED_DEM_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_aligned_dem.tif")
ALIGNED_LCM_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_aligned_lcm.tif")

# Slope and Suitability Rasters
SLOPE_RASTER_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_slope_degrees.tif") # Raw slope in degrees
SUITABILITY_RASTER_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_land_cover_suitability.tif") # Land cover suitability score

# Original combined score (steeper is better slope)
COMBINED_SCORE_STEEP_BETTER_RASTER_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_combined_score_steep_better.tif")

# NEW Paths for Target Slope Method
TARGET_SLOPE_SCORE_RASTER_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_slope_score_target_{TARGET_SLOPE_DEGREES}deg_{SLOPE_PENALTY_TYPE}.tif")
COMBINED_SCORE_TARGET_SLOPE_RASTER_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_combined_score_target_slope_{TARGET_SLOPE_DEGREES}deg_{SLOPE_PENALTY_TYPE}.tif")

# Visualization Paths (PNGs)
VIS_LCM_CLASS_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_map_lcm_classes.png")
VIS_SLOPE_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_map_slope_degrees.png") # Raw slope map
VIS_SUITABILITY_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_map_land_cover_suitability.png")
VIS_BASEMAP_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_map_basemap_only.png")

# Original combined score visualization
VIS_COMBINED_SCORE_STEEP_BETTER_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_map_combined_score_steep_better.png")

# NEW Visualization Paths for Target Slope Method
VIS_TARGET_SLOPE_SCORE_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_map_slope_score_target.png")
VIS_COMBINED_SCORE_TARGET_SLOPE_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_map_combined_score_target_slope.png")

# --- Main Processing Steps ---
try:
    # --- 1. Geocode Place Name and Define AOI Bounds ---
    print(f"\n--- Step 1: Defining Area of Interest (AOI) ---")
    print(f"  Geocoding '{PLACE_NAME}'...")
    geolocator = Nominatim(user_agent=f"barefoot_suitability_mapper_v3.1_{place_slug}")
    center_lat, center_lon = None, None
    try:
        location = geolocator.geocode(PLACE_NAME, timeout=20) # Increased timeout
        if not location:
            print(f"FATAL ERROR: Could not geocode '{PLACE_NAME}'. Check place name and network."); sys.exit(1)
        center_lat, center_lon = location.latitude, location.longitude
        print(f"  Coordinates found (WGS84): Lat={center_lat:.4f}, Lon={center_lon:.4f}")
        # Get AOI bounds in WGS84
        bounds_wgs84 = get_bounding_box_wgs84(center_lat, center_lon, RADIUS_KM)
        print(f"  Calculated Bounding Box (WGS84 {SOURCE_CRS_GPS}): {bounds_wgs84}")
        # Transform AOI bounds to the target CRS
        target_bounds_proj = rasterio.warp.transform_bounds(SOURCE_CRS_GPS, TARGET_CRS, *bounds_wgs84)
        print(f"  Transformed Bounding Box ({TARGET_CRS}): {target_bounds_proj}")
        crop_box_geom_proj = box(*target_bounds_proj) # Shapely geometry for intersection checks
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"FATAL ERROR: Geocoding failed for '{PLACE_NAME}': {e}. Check network or try a different place name."); sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR during AOI definition: {e}"); traceback.print_exc(); sys.exit(1)

    # --- 1b. Plot Standalone Basemap for AOI ---
    print(f"\n--- Step 1b: Visualizing Standalone Basemap for AOI ---")
    plot_contextily_basemap_standalone(
        target_bounds_proj, TARGET_CRS, VIS_BASEMAP_PATH,
        f"Basemap Only - {PLACE_NAME} ({RADIUS_KM}km radius)"
    )

    # --- 2. Open Input Rasters & Check Intersection ---
    print("\n--- Step 2: Opening Input Rasters & Checking AOI Intersection ---")
    # Open DEM and LCM sources to get their properties
    with rasterio.open(PROJECTED_DEM_PATH) as dem_src, rasterio.open(LCM_PATH) as lcm_src:
        print(f"  DEM: {PROJECTED_DEM_PATH} (CRS: {dem_src.crs}, Resolution: {dem_src.res})")
        print(f"  LCM: {LCM_PATH} (CRS: {lcm_src.crs}, Resolution: {lcm_src.res}, Bands: {lcm_src.count})")

        # Validate CRS
        if not dem_src.crs or dem_src.crs.to_string().upper() != TARGET_CRS.upper():
            print(f"FATAL ERROR: DEM CRS ({dem_src.crs}) must be {TARGET_CRS}. Please reproject DEM first."); sys.exit(1)
        if not lcm_src.crs:
            print(f"FATAL ERROR: LCM CRS ({lcm_src.crs}) is undefined. Cannot proceed."); sys.exit(1)
        
        # Check if AOI intersects with DEM
        dem_bounds_geom = box(*dem_src.bounds)
        if not crop_box_geom_proj.intersects(dem_bounds_geom):
            print(f"FATAL ERROR: AOI bounds {target_bounds_proj} do not intersect with DEM bounds {dem_src.bounds}."); sys.exit(1)
        else: print(f"  AOI intersects with DEM bounds.")

        # Check if AOI intersects with LCM (transforming AOI to LCM's CRS if different)
        if lcm_src.crs.to_string().upper() != TARGET_CRS.upper():
            print(f"  Warning: LCM CRS ({lcm_src.crs}) differs from TARGET_CRS ({TARGET_CRS}). Transforming AOI for LCM intersection check.")
            try:
                aoi_bounds_lcm_crs = rasterio.warp.transform_bounds(TARGET_CRS, lcm_src.crs, *target_bounds_proj)
                crop_box_geom_lcm_crs = box(*aoi_bounds_lcm_crs)
            except Exception as e:
                print(f"    Warning: Could not transform AOI bounds to LCM CRS for check: {e}")
                crop_box_geom_lcm_crs = crop_box_geom_proj # Fallback, less accurate if CRSs differ significantly
        else:
            crop_box_geom_lcm_crs = crop_box_geom_proj # AOI is already in LCM's CRS

        lcm_bounds_geom = box(*lcm_src.bounds)
        if not crop_box_geom_lcm_crs.intersects(lcm_bounds_geom):
            print(f"FATAL ERROR: AOI bounds (in LCM CRS) do not intersect with LCM bounds {lcm_src.bounds}."); sys.exit(1)
        else: print(f"  AOI intersects with LCM bounds.")

        # --- Step 3: Define Target Profile & Align Rasters ---
        # Align rasters to the AOI extent (target_bounds_proj) at the LCM's resolution.
        print(f"\n--- Step 3: Aligning Rasters to Target Grid within AOI ---")
        print(f"  Targeting LCM resolution for alignment: X={lcm_src.res[0]}, Y={lcm_src.res[1]}")

        # Calculate dimensions and transform for the aligned rasters based on AOI and LCM resolution
        # AOI bounds: target_bounds_proj = (minx, miny, maxx, maxy)
        # LCM resolution: lcm_src.res = (x_res, y_res) where y_res is often negative
        out_width = int(math.ceil((target_bounds_proj[2] - target_bounds_proj[0]) / lcm_src.res[0]))
        out_height = int(math.ceil((target_bounds_proj[3] - target_bounds_proj[1]) / abs(lcm_src.res[1])))

        # Output transform: Affine(x_resolution, 0, x_min_coord (left), 0, -y_resolution (negative for north-up), y_max_coord (top))
        out_transform = rasterio.Affine(lcm_src.res[0], 0.0, target_bounds_proj[0],
                                        0.0, -abs(lcm_src.res[1]), target_bounds_proj[3])
        
        # Base profile for aligned rasters
        align_profile_base = {
            'driver': 'GTiff',
            'crs': TARGET_CRS,
            'transform': out_transform,
            'width': out_width,
            'height': out_height,
            'count': 1 # All our processed rasters are single-band
        }

    # --- 3a. Align DEM ---
    align_profile_dem = align_profile_base.copy()
    align_profile_dem.update({'dtype': 'float32', 'nodata': -9999.0}) # DEM nodata
    print(f"  Aligning DEM to: {ALIGNED_DEM_PATH} (Dimensions: {out_width}x{out_height})")
    with rasterio.open(PROJECTED_DEM_PATH) as dem_src, \
         rasterio.open(ALIGNED_DEM_PATH, 'w', **align_profile_dem) as dst:
        rasterio.warp.reproject(
            source=rasterio.band(dem_src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=dem_src.transform,
            src_crs=dem_src.crs,
            dst_transform=align_profile_dem['transform'],
            dst_crs=align_profile_dem['crs'],
            resampling=Resampling.bilinear, # Bilinear for continuous DEM data
            dst_nodata=align_profile_dem['nodata']
        )
    print("  DEM Alignment complete.")

    # --- 3b. Align LCM ---
    align_profile_lcm = align_profile_base.copy()
    align_profile_lcm.update({'dtype': 'uint8', 'nodata': LCM_NODATA_VAL}) # LCM uses uint8 and specific nodata
    print(f"  Aligning LCM to: {ALIGNED_LCM_PATH} (Dimensions: {out_width}x{out_height})")
    with rasterio.open(LCM_PATH) as lcm_src, \
         rasterio.open(ALIGNED_LCM_PATH, 'w', **align_profile_lcm) as dst:
        rasterio.warp.reproject(
            source=rasterio.band(lcm_src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=lcm_src.transform,
            src_crs=lcm_src.crs,
            dst_transform=align_profile_lcm['transform'],
            dst_crs=align_profile_lcm['crs'],
            resampling=Resampling.nearest, # Nearest neighbor for categorical LCM data
            dst_nodata=align_profile_lcm['nodata']
        )
    print("  LCM Alignment complete.")

    # --- 3c. Visualize Raw LCM Classes (Aligned) ---
    print(f"\n--- Step 3c: Visualizing Aligned Land Cover Classes ---")
    plot_raster_with_contextily(
        ALIGNED_LCM_PATH, VIS_LCM_CLASS_PATH, f"Land Cover Classes - {PLACE_NAME} (Aligned)",
        is_discrete=True, class_info=LCM_CLASS_INFO, nodata_val=LCM_NODATA_VAL, ctx_alpha=CTX_ALPHA
    )

    # --- 4. Calculate Slope (Raw Degrees) ---
    print(f"\n--- Step 4: Calculating Slope (Degrees) from Aligned DEM ---")
    slope_profile = align_profile_dem.copy() # Use DEM's profile as base
    slope_profile.update({'nodata': -9999.0, 'dtype': 'float32'}) # Slope output nodata
    try:
        with rasterio.open(ALIGNED_DEM_PATH) as dem_aligned_src:
            dem_data = dem_aligned_src.read(1)
            x_res, y_res = dem_aligned_src.res # Get resolution from aligned DEM
            dem_nodata_val = dem_aligned_src.nodata
            
            slope_deg_data = calculate_slope_degrees(dem_data, x_res, abs(y_res), dem_nodata_val) # Pass abs(y_res)
            
            with rasterio.open(SLOPE_RASTER_PATH, 'w', **slope_profile) as dst:
                dst.write(slope_deg_data, 1)
        print(f"  Slope (degrees) raster saved to: {SLOPE_RASTER_PATH}")
        plot_raster_with_contextily(
            SLOPE_RASTER_PATH, VIS_SLOPE_PATH, f"Slope (Degrees) - {PLACE_NAME}",
            cmap=VIS_CMAP_SLOPE, ctx_alpha=CTX_ALPHA, vmin=0, vmax=MAX_SLOPE_CONSIDERED,
            nodata_val=slope_profile['nodata']
        )
    except FileNotFoundError: print(f"  ERROR: Aligned DEM file not found: {ALIGNED_DEM_PATH}")
    except Exception as e: print(f"  ERROR during slope calculation: {e}"); traceback.print_exc()

    # --- 4b. NEW: Calculate Target Slope Suitability Score ---
    print(f"\n--- Step 4b: Calculating Target Slope Suitability Score ---")
    target_slope_score_profile = align_profile_dem.copy() # Base on DEM profile
    target_slope_score_profile.update({'dtype': 'float32', 'nodata': -1.0}) # Score nodata
    try:
        # Ensure raw slope raster exists from Step 4
        if not os.path.exists(SLOPE_RASTER_PATH):
            raise FileNotFoundError(f"Raw slope raster not found: {SLOPE_RASTER_PATH}")

        with rasterio.open(SLOPE_RASTER_PATH) as slope_src:
            slope_degrees_data_for_target = slope_src.read(1)
            # Nodata for the input slope raster is -9999.0 (as defined in slope_profile)
            slope_nodata_for_target_func = slope_src.nodata if slope_src.nodata is not None else -9999.0

            target_slope_score_data = calculate_target_slope_score(
                slope_degrees_data_for_target,
                TARGET_SLOPE_DEGREES,
                MAX_SLOPE_CONSIDERED,
                SLOPE_PENALTY_TYPE,
                nodata_val_slope=slope_nodata_for_target_func, # from the slope_degrees.tif
                output_nodata_val=target_slope_score_profile['nodata'] # for the output score raster
            )
            with rasterio.open(TARGET_SLOPE_SCORE_RASTER_PATH, 'w', **target_slope_score_profile) as dst:
                dst.write(target_slope_score_data, 1)
            print(f"  Target slope suitability score raster saved to: {TARGET_SLOPE_SCORE_RASTER_PATH}")

        plot_raster_with_contextily(
            TARGET_SLOPE_SCORE_RASTER_PATH, VIS_TARGET_SLOPE_SCORE_PATH,
            f"Target Slope Suitability (Target: {TARGET_SLOPE_DEGREES}°, Penalty: {SLOPE_PENALTY_TYPE}) - {PLACE_NAME}",
            cmap=VIS_CMAP_TARGET_SLOPE_SCORE, vmin=0, vmax=1, ctx_alpha=CTX_ALPHA,
            nodata_val=target_slope_score_profile['nodata']
        )
    except FileNotFoundError as e: print(f"  ERROR: Prerequisite file not found for target slope score: {e}")
    except Exception as e: print(f"  ERROR during target slope score calculation: {e}"); traceback.print_exc()

    # --- 5. Reclassify Land Cover for Suitability ---
    print(f"\n--- Step 5: Reclassifying Land Cover for Suitability ---")
    lc_suitability_profile = align_profile_lcm.copy() # Base on LCM profile
    lc_suitability_profile.update({'dtype': 'float32', 'nodata': -1.0}) # Score nodata
    try:
        # Ensure aligned LCM raster exists
        if not os.path.exists(ALIGNED_LCM_PATH):
            raise FileNotFoundError(f"Aligned LCM raster not found: {ALIGNED_LCM_PATH}")

        with rasterio.open(ALIGNED_LCM_PATH) as lcm_aligned_src:
            lcm_data = lcm_aligned_src.read(1)
            nodata_lcm_val = lcm_aligned_src.nodata if lcm_aligned_src.nodata is not None else LCM_NODATA_VAL
            
            # Initialize suitability_data with the nodata value
            suitability_data = np.full(lcm_data.shape, lc_suitability_profile['nodata'], dtype=np.float32)
            
            # Create a mask for valid (non-nodata) LCM pixels
            valid_lcm_mask = (lcm_data != nodata_lcm_val) & (~np.isnan(lcm_data))
            
            # Apply suitability scores from the dictionary
            for code, score in LAND_COVER_SUITABILITY.items():
                suitability_data[(lcm_data == code) & valid_lcm_mask] = score
            
            # Handle codes present in LCM but not in LAND_COVER_SUITABILITY dict (assign default score)
            present_lcm_codes = np.unique(lcm_data[valid_lcm_mask])
            unmapped_codes = set(present_lcm_codes) - set(LAND_COVER_SUITABILITY.keys())
            if unmapped_codes:
                print(f"    Codes in LCM not in suitability dict (will get default score {DEFAULT_SUITABILITY_SCORE}): {sorted(list(unmapped_codes))}")
                for code in unmapped_codes:
                     suitability_data[(lcm_data == code) & valid_lcm_mask] = DEFAULT_SUITABILITY_SCORE
            
            # Ensure areas that were nodata in original LCM remain nodata in suitability raster
            suitability_data[~valid_lcm_mask] = lc_suitability_profile['nodata']

        with rasterio.open(SUITABILITY_RASTER_PATH, 'w', **lc_suitability_profile) as dst:
            dst.write(suitability_data, 1)
        print(f"  Land cover suitability raster saved to: {SUITABILITY_RASTER_PATH}")
        plot_raster_with_contextily(
            SUITABILITY_RASTER_PATH, VIS_SUITABILITY_PATH, f"Land Cover Suitability Score - {PLACE_NAME}",
            cmap='YlGn', vmin=0, vmax=1, ctx_alpha=CTX_ALPHA,
            nodata_val=lc_suitability_profile['nodata']
        )
    except FileNotFoundError as e: print(f"  ERROR: Prerequisite file not found for land cover reclassification: {e}")
    except Exception as e: print(f"  ERROR during land cover reclassification: {e}"); traceback.print_exc()

    # --- 6. Combine Scores (Two Methods) ---
    print("\n--- Step 6: Combining Scores ---")
    # Profile for combined score rasters (float, nodata -1.0)
    combined_score_profile = align_profile_base.copy() # Use the common alignment profile
    combined_score_profile.update({'dtype': 'float32', 'nodata': -1.0, 'count': 1})

    # --- 6a. Original Method: (Normalized "Steeper is Better" Slope) * Land Cover Suitability ---
    print("  --- Method 1: Combining 'Steeper is Better' Slope Score with Land Cover Suitability ---")
    if not os.path.exists(SLOPE_RASTER_PATH) or not os.path.exists(SUITABILITY_RASTER_PATH):
        print("    Skipping 'Steeper is Better' combination: Input raw slope or land cover suitability raster missing.")
    else:
        try:
            with rasterio.open(SLOPE_RASTER_PATH) as slope_src, \
                 rasterio.open(SUITABILITY_RASTER_PATH) as lc_suit_src:

                # Read data as masked arrays to handle nodata automatically
                slope_data_raw_masked = slope_src.read(1, masked=True)
                lc_suitability_masked = lc_suit_src.read(1, masked=True)

                # Normalize raw slope (steeper is better, capped at MAX_SLOPE_CONSIDERED)
                # Result is 0-1, or masked where input was nodata/invalid
                normalized_slope_steep_better = np.ma.clip(slope_data_raw_masked, 0, MAX_SLOPE_CONSIDERED) / MAX_SLOPE_CONSIDERED
                
                # Combine scores (multiplication handles masked values correctly)
                combined_score_steep_better_masked = normalized_slope_steep_better * lc_suitability_masked
                
                # Fill nodata areas and convert to standard numpy array
                combined_score_steep_better_filled = np.ma.filled(combined_score_steep_better_masked, combined_score_profile['nodata'])

            with rasterio.open(COMBINED_SCORE_STEEP_BETTER_RASTER_PATH, 'w', **combined_score_profile) as dst:
                dst.write(combined_score_steep_better_filled.astype(np.float32), 1)
            print(f"    Combined Score (Steeper is Better) raster saved to: {COMBINED_SCORE_STEEP_BETTER_RASTER_PATH}")
        except Exception as e: print(f"    ERROR during 'Steeper is Better' score combination: {e}"); traceback.print_exc()

    # --- 6b. NEW Method: (Target Slope Score) * Land Cover Suitability ---
    print("  --- Method 2: Combining Target Slope Score with Land Cover Suitability ---")
    if not os.path.exists(TARGET_SLOPE_SCORE_RASTER_PATH) or not os.path.exists(SUITABILITY_RASTER_PATH):
        print("    Skipping 'Target Slope' combination: Target slope score or land cover suitability raster missing.")
    else:
        try:
            with rasterio.open(TARGET_SLOPE_SCORE_RASTER_PATH) as target_slope_score_src, \
                 rasterio.open(SUITABILITY_RASTER_PATH) as lc_suit_src:

                target_slope_score_masked = target_slope_score_src.read(1, masked=True)
                lc_suitability_masked = lc_suit_src.read(1, masked=True) # Re-read or use from above if scope allows
                
                combined_score_target_slope_masked = target_slope_score_masked * lc_suitability_masked
                combined_score_target_slope_filled = np.ma.filled(combined_score_target_slope_masked, combined_score_profile['nodata'])

            with rasterio.open(COMBINED_SCORE_TARGET_SLOPE_RASTER_PATH, 'w', **combined_score_profile) as dst:
                dst.write(combined_score_target_slope_filled.astype(np.float32), 1)
            print(f"    Combined Score (Target Slope Method) raster saved to: {COMBINED_SCORE_TARGET_SLOPE_RASTER_PATH}")
        except Exception as e: print(f"    ERROR during 'Target Slope' score combination: {e}"); traceback.print_exc()


    # --- 7. Visualize Final Combined Scores ---
    print("\n--- Step 7: Visualizing Final Combined Scores ---")
    # 7a. Visualize original combined score (Steeper is Better)
    if os.path.exists(COMBINED_SCORE_STEEP_BETTER_RASTER_PATH):
        plot_raster_with_contextily(
            COMBINED_SCORE_STEEP_BETTER_RASTER_PATH, VIS_COMBINED_SCORE_STEEP_BETTER_PATH,
            f"Combined Score (Steeper is Better) - {PLACE_NAME}",
            cmap=VIS_CMAP_SCORE, vmin=0, vmax=1, ctx_alpha=CTX_ALPHA,
            nodata_val=combined_score_profile['nodata'] # -1.0
        )
    else:
        print(f"  Skipping visualization: {COMBINED_SCORE_STEEP_BETTER_RASTER_PATH} not found.")

    # 7b. Visualize new target slope combined score
    if os.path.exists(COMBINED_SCORE_TARGET_SLOPE_RASTER_PATH):
        plot_raster_with_contextily(
            COMBINED_SCORE_TARGET_SLOPE_RASTER_PATH, VIS_COMBINED_SCORE_TARGET_SLOPE_PATH,
            f"Combined Score (Target Slope: {TARGET_SLOPE_DEGREES}°, {SLOPE_PENALTY_TYPE}) - {PLACE_NAME}",
            cmap=VIS_CMAP_SCORE, vmin=0, vmax=1, ctx_alpha=CTX_ALPHA,
            nodata_val=combined_score_profile['nodata'] # -1.0
        )
    else:
        print(f"  Skipping visualization: {COMBINED_SCORE_TARGET_SLOPE_RASTER_PATH} not found.")


except rasterio.RasterioIOError as e:
    print(f"\nFATAL ERROR: Raster I/O Error: {e}")
    print(f"  Please check paths and file integrity, especially for: DEM='{PROJECTED_DEM_PATH}', LCM='{LCM_PATH}'")
    traceback.print_exc(); sys.exit(1)
except ImportError as e: # Should be caught at the top, but as a fallback
     print(f"\nFATAL ERROR: Missing required library: {e}"); sys.exit(1)
except Exception as e:
    print(f"\nAn unexpected FATAL error occurred in the main script body: {e}")
    traceback.print_exc(); sys.exit(1)
finally:
    plt.close('all') # Ensure all matplotlib figures are closed

print("\n--- Workflow Finished ---")
print(f"Outputs saved in: {OUTPUT_DIR}")

