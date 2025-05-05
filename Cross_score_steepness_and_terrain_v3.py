#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlined script to calculate and combine terrain steepness and
land cover suitability for barefoot running potential within a defined radius.

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
#PLACE_NAME = "Burnley, UK" # <- UPDATE IF NEEDED
#PLACE_NAME = "Edale, UK" # <- UPDATE IF NEEDED
#PLACE_NAME = "Sedbergh, UK" # <- UPDATE IF NEEDED
#PLACE_NAME = "Harrogate, UK" # <- UPDATE IF NEEDED
#PLACE_NAME = "Hawick, UK" # <- UPDATE IF NEEDED
#PLACE_NAME = "Borrowdale, UK" # <- UPDATE IF NEEDED
#PLACE_NAME = "Peak District National Park" # <- Define center place name
PLACE_NAME = "Lake District National Park" # <- Define center place name
RADIUS_KM = 50 # <- UPDATE IF NEEDED

# --- 2. Input Data Paths ---
PROJECTED_DEM_PATH = "/home/ajay/Python_Projects/steepest_trails_search/outputs_united_kingdom_dem/united_kingdom_dem_proj_27700.tif" # Keep user path
LCM_PATH = "/home/ajay/Python_Projects/steepest_trails_search/LCM_data_and_docs/data/gblcm2023_10m.tif" # Keep user path

# --- 3. Coordinate Reference System (CRS) ---
TARGET_CRS = "EPSG:27700" # OS National Grid
SOURCE_CRS_GPS = "EPSG:4326" # WGS84 for geocoding

# --- 4. Land Cover Suitability & Class Info ---
# --- User-provided suitability dictionary ---

"""
LAND_COVER_SUITABILITY = {
    # Codes based on LCM2021/2023 documentation (verify if needed)
    1: 0.7,  # Broadleaved Woodland (leaf litter good, roots/branches caution)
    2: 0.7,  # Coniferous Woodland (needles can be sharp)
    3: 0.2,  # Arable (often uneven, potentially sharp stubble)
    4: 0.9,  # Improved Grassland (generally good, watch for stones/thistles)
    5: 1.0,  # Neutral Grassland (often excellent)
    6: 1.0,  # Calcareous Grassland (often excellent, maybe flinty)
    7: 0.7,  # Acid Grassland (good, can be tussocky/boggy patches)
    8: 0.4,  # Fen, Marsh and Swamp (too wet/uneven)
    9: 0.7,  # Heather (can be nice, but woody stems can be tough)
    10: 0.8, # Heather Grassland (mix of heather and grass)
    11: 0.3, # Bog (generally too wet/unstable)
    12: 0.0, # Inland Rock (unsuitable)
    13: 0.0, # Saltwater (unsuitable)
    14: 0.0, # Freshwater (unsuitable)
    15: 0.0, # Supralittoral Rock (coastal rock above high tide - unsuitable)
    16: 0.9, # Supralittoral Sediment (sand dunes/shingle above high tide - excellent if sand)
    17: 0.0, # Littoral Rock (coastal rock intertidal - unsuitable)
    18: 0.8, # Littoral Sediment (sand/mud intertidal - good if sand, avoid mud)
    19: 0.4, # Saltmarsh (often muddy, uneven, creeks)
    20: 0.0, # Urban (unsuitable)
    21: 0.0, # Suburban (unsuitable)
} # Keep user dictionary
"""
LAND_COVER_SUITABILITY = {
    # Codes based on LCM2021/2023 documentation (verify if needed)
    1: 1.0,  # Broadleaved Woodland (leaf litter good, roots/branches caution)
    2: 1.0,  # Coniferous Woodland (needles can be sharp)
    3: 0.0,  # Arable (often uneven, potentially sharp stubble)
    4: 0.0,  # Improved Grassland (generally good, watch for stones/thistles)
    5: 0.0,  # Neutral Grassland (often excellent)
    6: 0.0,  # Calcareous Grassland (often excellent, maybe flinty)
    7: 0.0,  # Acid Grassland (good, can be tussocky/boggy patches)
    8: 0.0,  # Fen, Marsh and Swamp (too wet/uneven)
    9: 0.0,  # Heather (can be nice, but woody stems can be tough)
    10: 0.0, # Heather Grassland (mix of heather and grass)
    11: 0.0, # Bog (generally too wet/unstable)
    12: 0.0, # Inland Rock (unsuitable)
    13: 0.0, # Saltwater (unsuitable)
    14: 0.0, # Freshwater (unsuitable)
    15: 0.0, # Supralittoral Rock (coastal rock above high tide - unsuitable)
    16: 0.0, # Supralittoral Sediment (sand dunes/shingle above high tide - excellent if sand)
    17: 0.0, # Littoral Rock (coastal rock intertidal - unsuitable)
    18: 0.0, # Littoral Sediment (sand/mud intertidal - good if sand, avoid mud)
    19: 0.0, # Saltmarsh (often muddy, uneven, creeks)
    20: 0.0, # Urban (unsuitable)
    21: 0.0, # Suburban (unsuitable)
} # Keep user dictionary


DEFAULT_SUITABILITY_SCORE = 0.0

# --- Class Names and Colors for Legend ---
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
    21: ('Suburban', cmap_tab20(20)), 0: ('Nodata / Unclassified', (1, 1, 1, 0)) # Nodata color set to transparent white
}
LCM_NODATA_VAL = 0

# --- 5. Output Configuration ---
OUTPUT_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_SLOPE_CONSIDERED = 45.0

# --- 6. Visualization ---
VIS_CMAP_SCORE = 'viridis'
VIS_CMAP_SLOPE = 'magma'
CTX_PROVIDER = cx.providers.OpenStreetMap.Mapnik
CTX_ALPHA = 0.7 # Transparency of the main raster overlay

# =============================================================================
# --- END OF Configuration ---
# =============================================================================

print("--- Barefoot Suitability Mapping v3.0 + Basemap Plot ---") # Updated title

# --- Helper Functions ---

def get_bounding_box_wgs84(latitude, longitude, radius_km):
    """ Calculates bounding box (min_lon, min_lat, max_lon, max_lat) in WGS84."""
    # (Function content unchanged)
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180): raise ValueError("Invalid lat/lon.")
    if radius_km <= 0: raise ValueError("Radius must be positive.")
    earth_radius_km = 6371.0
    lat_delta_deg = math.degrees(radius_km / earth_radius_km)
    clamped_lat_rad = math.radians(max(-89.9, min(89.9, latitude)))
    lon_delta_deg = 180.0 if abs(math.cos(clamped_lat_rad)) < 1e-9 else math.degrees(radius_km / (earth_radius_km * math.cos(clamped_lat_rad)))
    min_lat, max_lat = max(-90.0, latitude - lat_delta_deg), min(90.0, latitude + lat_delta_deg)
    min_lon = (longitude - lon_delta_deg + 540) % 360 - 180
    max_lon = (longitude + lon_delta_deg + 540) % 360 - 180
    if min_lon > max_lon: min_lon, max_lon = max_lon, min_lon
    return (min_lon, min_lat, max_lon, max_lat)

def plot_raster_with_contextily(raster_path, output_png_path, title, cmap='viridis',
                                vmin=None, vmax=None, is_discrete=False,
                                class_info=None, nodata_val=None, ctx_alpha=0.8):
    """Visualizes a raster with contextily basemap, saving it to a PNG file."""
    print(f"  Visualizing Combined: {title} -> {os.path.basename(output_png_path)}") # Clarified print
    try:
        with rasterio.open(raster_path) as src:
            if src.crs != TARGET_CRS:
                 print(f"    WARNING: Raster CRS ({src.crs}) does not match TARGET_CRS ({TARGET_CRS}) for plotting!")

            data = src.read(1, masked=True)
            valid_data = data[~data.mask]
            if valid_data.size == 0:
                print(f"    WARNING: No valid data pixels found after masking nodata. Skipping plot.")
                return

            fig, ax = plt.subplots(1, 1, figsize=(12, 10))

            # --- Plot Raster Data ---
            # Added zorder=10 to ensure raster plots on top of basemap
            plot_kwargs = {'alpha': ctx_alpha, 'zorder': 10}

            raster_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

            if is_discrete and class_info:
                codes = sorted(class_info.keys())
                colors = [class_info[code][1] for code in codes]
                names = [class_info[code][0] for code in codes]
                if nodata_val is not None and nodata_val in codes:
                     nodata_idx = codes.index(nodata_val)
                     colors[nodata_idx] = (1, 1, 1, 0) # RGBA Transparent White

                discrete_cmap = ListedColormap(colors)
                bounds = [code - 0.5 for code in codes] + [codes[-1] + 0.5]
                norm = BoundaryNorm(bounds, discrete_cmap.N)

                # Added zorder=10
                img = ax.imshow(data, cmap=discrete_cmap, norm=norm, extent=raster_extent, **plot_kwargs)

                legend_patches = [Patch(facecolor=colors[i], edgecolor='black', label=f'{codes[i]}: {names[i]}')
                                  for i in range(len(codes)) if codes[i] != nodata_val]
                ax.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="LCM Classes", fontsize='small')

            else: # Continuous data
                # Added zorder=10
                img = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=raster_extent, **plot_kwargs)
                fig.colorbar(img, ax=ax, label='Score/Value', shrink=0.7)

            # --- Add Contextily Basemap ---
            try:
                print(f"    Adding Contextily basemap (CRS: {src.crs.to_string()})...")
                # Basemap should plot underneath due to lower default zorder
                cx.add_basemap(ax, crs=src.crs.to_string(), source=CTX_PROVIDER, zoom='auto')
                print("    Basemap added successfully.")
            except Exception as ctx_e:
                print(f"    ERROR adding contextily basemap: {ctx_e}")
                print(f"    Check network connection and if provider {CTX_PROVIDER} is available.")

            ax.set_title(title, fontsize=14)
            ax.set_axis_off()
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.savefig(output_png_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"    Combined visualization saved: {output_png_path}") # Clarified print

    except FileNotFoundError:
         print(f"    Error: Raster file not found for visualization: {raster_path}")
    except Exception as e:
        print(f"    Error visualizing {raster_path}: {e}")
        import traceback
        traceback.print_exc()

# <<< NEW FUNCTION for standalone basemap >>>
def plot_contextily_basemap_standalone(target_bounds, target_crs_str, output_png_path, title):
    """Plots ONLY the contextily basemap for the given bounds and CRS."""
    print(f"  Visualizing Basemap Only: {title} -> {os.path.basename(output_png_path)}")
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Set axis limits based on target_bounds BEFORE adding basemap
        minx, miny, maxx, maxy = target_bounds
        if not all(np.isfinite([minx, miny, maxx, maxy])):
             print("    ERROR: Invalid target bounds for basemap plot.")
             plt.close(fig)
             return
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        # Add Contextily Basemap
        print(f"    Adding Contextily basemap (CRS: {target_crs_str})...")
        cx.add_basemap(ax, crs=target_crs_str, source=CTX_PROVIDER, zoom='auto')
        print("    Basemap added successfully.")

        ax.set_title(title, fontsize=14)
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(output_png_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"    Basemap visualization saved: {output_png_path}")

    except Exception as e:
        print(f"    Error visualizing basemap: {e}")
        import traceback
        traceback.print_exc()
# <<< END NEW FUNCTION >>>

def calculate_slope_degrees(dem_data, x_res, y_res, nodata_val):
    """Calculates slope in degrees using numpy.gradient, handling nodata."""
    # (Function content unchanged)
    dem_data_float = dem_data.astype(np.float32)
    if nodata_val is not None:
        nodata_mask = np.isclose(dem_data_float, nodata_val) | np.isnan(dem_data_float)
        dem_data_float[nodata_mask] = np.nan
    else:
        nodata_mask = np.isnan(dem_data_float)
    gy, gx = np.gradient(dem_data_float, y_res, x_res)
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    slope_deg = np.degrees(slope_rad)
    slope_deg[nodata_mask] = nodata_val if nodata_val is not None else -9999.0
    return slope_deg.astype(np.float32)

# --- Dynamic Path Generation ---
# (Section content unchanged)
place_slug = PLACE_NAME.replace(' ', '_').replace(',', '').lower()
OUTPUT_SUBDIR_NAME = f"outputs_{place_slug}_radius{RADIUS_KM}km_barefoot"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, OUTPUT_SUBDIR_NAME)
print(f"Creating output directory: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
if not os.path.isdir(OUTPUT_DIR):
    print(f"FATAL ERROR: Failed to create output directory: {OUTPUT_DIR}"); sys.exit(1)

# --- Define Output Paths ---
# (Section content unchanged)
ALIGNED_DEM_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_aligned_dem.tif")
ALIGNED_LCM_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_aligned_lcm.tif")
SLOPE_RASTER_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_slope_degrees.tif")
SUITABILITY_RASTER_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_land_cover_suitability.tif")
COMBINED_SCORE_RASTER_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_combined_suitability_score.tif")
VIS_SCORE_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_combined_suitability_map.png")
VIS_SLOPE_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_slope_map.png")
VIS_SUITABILITY_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_suitability_map.png")
VIS_LCM_CLASS_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_lcm_classes_map.png")
VIS_BASEMAP_PATH = os.path.join(OUTPUT_DIR, f"{place_slug}_basemap_only.png") # <<< New path for basemap plot >>>

# --- Main Processing Steps ---

try:
    # --- 1. Geocode Place Name and Define AOI Bounds ---
    # (Section content unchanged)
    print(f"\n--- Step 1: Defining Area of Interest (AOI) ---")
    print(f"  Geocoding '{PLACE_NAME}'...")
    geolocator = Nominatim(user_agent="barefoot_suitability_mapper_v3.0_basemap") # Updated agent
    center_lat, center_lon = None, None
    try:
        location = geolocator.geocode(PLACE_NAME, timeout=15)
        if not location: print(f"FATAL ERROR: Could not geocode '{PLACE_NAME}'."); sys.exit(1)
        center_lat, center_lon = location.latitude, location.longitude
        print(f"  Coordinates found (WGS84): Lat={center_lat:.4f}, Lon={center_lon:.4f}")
        bounds_wgs84 = get_bounding_box_wgs84(center_lat, center_lon, RADIUS_KM)
        print(f"  Calculated Bounding Box (WGS84): {bounds_wgs84}")
        target_bounds = rasterio.warp.transform_bounds(SOURCE_CRS_GPS, TARGET_CRS, *bounds_wgs84)
        print(f"  Transformed Bounding Box ({TARGET_CRS}): {target_bounds}")
        crop_box_geom = box(*target_bounds)
    except (GeocoderTimedOut, GeocoderServiceError) as e: print(f"FATAL ERROR: Geocoding failed: {e}."); sys.exit(1)
    except Exception as e: print(f"FATAL ERROR during AOI definition: {e}"); sys.exit(1)

    # --- 1b. Plot Standalone Basemap --- <<< ADDED STEP >>>
    print(f"\n--- Step 1b: Visualizing Standalone Basemap ---")
    plot_contextily_basemap_standalone(
        target_bounds, TARGET_CRS, VIS_BASEMAP_PATH,
        f"Contextily Basemap Only - {PLACE_NAME}"
    )

    # --- 2. Open Input Rasters & Check Intersection ---
    # (Section content unchanged)
    print("\n--- Step 2: Opening Input Rasters & Checking AOI Intersection ---")
    with rasterio.open(PROJECTED_DEM_PATH) as dem_src, rasterio.open(LCM_PATH) as lcm_src:
        print(f"  DEM: {PROJECTED_DEM_PATH} (CRS: {dem_src.crs}, Resolution: {dem_src.res})")
        print(f"  LCM: {LCM_PATH} (CRS: {lcm_src.crs}, Resolution: {lcm_src.res}, Bands: {lcm_src.count})")
        if dem_src.crs != TARGET_CRS: print(f"FATAL ERROR: DEM CRS ({dem_src.crs}) != TARGET_CRS ({TARGET_CRS})."); sys.exit(1)
        if lcm_src.crs != TARGET_CRS: print(f"  Warning: LCM CRS ({lcm_src.crs}) != TARGET_CRS ({TARGET_CRS}). Will reproject.")
        dem_bounds_geom = box(*dem_src.bounds)
        if not crop_box_geom.intersects(dem_bounds_geom): print(f"FATAL ERROR: AOI bounds {target_bounds} do not intersect DEM bounds {dem_src.bounds}."); sys.exit(1)
        else: print(f"  AOI intersects with DEM bounds.")
        try:
            aoi_bounds_lcm_crs = rasterio.warp.transform_bounds(TARGET_CRS, lcm_src.crs, *target_bounds)
            lcm_bounds_geom = box(*lcm_src.bounds)
            aoi_in_lcm_crs_geom = box(*aoi_bounds_lcm_crs)
            if not aoi_in_lcm_crs_geom.intersects(lcm_bounds_geom): print(f"FATAL ERROR: AOI bounds (transformed to {lcm_src.crs}: {aoi_bounds_lcm_crs}) do not intersect LCM bounds {lcm_src.bounds}."); sys.exit(1)
            else: print(f"  AOI intersects with LCM bounds.")
        except Exception as e: print(f"    Warning: Could not perform robust LCM bounds check: {e}")

        # --- Step 3: Define Target Profile & Align Rasters ---
        # (Section content unchanged)
        print(f"\n--- Step 3: Aligning Rasters to Target Grid within AOI ---")
        print(f"  Targeting LCM resolution: {lcm_src.res}")
        dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
            TARGET_CRS, TARGET_CRS,
            int(np.ceil((target_bounds[2]-target_bounds[0]) / lcm_src.res[0])),
            int(np.ceil((target_bounds[3]-target_bounds[1]) / lcm_src.res[1])),
            *target_bounds, resolution=lcm_src.res
        )
        align_profile = dem_src.profile.copy()
        align_profile.update({'crs': TARGET_CRS, 'transform': dst_transform, 'width': dst_width, 'height': dst_height, 'nodata': -9999.0})

    # --- 3a. Align DEM ---
    # (Section content unchanged)
    print(f"  Aligning DEM to: {ALIGNED_DEM_PATH}")
    align_profile_dem = align_profile.copy(); align_profile_dem.update(dtype='float32')
    with rasterio.open(PROJECTED_DEM_PATH) as dem_src, rasterio.open(ALIGNED_DEM_PATH, 'w', **align_profile_dem) as dst:
        rasterio.warp.reproject(
            source=rasterio.band(dem_src, 1), destination=rasterio.band(dst, 1),
            src_transform=dem_src.transform, src_crs=dem_src.crs,
            dst_transform=align_profile_dem['transform'], dst_crs=align_profile_dem['crs'],
            resampling=Resampling.bilinear, dst_nodata=align_profile_dem['nodata']
        )
    print("  DEM Alignment complete.")

    # --- 3b. Align LCM ---
    # (Section content unchanged)
    print(f"  Aligning LCM to: {ALIGNED_LCM_PATH}")
    align_profile_lcm = align_profile.copy(); align_profile_lcm.update(dtype='int32', nodata=LCM_NODATA_VAL)
    with rasterio.open(LCM_PATH) as lcm_src, rasterio.open(ALIGNED_LCM_PATH, 'w', **align_profile_lcm) as dst:
        rasterio.warp.reproject(
            source=rasterio.band(lcm_src, 1), destination=rasterio.band(dst, 1),
            src_transform=lcm_src.transform, src_crs=lcm_src.crs,
            dst_transform=align_profile_lcm['transform'], dst_crs=align_profile_lcm['crs'],
            resampling=Resampling.nearest, dst_nodata=align_profile_lcm['nodata']
        )
    print("  LCM Alignment complete.")

    # --- 3c. Visualize Raw LCM Classes (Combined) ---
    # (Calls combined plotting function - unchanged)
    print(f"\n--- Step 3c: Visualizing Raw Land Cover Classes ---")
    plot_raster_with_contextily(
        ALIGNED_LCM_PATH, VIS_LCM_CLASS_PATH, f"Land Cover Classes - {PLACE_NAME}",
        is_discrete=True, class_info=LCM_CLASS_INFO, nodata_val=LCM_NODATA_VAL, ctx_alpha=CTX_ALPHA
    )

    # --- 4. Calculate Slope ---
    # (Calls combined plotting function - unchanged)
    print(f"\n--- Step 4: Calculating Slope ---")
    print(f"  Processing DEM: {ALIGNED_DEM_PATH}"); print(f"  Saving Slope to: {SLOPE_RASTER_PATH}")
    slope_profile = align_profile_dem.copy()
    try:
        with rasterio.open(ALIGNED_DEM_PATH) as dem_aligned_src:
            dem_data = dem_aligned_src.read(1); x_res, y_res = dem_aligned_src.res; nodata_val = dem_aligned_src.nodata
            slope_deg = calculate_slope_degrees(dem_data, x_res, abs(y_res), nodata_val)
            with rasterio.open(SLOPE_RASTER_PATH, 'w', **slope_profile) as dst: dst.write(slope_deg, 1)
        print("  Slope calculation complete.")
        plot_raster_with_contextily(
            SLOPE_RASTER_PATH, VIS_SLOPE_PATH, f"Slope (Degrees) - {PLACE_NAME}",
            cmap=VIS_CMAP_SLOPE, ctx_alpha=CTX_ALPHA, vmin=0, vmax=MAX_SLOPE_CONSIDERED
        )
    except FileNotFoundError: print(f"  ERROR: Aligned DEM file not found: {ALIGNED_DEM_PATH}")
    except Exception as e: print(f"  ERROR during slope calculation: {e}")

    # --- 5. Reclassify Land Cover ---
    # (Calls combined plotting function - unchanged)
    print(f"\n--- Step 5: Reclassifying Land Cover Suitability ---")
    print(f"  Processing LCM: {ALIGNED_LCM_PATH}"); print(f"  Saving Suitability to: {SUITABILITY_RASTER_PATH}")
    suitability_profile = align_profile.copy(); suitability_profile.update(dtype='float32', nodata=-1.0)
    try:
        with rasterio.open(ALIGNED_LCM_PATH) as lcm_aligned_src:
            lcm_data = lcm_aligned_src.read(1); nodata_lcm = lcm_aligned_src.nodata
            suitability_data = np.full(lcm_data.shape, suitability_profile['nodata'], dtype=np.float32)
            valid_lcm_mask = (lcm_data != nodata_lcm)
            for code, score in LAND_COVER_SUITABILITY.items(): suitability_data[(lcm_data == code) & valid_lcm_mask] = score
            mapped_codes = set(LAND_COVER_SUITABILITY.keys()); unmapped_mask = valid_lcm_mask.copy()
            for code in mapped_codes: unmapped_mask &= (lcm_data != code)
            suitability_data[unmapped_mask] = DEFAULT_SUITABILITY_SCORE
        with rasterio.open(SUITABILITY_RASTER_PATH, 'w', **suitability_profile) as dst: dst.write(suitability_data, 1)
        print("  Land cover reclassification complete.")
        plot_raster_with_contextily(
            SUITABILITY_RASTER_PATH, VIS_SUITABILITY_PATH, f"Barefoot Suitability Score - {PLACE_NAME}",
            cmap='YlGn', vmin=0, vmax=1, ctx_alpha=CTX_ALPHA
        )
    except FileNotFoundError: print(f"  ERROR: Aligned LCM file not found: {ALIGNED_LCM_PATH}")
    except Exception as e: print(f"  ERROR during land cover reclassification: {e}")

    # --- 6. Combine Scores ---
    # (Calls combined plotting function - unchanged)
    print("\n--- Step 6: Combining Slope and Suitability ---")
    if not os.path.exists(SLOPE_RASTER_PATH) or not os.path.exists(SUITABILITY_RASTER_PATH):
        print("  Skipping combination: Input slope or suitability raster missing.")
    else:
        print(f"  Reading Slope: {SLOPE_RASTER_PATH}"); print(f"  Reading Suitability: {SUITABILITY_RASTER_PATH}")
        print(f"  Saving Combined Score to: {COMBINED_SCORE_RASTER_PATH}")
        combined_profile = align_profile.copy(); combined_profile.update(dtype='float32', nodata=-1.0)
        try:
            with rasterio.open(SLOPE_RASTER_PATH) as slope_src, rasterio.open(SUITABILITY_RASTER_PATH) as suitability_src:
                if not (slope_src.profile['transform'] == suitability_src.profile['transform'] and \
                        slope_src.profile['crs'] == suitability_src.profile['crs'] and \
                        slope_src.profile['width'] == suitability_src.profile['width'] and \
                        slope_src.profile['height'] == suitability_src.profile['height']):
                    print("  FATAL ERROR: Slope and Suitability rasters do not align."); sys.exit(1)
                slope_data = slope_src.read(1, masked=True); suitability_data = suitability_src.read(1, masked=True)
                normalized_slope = np.ma.clip(slope_data, 0, MAX_SLOPE_CONSIDERED) / MAX_SLOPE_CONSIDERED
                combined_score_masked = normalized_slope * suitability_data
                combined_score = np.ma.filled(combined_score_masked, combined_profile['nodata'])
            with rasterio.open(COMBINED_SCORE_RASTER_PATH, 'w', **combined_profile) as dst: dst.write(combined_score.astype(np.float32), 1)
            print("  Score combination complete.")

            # --- 7. Visualize Final Score ---
            print("\n--- Step 7: Visualizing Final Score ---")
            plot_raster_with_contextily(
                COMBINED_SCORE_RASTER_PATH, VIS_SCORE_PATH, f"Combined Barefoot Suitability Score - {PLACE_NAME}",
                cmap=VIS_CMAP_SCORE, vmin=0, vmax=1, ctx_alpha=CTX_ALPHA
            )
        except FileNotFoundError: print(f"  ERROR: Could not find slope or suitability file for combination.")
        except Exception as e: print(f"  ERROR during score combination: {e}")


except rasterio.RasterioIOError as e:
    print(f"\nFATAL ERROR: Could not open input raster file: {e}")
    print(f"  Check paths: DEM='{PROJECTED_DEM_PATH}', LCM='{LCM_PATH}'"); sys.exit(1)
except ImportError as e:
     print(f"\nFATAL ERROR: Missing required library: {e}")
     print("Ensure contextily is installed ('pip install contextily')"); sys.exit(1)
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
    import traceback
    traceback.print_exc(); sys.exit(1)

print("\n--- Workflow Finished ---")
print(f"Outputs saved in: {OUTPUT_DIR}")
