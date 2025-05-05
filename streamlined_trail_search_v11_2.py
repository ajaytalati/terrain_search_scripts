#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 21:36:53 2025

@author: ajay
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidated script for finding, analyzing, ranking, and visualizing steep trails
based on OSM data and a DEM raster.

Version 11.2:
- Added rank number labels to trails on the output map visualization.
- Added 'max_elevation_m' column (max of start/end elevation) to outputs.
- Captures all non-specified OSM tags present on trail ways into a single
  dictionary column ('other_osm_tags' by default).
- Calculates minimum distance from trail centroid to nearest car park and relevant road.
- Extracts potential popularity proxy tags (name, wikidata, wikipedia, tourism, historic).
- Adds distance metrics and proxy tags to the output.
- Adds configuration for relevant road types for distance calculation.
- Improved documentation and structure in the configuration section.
- Implemented tag-based exclusion filters (from v9.0).
- Uses GeoJSON export method for reliable tag extraction (from v8.3).

Requires: osmium-tool, geopandas, rasterio, pandas, numpy, matplotlib, contextily, geopy, time, rtree (for geopandas sjoin_nearest)
"""

import os
import sys
import warnings
import subprocess
import shutil
import time # Added for delay in reverse geocoding
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
# import elevation # For downloading DEM - Commented out if DEM is pre-downloaded
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Point, LineString, MultiPoint, MultiLineString, Polygon, MultiPolygon # Added more types for access points
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError # Added for error handling
from geopy.distance import distance as geopy_distance # Added for distance calculation
import math
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects # Import for text outline
import contextily as cx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import json # For handling the other_tags dictionary nicely

# Check for rtree dependency, crucial for spatial indexing / sjoin_nearest
try:
    import rtree
except ImportError:
    print("Error: The 'rtree' library is required for spatial indexing but is not installed.")
    print("Please install it, e.g., using 'pip install rtree' or 'conda install rtree'.")
    sys.exit(1)

# =============================================================================
# =============================================================================
# --- Configuration ---
# =============================================================================
# =============================================================================
# This section contains all the parameters you might need to adjust for your specific search.

# --- 1. Search Area ---
# Define the center point for the search.
#PLACE_NAME = "Kendal, UK" # Examples: "Edale, UK", "Fort William, UK", "Buxton, UK"
#PLACE_NAME = "Edale, UK"
#PLACE_NAME = "Grimsby, UK"
#PLACE_NAME = "Hathersage, UK"

# --- Lake District is simply another level compared to Peak District - BUT VERY HARD/Unsuitable for barefoot
#PLACE_NAME = "Scafell Pike, UK"
#PLACE_NAME = "Keswick, UK"

# --- Gemini advises optimal places
#PLACE_NAME = "Howgills, UK" # Ribble Valley, this is East of Kendal/Killington Lake # Border of Yorkshire Dales and Lake District # both steep slopes, AND grass/fells
#PLACE_NAME = "Sedbergh, UK"
#PLACE_NAME = "Merthyr Mawr Nature Reserve, UK" # Sand dunes can be steep, and sand is perfect for barefoot !!!

# ---- Local sand
PLACE_NAME = "Saltfleet, UK"

# Define the search radius around the center point in kilometers.
RADIUS_KM = 10

# --- 2. Input Data Paths ---
# Full path to the OpenStreetMap PBF file covering your region of interest.
# Download from sources like Geofabrik: https://download.geofabrik.de/
PBF_FILE_PATH = "/home/ajay/Python_Projects/steepest_trails_search/united-kingdom-latest.osm.pbf" # <- VERIFY THIS PATH
# Full path to the Digital Elevation Model (DEM) raster file covering your region.
# IMPORTANT: The DEM *must* be pre-projected to the TARGET_CRS specified below.
PROJECTED_DEM_PATH = "/home/ajay/Python_Projects/steepest_trails_search/outputs_united_kingdom_dem/united_kingdom_dem_proj_27700.tif" # <- UPDATE IF NEEDED

# --- 3. Coordinate Reference System (CRS) ---
# Target CRS for all analysis and output. MUST match the CRS of the PROJECTED_DEM_PATH.
# Example: "EPSG:27700" for Ordnance Survey National Grid (UK).
TARGET_CRS = "EPSG:27700"

# --- 4. Trail Identification & Tag Extraction ---
# Initial filter: OSM 'highway' tag values used to identify potential trails.
# See https://wiki.openstreetmap.org/wiki/Key:highway for common values.
# Example: TRAIL_TAGS = { "highway": ["path", "footway", "bridleway", "track"] }
TRAIL_TAGS = { "highway": ["bridleway", "track", "path", "footway"] } # Broader example
# List of specific OSM tag keys you want to have as DEDICATED COLUMNS in the output CSV files,
# if they are present on the OSM ways.
# Other tags found on the ways will be collected into the 'other_osm_tags' dictionary column (see below).
OPTIONAL_OSM_TAGS = [
    'highway', 'name', 'surface', 'sac_scale', 'trail_visibility', 'access',
    'designation', 'tracktype', 'smoothness'
    # Note: 'wikidata', 'wikipedia', 'tourism', 'historic' etc. are removed from here.
    # They will appear in the 'other_osm_tags' dict if present on the trail way itself.
    # Add them back here ONLY if you specifically want them as separate columns.
]
# The column name used internally and in outputs for the OSM unique identifier.
ID_COLUMN = 'osmid'
# The column name for the dictionary that will store all other tags found on the trail ways.
OTHER_TAGS_COLUMN_NAME = 'other_osm_tags'

# --- 5. Trail Filtering Criteria ---
# --- 5a. Tag-Based Exclusion Filter (Applied BEFORE Elevation Analysis) ---
# Define specific tag values that should cause a trail to be EXCLUDED early.
# Format: { "tag_key": ["value1_to_exclude", "value2_to_exclude", ...], ... }
# Note: 'access: [no, private]' is also handled by the initial osmium filter for efficiency,
#       but including it here documents the intent and acts as a safeguard.
EXCLUDE_TAG_VALUES = {
    "surface": ["paved", "asphalt", "concrete", "sett"], # Exclude hard surfaces
    "highway": ["motorway", "trunk", "primary", "secondary", "tertiary", "residential", "service", "steps", "construction"], # Exclude roads, steps, construction
    "access": ["no", "private"], # Exclude explicitly private/no access ways
    "area": ["yes"] # Exclude ways that are primarily areas (e.g., area:highway=path)
    # Add more tags and values to exclude as needed, e.g.:
    # "motor_vehicle": ["yes", "designated"]
}
# --- 5b. Metric-Based Filters (Applied AFTER Elevation Analysis) ---
# Trails shorter than this length (in meters) will be excluded. Set to 0 or None to disable.
MIN_TRAIL_LENGTH_FILTER = 400
# Trails with an absolute net elevation gain/loss less than this (in meters) will be excluded.
# Set to 0 or None to disable.
ABSOLUTE_MIN_NET_GAIN = 25
# Column name used for the calculated net elevation gain.
NET_GAIN_COLUMN = 'net_gain_m'
# Column name for the calculated maximum elevation (max of start/end).
MAX_ELEVATION_COLUMN = 'max_elevation_m' # New column name

# --- 6. Remoteness / Access Point Proximity ---
# Calculate minimum distance to nearest car park and specified road types.
CALCULATE_ACCESS_DISTANCE = True # Set to False to disable this calculation
# OSM highway tag values to consider as "access roads" for distance calculation.
# Exclude trail/path types themselves unless you want distance to *any* highway.
ACCESS_ROAD_HIGHWAY_VALUES = [
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
    "residential", "motorway_link", "trunk_link", "primary_link", "secondary_link",
    "tertiary_link", "living_street", "service", "road"
]
# Column names for the calculated minimum distances.
DIST_CARPARK_COL = 'min_dist_carpark_m'
DIST_ROAD_COL = 'min_dist_road_m'

# --- 7. Trail Ranking Criteria ---
# The column name containing the primary metric used to rank the filtered trails.
# Common options calculated by the script:
#   'absolute_avg_gradient_degrees': Absolute value of the average gradient.
#   'avg_uphill_gradient_degrees': Average gradient (positive for uphill, negative for downhill).
#   'net_gain_m': Net elevation gain (can be negative).
#   MAX_ELEVATION_COLUMN: Maximum elevation reached (max of start/end).
#   DIST_CARPARK_COL, DIST_ROAD_COL (if CALCULATE_ACCESS_DISTANCE is True)
RANKING_COLUMN = 'absolute_avg_gradient_degrees'
# Sort order for ranking: True for descending (highest value is Rank 1), False for ascending.
RANK_DESCENDING = True
# The name for the column that will store the calculated rank (1, 2, 3...).
RANK_COLUMN_NAME = 'rank'
# --- Note: Combining steepness and distance into a single score is not implemented ---
# --- by default, but all metrics will be available in the output for sorting/filtering. ---

# --- 8. Output & Visualization ---
# Number of top-ranked trails to include in the detailed output CSV and map visualization.
N_TOP_TRAILS = 20
# Matplotlib colormap name for visualizing trail ranks on the map.
# Examples: 'viridis_r', 'plasma', 'inferno', 'magma_r', 'coolwarm'
VIS_COLORMAP = 'viridis_r'
# Base directory where the output subfolder will be created. Defaults to the script's directory.
OUTPUT_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# =============================================================================
# --- END OF Configuration ---
# --- Script Logic Below (Generally no need to modify) ---
# =============================================================================
# =============================================================================

print(f"--- Consolidated Trail Analysis Workflow ---")
print(f"Start time: {np.datetime64('now', 's')} UTC")

# --- Dynamic Path Generation ---
def generate_paths(base_dir, place_name, radius_km):
    """Generates output directory and file paths based on configuration."""
    place_slug = place_name.replace(' ', '_').replace(',', '').lower()
    output_dir_name = f"outputs_{place_slug}_radius{radius_km}km_consolidated"
    output_dir = os.path.join(base_dir, output_dir_name)
    paths = {
        "output_dir": output_dir,
        "temp_bbox_pbf": os.path.join(output_dir, "temp_bbox_extract.osm.pbf"),
        "temp_trails_pbf": os.path.join(output_dir, "temp_filtered_trails_initial.osm.pbf"),
        "temp_trails_geojson": os.path.join(output_dir, "temp_filtered_trails.geojson"),
        "temp_access_geojson": os.path.join(output_dir, "temp_access_points.geojson"), # For car parks/roads
        "stats_csv": os.path.join(output_dir, f"{place_slug}_trail_elevation_stats.csv"),
        "ranked_stats_csv": os.path.join(output_dir, f"{place_slug}_trail_elevation_stats_ranked.csv"),
        "top_details_csv": os.path.join(output_dir, f"{place_slug}_top_{N_TOP_TRAILS}_details.csv"),
        "visualization_png": os.path.join(output_dir, f"{place_slug}_top_{N_TOP_TRAILS}_map.png"),
    }
    return paths

PATHS = generate_paths(OUTPUT_BASE_DIR, PLACE_NAME, RADIUS_KM)
print(f"Ensuring output directory exists: {PATHS['output_dir']}")
os.makedirs(PATHS['output_dir'], exist_ok=True)
if not os.path.isdir(PATHS['output_dir']):
    print(f"FATAL ERROR: Failed to create output directory: {PATHS['output_dir']}"); sys.exit(1)

# --- Helper Functions ---
def get_bounding_box(latitude, longitude, radius_km):
    """ Calculates bounding box (min_lon, min_lat, max_lon, max_lat) in WGS84."""
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180): raise ValueError("Invalid lat/lon.")
    if radius_km <= 0: raise ValueError("Radius must be positive.")
    earth_radius_km = 6371.0
    lat_delta_deg = math.degrees(radius_km / earth_radius_km)
    clamped_lat_rad = math.radians(max(-89.9, min(89.9, latitude)))
    lon_delta_deg = 180.0 if math.cos(clamped_lat_rad) < 1e-9 else math.degrees(radius_km / (earth_radius_km * math.cos(clamped_lat_rad)))
    min_lat, max_lat = max(-90.0, latitude - lat_delta_deg), min(90.0, latitude + lat_delta_deg)
    min_lon, max_lon = (longitude - lon_delta_deg + 180) % 360 - 180, (longitude + lon_delta_deg + 180) % 360 - 180
    return (min_lon, min_lat, max_lon, max_lat)

def run_osmium_command(cmd, description):
    """Runs an osmium command and handles errors."""
    print(f"Running osmium command: {description}")
    print(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error during '{description}':\n{result.stderr}")
            return False
        print(f"Osmium command '{description}' completed successfully.")
        return True
    except Exception as e:
        print(f"Failed to run osmium command '{description}': {e}")
        return False

def extract_osm_features(pbf_path, bbox_ll, tags_filter, output_geojson_path, description, add_id=True):
    """
    Extracts specific OSM features (based on tags) within a bbox to GeoJSON using osmium.
    Handles nodes, ways, and relations. Adds unique ID by default.
    """
    print(f"\n--- Extracting {description} ---")
    temp_extract_pbf = os.path.join(os.path.dirname(output_geojson_path), f"temp_extract_{description}.osm.pbf")
    temp_filter_pbf = os.path.join(os.path.dirname(output_geojson_path), f"temp_filter_{description}.osm.pbf")

    # Step 1: Extract BBox
    bbox_str = f"{bbox_ll[0]},{bbox_ll[1]},{bbox_ll[2]},{bbox_ll[3]}"
    extract_cmd = ["osmium", "extract", "-b", bbox_str, pbf_path, "-o", temp_extract_pbf, "--overwrite"]
    if not run_osmium_command(extract_cmd, f"bbox extract for {description}"): return None

    # Step 2: Filter by Tags
    tag_filter_parts = []
    for key, values in tags_filter.items():
        # Apply filter to nodes, ways, and relations (nwr)
        tag_filter_parts.append(f"nwr/{key}={','.join(values)}")

    filter_cmd = ["osmium", "tags-filter", temp_extract_pbf] + tag_filter_parts + ["-o", temp_filter_pbf, "--overwrite"]
    if not run_osmium_command(filter_cmd, f"tags filter for {description}"):
        if os.path.exists(temp_extract_pbf): os.remove(temp_extract_pbf)
        return None

    # Step 3: Export to GeoJSON
    export_cmd_base = ["osmium", "export", temp_filter_pbf]
    if add_id:
        export_cmd_base.extend(["--add-unique-id=type_id"])
    export_cmd = export_cmd_base + ["-o", output_geojson_path, "--overwrite"]
    if not run_osmium_command(export_cmd, f"geojson export for {description}"):
        # Clean up even if export fails
        if os.path.exists(temp_extract_pbf): os.remove(temp_extract_pbf)
        if os.path.exists(temp_filter_pbf): os.remove(temp_filter_pbf)
        if os.path.exists(output_geojson_path): os.remove(output_geojson_path)
        return None

    # Cleanup intermediate PBFs
    try:
        if os.path.exists(temp_extract_pbf): os.remove(temp_extract_pbf)
        if os.path.exists(temp_filter_pbf): os.remove(temp_filter_pbf)
    except OSError as e:
        print(f"Warning: Could not remove temporary PBF file for {description}: {e}")

    # Step 4: Read GeoJSON
    print(f"Reading GeoJSON for {description}: {output_geojson_path}...")
    if os.path.exists(output_geojson_path) and os.path.getsize(output_geojson_path) > 0:
        try:
            gdf = gpd.read_file(output_geojson_path)
            print(f"Successfully read {len(gdf)} {description} features from GeoJSON.")
            return gdf
        except Exception as read_err:
            print(f"Error reading GeoJSON file for {description}: {read_err}")
            return None
    else:
        print(f"GeoJSON file for {description} is empty or does not exist.")
        return gpd.GeoDataFrame() # Return empty GDF if no features found

def standardize_and_reproject(gdf, target_crs, id_column=None, expected_geojson_id_col='id',
                              optional_tags_as_cols=None, other_tags_col_name=None):
    """
    Standardizes ID column, reprojects GeoDataFrame, and optionally consolidates
    extra tags into a dictionary column.
    """
    if gdf is None or gdf.empty: return gdf

    # --- Standardize ID Column ---
    if id_column and expected_geojson_id_col:
        if expected_geojson_id_col in gdf.columns:
            if id_column != expected_geojson_id_col:
                print(f"Renaming GeoJSON ID column '{expected_geojson_id_col}' to '{id_column}'.")
                gdf = gdf.rename(columns={expected_geojson_id_col: id_column})
            else: print(f"Found expected GeoJSON ID column '{id_column}'.")
            try: # Extract numeric part (e.g., '12345' from 'way/12345')
                gdf[id_column] = gdf[id_column].astype(str).str.extract(r'(\d+)$', expand=False).fillna(gdf[id_column])
                gdf[id_column] = gdf[id_column].astype(str)
                print(f"ID column '{id_column}' standardized.")
            except Exception as e: print(f"Warning: Could not extract numeric part from ID '{id_column}': {e}.")
        else: print(f"Warning: Expected GeoJSON ID column '{expected_geojson_id_col}' not found.")

    # --- Consolidate Other Tags ---
    if other_tags_col_name:
        print(f"Consolidating non-primary tags into '{other_tags_col_name}' column...")
        # Define columns to keep separate (ID, geometry, and explicitly requested optional tags)
        cols_to_keep = ['geometry']
        if id_column and id_column in gdf.columns:
            cols_to_keep.append(id_column)
        if optional_tags_as_cols:
            # Ensure optional_tags_as_cols is a list
            if not isinstance(optional_tags_as_cols, list):
                 optional_tags_as_cols = list(optional_tags_as_cols)
            cols_to_keep.extend([tag for tag in optional_tags_as_cols if tag in gdf.columns])

        # Identify columns to consolidate
        cols_to_consolidate = [col for col in gdf.columns if col not in cols_to_keep]
        print(f"  Columns to keep separate: {cols_to_keep}")
        print(f"  Columns to consolidate: {cols_to_consolidate}")

        if cols_to_consolidate:
            # Create the dictionary column
            # Apply row-wise to create dict, handling potential NaNs
            gdf[other_tags_col_name] = gdf[cols_to_consolidate].apply(
                lambda row: row.dropna().to_dict(), axis=1
            )
            # Drop the original consolidated columns
            gdf = gdf.drop(columns=cols_to_consolidate)
            print(f"  Consolidated tags into '{other_tags_col_name}'.")
        else:
            print("  No extra columns found to consolidate.")
            # Add an empty dictionary column if requested but no tags found
            gdf[other_tags_col_name] = [{} for _ in range(len(gdf))]


    # --- Reproject ---
    print(f"Reprojecting from {gdf.crs} to {target_crs}...")
    if gdf.crs is None:
        print(f"Warning: CRS not found, assuming EPSG:4326 (WGS84).")
        gdf.crs = "EPSG:4326"
    if gdf.crs != target_crs:
        try:
            gdf = gdf.to_crs(target_crs)
            print(f"Reprojection complete. New CRS: {gdf.crs}")
        except Exception as e:
            print(f"Error during reprojection: {e}"); return None
    else:
        print("Data already in target CRS.")
    return gdf

# --- Core Logic Functions ---
def analyze_elevations(trails_gdf, dem_raster_path, id_column='osmid'):
    """ Calculates elevation statistics for each trail using a DEM. """
    print(f"\n--- Step 3: Calculating Elevation Statistics ---")
    output_columns = [
        id_column, 'start_coord_str', 'end_coord_str', 'length_m',
        'start_elev_m', 'end_elev_m', NET_GAIN_COLUMN, MAX_ELEVATION_COLUMN, # Added max elevation
        'avg_uphill_gradient_degrees', 'absolute_avg_gradient_degrees'
    ]
    if trails_gdf is None or trails_gdf.empty: print("Input GDF empty."); return pd.DataFrame(columns=output_columns)
    if not os.path.exists(dem_raster_path): print(f"Error: DEM not found: {dem_raster_path}"); return None

    results = []
    print(f"Opening DEM raster: {dem_raster_path}")
    try:
        with rasterio.open(dem_raster_path) as dem_src:
            if trails_gdf.crs != dem_src.crs: print(f"FATAL ERROR: Trail CRS ({trails_gdf.crs}) != DEM CRS ({dem_src.crs})."); return None
            print(f"DEM CRS ({dem_src.crs}) matches Trail CRS ({trails_gdf.crs}).")
            dem_nodata = dem_src.nodata
            print(f"DEM NoData value: {dem_nodata}")
            print(f"Processing {len(trails_gdf)} trails for elevation...")
            trails_gdf_proc = trails_gdf.reset_index(drop=True)

            def get_elevation_stats(line_geom, raster_src, nodata_val):
                default_stats = {'start_elev': np.nan, 'end_elev': np.nan, 'net_gain': np.nan, 'max_elev': np.nan, 'gradient_degrees': np.nan}
                if not isinstance(line_geom, LineString) or line_geom.is_empty or line_geom.length < 1e-6: return default_stats
                start_pt, end_pt = line_geom.interpolate(0), line_geom.interpolate(line_geom.length)
                sample_coords = [(p.x, p.y) for p in [start_pt, end_pt] if isinstance(p, Point)]
                if len(sample_coords) != 2: return default_stats
                try:
                    sampled_elevs = np.array([val[0] for val in raster_src.sample(sample_coords)], dtype=np.float32)
                    valid_mask = ~np.isnan(sampled_elevs)
                    if nodata_val is not None:
                        if np.issubdtype(type(nodata_val), np.floating): valid_mask &= ~np.isclose(sampled_elevs, nodata_val)
                        else: valid_mask &= (sampled_elevs != nodata_val)
                    if valid_mask.sum() != 2: return default_stats
                    start_elev, end_elev = sampled_elevs[0], sampled_elevs[1]
                    net_gain = end_elev - start_elev
                    length = line_geom.length
                    gradient_degrees = 0.0 if length < 1e-6 else np.degrees(np.arctan2(net_gain, length))
                    max_elev = np.nanmax(sampled_elevs[valid_mask]) if valid_mask.any() else np.nan # Use nanmax
                    return {'start_elev': start_elev, 'end_elev': end_elev, 'net_gain': net_gain, 'max_elev': max_elev, 'gradient_degrees': gradient_degrees}
                except Exception: return default_stats

            processed_count, skipped_no_id, skipped_invalid_geom = 0, 0, 0
            for i, row in trails_gdf_proc.iterrows():
                geom = row.geometry
                if not isinstance(geom, LineString) or geom.is_empty: skipped_invalid_geom += 1; continue
                if id_column not in row or pd.isna(row[id_column]): skipped_no_id += 1; continue
                trail_id = str(row[id_column])
                length = geom.length
                start_coord, end_coord = geom.coords[0], geom.coords[-1]
                elev_stats = get_elevation_stats(geom, dem_src, dem_nodata)
                absolute_gradient = abs(elev_stats['gradient_degrees']) if pd.notna(elev_stats['gradient_degrees']) else np.nan
                results.append({
                    id_column: trail_id,
                    'start_coord_str': str(start_coord),
                    'end_coord_str': str(end_coord),
                    'length_m': length,
                    'start_elev_m': elev_stats['start_elev'],
                    'end_elev_m': elev_stats['end_elev'],
                    NET_GAIN_COLUMN: elev_stats['net_gain'],
                    MAX_ELEVATION_COLUMN: elev_stats['max_elev'], # Add max elevation
                    'avg_uphill_gradient_degrees': elev_stats['gradient_degrees'],
                    'absolute_avg_gradient_degrees': absolute_gradient
                })
                processed_count += 1
                if processed_count % 500 == 0: print(f"  Processed {processed_count}/{len(trails_gdf_proc)} trails...")

            print(f"Finished processing trails. Successfully processed: {processed_count}")
            if skipped_no_id > 0: print(f"Skipped {skipped_no_id} trails due to missing ID.")
            if skipped_invalid_geom > 0: print(f"Skipped {skipped_invalid_geom} features with invalid geometry.")

    except rasterio.RasterioIOError as e: print(f"Error opening/reading DEM '{dem_raster_path}': {e}"); return None
    except Exception as e: print(f"Unexpected error during elevation analysis: {e}"); return None

    stats_df = pd.DataFrame(results)
    if stats_df.empty: print("No trail statistics were generated."); return pd.DataFrame(columns=output_columns)
    else:
        # Ensure all expected columns are present, including the new max_elevation_m
        existing_output_cols = [col for col in output_columns if col in stats_df.columns]
        stats_df = stats_df[existing_output_cols]
        print(f"Generated statistics DataFrame with {len(stats_df)} rows.")
        return stats_df

def calculate_min_distances(trails_gdf, access_points_gdf, id_col):
    """
    Calculates the minimum distance from each trail centroid to the nearest access point.
    Requires 'rtree' package for geopandas.sjoin_nearest.
    """
    print(f"Calculating minimum distances to {len(access_points_gdf)} access points...")
    if trails_gdf is None or trails_gdf.empty or access_points_gdf is None or access_points_gdf.empty:
        print("Input GDFs empty or None, cannot calculate distances.")
        return None # Return None if inputs are invalid

    # Ensure both GeoDataFrames have the same CRS
    if trails_gdf.crs != access_points_gdf.crs:
        print(f"Error: CRS mismatch between trails ({trails_gdf.crs}) and access points ({access_points_gdf.crs}).")
        return None

    # Use trail centroids for distance calculation
    trails_centroids = trails_gdf.copy()
    trails_centroids['geometry'] = trails_centroids.geometry.centroid

    # Find the nearest access point for each trail centroid
    # sjoin_nearest requires GeoPandas >= 0.10.0 and rtree
    try:
        # Perform spatial join to find the nearest access point geometry for each trail centroid
        joined_gdf = gpd.sjoin_nearest(trails_centroids[[id_col, 'geometry']], access_points_gdf[['geometry']], how='left')
        # Calculate the distance between the trail centroid and its nearest access point geometry
        # Note: joined_gdf index might not align perfectly with trails_centroids if some trails have no nearby access points
        # We need to calculate distance based on the matched geometries
        distances = joined_gdf.geometry.distance(
            gpd.GeoSeries(access_points_gdf.loc[joined_gdf['index_right'], 'geometry'].values, index=joined_gdf.index)
        )
        distances.name = "distance" # Name the series

    except ValueError as ve:
         if "Input shapes do not overlap" in str(ve):
             print("Warning: Trail centroids and access points do not overlap spatially. Distances cannot be calculated.")
             return None
         else:
             print(f"Error during sjoin_nearest/distance calculation: {ve}")
             print("Ensure 'rtree' is installed and GeoPandas is up to date.")
             return None
    except Exception as e:
        print(f"Error during sjoin_nearest/distance calculation: {e}")
        print("Ensure 'rtree' is installed and GeoPandas is up to date.")
        return None

    # The result might have multiple rows per trail if multiple access points are equidistant.
    # We only need the minimum distance for each original trail.
    # Add the calculated distances back to the joined_gdf
    joined_gdf['distance'] = distances

    # Group by original trail ID (from the left side of the join) and get the minimum distance
    min_distances = joined_gdf.groupby(id_col)['distance'].min()

    print(f"Finished calculating minimum distances for {len(min_distances)} trails.")
    return min_distances.rename("min_distance") # Return a Pandas Series

def rank_trails(stats_df, id_col, length_col, rank_by_col, rank_col_name, net_gain_col, min_length=None, absolute_min_net_gain=None, ascending=False):
    """ Filters (by length, net gain) and ranks a DataFrame of trail statistics. """
    print("\n--- Step 5: Filtering (Length/Gain) and Ranking Trails ---") # Updated step number
    if stats_df is None or stats_df.empty: print("Input stats empty."); return pd.DataFrame()
    required_cols = [id_col, length_col, rank_by_col]
    if absolute_min_net_gain is not None and absolute_min_net_gain > 0: required_cols.append(net_gain_col)
    # Add max elevation to required if it's the ranking column
    if rank_by_col == MAX_ELEVATION_COLUMN: required_cols.append(MAX_ELEVATION_COLUMN)

    missing_cols = [col for col in required_cols if col not in stats_df.columns]
    if missing_cols: print(f"Error: Input missing required columns for ranking/filtering: {missing_cols}"); return None

    df_processed = stats_df.copy(); original_rows = len(df_processed)
    print(f"Starting with {original_rows} trails for ranking/filtering.")
    print("Cleaning data...")
    # Include MAX_ELEVATION_COLUMN in numeric checks if it exists
    numeric_cols = [col for col in [length_col, rank_by_col, net_gain_col, MAX_ELEVATION_COLUMN] if col in df_processed.columns]
    for col in numeric_cols: df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    rows_before_nan = len(df_processed); df_processed.dropna(subset=numeric_cols, inplace=True); rows_after_nan = len(df_processed)
    if rows_after_nan < rows_before_nan: print(f"  Dropped {rows_before_nan - rows_after_nan} rows with NaN values.")
    if df_processed.empty: print("No valid data after cleaning."); return pd.DataFrame()

    print("Applying length/gain filters...")
    if min_length is not None and min_length > 0:
        mask = df_processed[length_col] >= min_length; print(f"  Length >= {min_length}m: Keeping {mask.sum()}/{len(df_processed)}"); df_processed = df_processed[mask]
    if absolute_min_net_gain is not None and absolute_min_net_gain > 0:
        mask = df_processed[net_gain_col].abs() >= absolute_min_net_gain; print(f"  Abs Net Gain >= {absolute_min_net_gain}m: Keeping {mask.sum()}/{len(df_processed)}"); df_processed = df_processed[mask]

    rows_after_filtering = len(df_processed)
    print(f"Rows remaining after length/gain filters: {rows_after_filtering}")
    if df_processed.empty: print("No data remaining after length/gain filters."); return pd.DataFrame()

    print(f"Ranking trails by '{rank_by_col}' (ascending={ascending})...")
    ranked_df = df_processed.sort_values(by=rank_by_col, ascending=ascending)
    print(f"Adding rank column '{rank_col_name}'")
    ranked_df[rank_col_name] = range(1, len(ranked_df) + 1)
    print(f"Ranking complete. Final ranked DataFrame has {len(ranked_df)} rows.")
    return ranked_df

def create_top_trails_details_df(top_trails_merged_gdf, n_top, id_col, rank_col, score_col, length_col, center_lat, center_lon, optional_osm_tags, other_tags_col, dist_cols):
    """ Creates a detailed DataFrame for the top N trails, including distances and optional/other tags. """
    print(f"\n--- Step 7: Creating Detailed Output DataFrame for Top {n_top} ---") # Updated step number
    if top_trails_merged_gdf is None or top_trails_merged_gdf.empty: print("Input GDF empty."); return pd.DataFrame()

    details = []
    # Add MAX_ELEVATION_COLUMN to base stats
    base_stat_cols = [id_col, rank_col, score_col, length_col, 'start_elev_m', 'end_elev_m', NET_GAIN_COLUMN, MAX_ELEVATION_COLUMN] + dist_cols
    if other_tags_col and other_tags_col in top_trails_merged_gdf.columns:
        base_stat_cols.append(other_tags_col)

    available_base_cols = [col for col in base_stat_cols if col in top_trails_merged_gdf.columns]
    available_osm_tags = [col for col in optional_osm_tags if col in top_trails_merged_gdf.columns] # Explicitly requested tags
    missing_base = [col for col in base_stat_cols if col not in available_base_cols]
    missing_osm = [col for col in optional_osm_tags if col not in available_osm_tags]
    if missing_base: print(f"Warning: Missing expected base stat/dist/other columns: {missing_base}.")
    if missing_osm: print(f"Info: Optional OSM tags (dedicated cols) not found/requested: {missing_osm}.")

    geom_col = top_trails_merged_gdf.geometry.name
    if geom_col not in top_trails_merged_gdf.columns: print("Error: Geometry column missing."); return pd.DataFrame()

    geolocator = Nominatim(user_agent="trail_detail_reverse_geocode_v6", timeout=10) # Updated agent
    print("Performing reverse geocoding and distance from center calculation...")
    center_coords = (center_lat, center_lon)
    try: gdf_wgs84 = top_trails_merged_gdf[[geom_col]].to_crs("EPSG:4326")
    except Exception as e: print(f"Error reprojecting to EPSG:4326: {e}"); gdf_wgs84 = None

    processed_count = 0
    for index, row in top_trails_merged_gdf.iterrows():
        # Start with base stats (incl distances, other_tags dict, max_elevation)
        trail_data = {col: row.get(col) for col in available_base_cols}
        # Add explicitly requested optional tags
        trail_data.update({col: row.get(col) for col in available_osm_tags})

        lat, lon, placename, distance_center_km = np.nan, np.nan, "Geocoding Failed", np.nan

        if gdf_wgs84 is not None:
            try: geom_wgs84 = gdf_wgs84.loc[index, geom_col]
            except KeyError: geom_wgs84 = None
            if isinstance(geom_wgs84, LineString) and not geom_wgs84.is_empty:
                centroid = geom_wgs84.centroid; lat, lon = centroid.y, centroid.x
                trail_coords = (lat, lon)
                if all(pd.notna(c) for c in [lat, lon, center_lat, center_lon]):
                    try: distance_center_km = geopy_distance(center_coords, trail_coords).km
                    except Exception as dist_e: print(f"Warning: Center dist calc error for {trail_data.get(id_col)}: {dist_e}")
                if pd.notna(lat) and pd.notna(lon):
                    try:
                        location = geolocator.reverse(f"{lat:.6f}, {lon:.6f}", exactly_one=True, language='en')
                        placename = location.address if location else "Placename not found"
                    except (GeocoderTimedOut, GeocoderServiceError) as geo_e: placename = f"Geocoding Error: {type(geo_e).__name__}"
                    except Exception as e: placename = f"Geocoding Error: Other"
                    finally: time.sleep(1.1) # Nominatim usage policy
                else: placename = "Invalid Coords for Geocoding"
            else: placename = "Invalid Geometry for Geocoding"
        else: placename = "Reprojection Failed"

        trail_data.update({'representative_latitude': lat, 'representative_longitude': lon, 'nearest_placename': placename, 'distance_from_center_km': distance_center_km})
        details.append(trail_data)
        processed_count += 1
        if processed_count % 5 == 0: print(f"  ...processed {processed_count}/{len(top_trails_merged_gdf)} trails for details.")

    details_df = pd.DataFrame(details)
    print("Finished creating details DataFrame.")

    # Add MAX_ELEVATION_COLUMN to final output order
    final_ordered_cols = [id_col, rank_col, score_col]
    if DIST_CARPARK_COL in details_df.columns: final_ordered_cols.append(DIST_CARPARK_COL)
    if DIST_ROAD_COL in details_df.columns: final_ordered_cols.append(DIST_ROAD_COL)
    final_ordered_cols.extend(['representative_latitude', 'representative_longitude', 'nearest_placename', 'distance_from_center_km', length_col, NET_GAIN_COLUMN, MAX_ELEVATION_COLUMN]) # Added here
    final_ordered_cols.extend([col for col in optional_osm_tags if col in details_df.columns]) # Add explicit OSM tags
    if other_tags_col and other_tags_col in details_df.columns: # Add the other_tags dict column
        final_ordered_cols.append(other_tags_col)
    # Add remaining base cols (excluding other_tags again and cols already added)
    final_ordered_cols.extend([col for col in available_base_cols if col not in final_ordered_cols])
    final_ordered_cols.extend([col for col in details_df.columns if col not in final_ordered_cols]) # Add any others
    final_ordered_cols = [col for col in final_ordered_cols if col in details_df.columns] # Ensure all columns exist
    details_df = details_df[final_ordered_cols]

    # Fill NaNs in explicitly requested OSM tag columns
    osm_tag_cols_to_fill = [tag for tag in optional_osm_tags if tag in details_df.columns]
    if osm_tag_cols_to_fill: details_df[osm_tag_cols_to_fill] = details_df[osm_tag_cols_to_fill].fillna('N/A')
    # Fill NaNs in distance columns
    for dist_col in dist_cols:
        if dist_col in details_df.columns:
            details_df[dist_col] = details_df[dist_col].fillna(-1).round(1) # Fill NaN and round
    # Fill NaNs in max elevation column
    if MAX_ELEVATION_COLUMN in details_df.columns:
        details_df[MAX_ELEVATION_COLUMN] = details_df[MAX_ELEVATION_COLUMN].fillna(-9999).round(1) # Use a distinct fill value
    # Ensure 'other_tags' column exists and fill NaNs with empty dicts before saving
    if other_tags_col and other_tags_col in details_df.columns:
         details_df[other_tags_col] = details_df[other_tags_col].apply(lambda x: {} if pd.isna(x) else x)

    return details_df

def visualize_results(top_trails_gdf, n_top, place_name, rank_col, score_col_name, vis_cmap, output_png_path):
    """ Creates and saves a map visualization of the top N ranked trails with rank labels. """
    print(f"\n--- Step 8: Visualizing Top Trails ---") # Updated step number
    if top_trails_gdf is None or top_trails_gdf.empty: print("Input GDF empty."); return
    if rank_col not in top_trails_gdf.columns: print(f"Error: Rank column '{rank_col}' missing."); return
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        # Plot trails, colored by rank
        top_trails_gdf.plot(column=rank_col, cmap=vis_cmap, ax=ax, legend=False, linewidth=2.5, aspect='equal')
        # Create a colorbar
        min_r, max_r = top_trails_gdf[rank_col].min(), top_trails_gdf[rank_col].max()
        norm = Normalize(vmin=min_r - 0.5, vmax=max_r + 0.5) if min_r == max_r else Normalize(vmin=min_r, vmax=max_r)
        sm = ScalarMappable(cmap=vis_cmap, norm=norm); sm.set_array([])
        fig.colorbar(sm, ax=ax, label=f"Trail Rank (Top {n_top}, 1=Best)", shrink=0.6)
        print("Adding basemap...")
        try: cx.add_basemap(ax, crs=top_trails_gdf.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik, zoom='auto'); print("Basemap added.")
        except Exception as e: print(f"Warning: Basemap failed: {e}.")
        title_score = score_col_name.replace('_', ' ').replace('degrees', 'deg').title()
        ax.set_title(f"Top {n_top} Trails near {place_name}\n(Ranked by: {title_score})", fontsize=14); ax.set_axis_off()

        # --- Modification: Add Rank Labels ---
        print("Adding rank labels to map...")
        # Define text properties
        text_kwargs = dict(ha='center', va='center', fontsize=8, color='black',
                           path_effects=[PathEffects.withStroke(linewidth=2, foreground='white')]) # White outline

        for idx, row in top_trails_gdf.iterrows():
            rank_label = str(row[rank_col])
            # Use representative_point() which is guaranteed to be on or within the geometry
            label_point = row.geometry.representative_point()
            # Add a small offset if desired (might be needed if points overlap lines too much)
            # label_x, label_y = label_point.x + offset_x, label_point.y + offset_y
            label_x, label_y = label_point.x, label_point.y
            ax.text(label_x, label_y, rank_label, **text_kwargs)
        print("Rank labels added.")
        # --- End Modification ---

        try: # Set map limits
             minx, miny, maxx, maxy = top_trails_gdf.total_bounds
             if np.all(np.isfinite([minx, miny, maxx, maxy])):
                  x_pad, y_pad = (maxx - minx) * 0.05 or 1000, (maxy - miny) * 0.05 or 1000
                  ax.set_xlim(minx - x_pad, maxx + x_pad); ax.set_ylim(miny - y_pad, maxy + y_pad)
             else: print("Warning: Invalid bounds.")
        except Exception as e: print(f"Warning: Could not set bounds: {e}")
        plt.tight_layout()
        print(f"Saving visualization to: {output_png_path}")
        plt.savefig(output_png_path, dpi=300, bbox_inches='tight'); print("Visualization saved."); plt.close(fig)
    except Exception as e:
        print(f"Error during visualization: {e}")
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)

# =============================================================================
# --- Main Execution Workflow ---
# =============================================================================

if __name__ == "__main__":

    # --- 1. Geocoding and Bounding Box ---
    print(f"\n--- Step 1: Geocoding and Bounding Box ---")
    print(f"Finding coordinates for '{PLACE_NAME}'...")
    geolocator = Nominatim(user_agent="consolidated_trail_request_v6") # Updated agent
    center_lat, center_lon = None, None
    try:
        location = geolocator.geocode(PLACE_NAME, timeout=15)
        if not location: print(f"Error: Could not geocode '{PLACE_NAME}'."); sys.exit(1)
        center_lat, center_lon = location.latitude, location.longitude
        print(f"Coordinates found: Lat={center_lat:.4f}, Lon={center_lon:.4f}")
        bounds_ll = get_bounding_box(center_lat, center_lon, RADIUS_KM)
        print(f"Calculated Bounding Box (WGS84): {bounds_ll}")
    except (GeocoderTimedOut, GeocoderServiceError) as e: print(f"Geocoding error: {e}."); sys.exit(1)
    except Exception as e: print(f"Unexpected geocoding/bbox error: {e}"); sys.exit(1)

    # --- 2. Extract Trail Geometries & Tags ---
    print(f"\n--- Step 2: Extract Trail Geometries & Tags ---")
    trails_gdf_raw = extract_osm_features(
        PBF_FILE_PATH, bounds_ll, TRAIL_TAGS,
        PATHS["temp_trails_geojson"], "trails", add_id=True
    )
    # Standardize, reproject, and consolidate tags
    trails_gdf_processed = standardize_and_reproject(
        trails_gdf_raw, TARGET_CRS, id_column=ID_COLUMN, expected_geojson_id_col='id',
        optional_tags_as_cols=OPTIONAL_OSM_TAGS, # Pass optional tags to keep separate
        other_tags_col_name=OTHER_TAGS_COLUMN_NAME # Pass name for the dict column
    )
    if trails_gdf_processed is None: print("Trail extraction/processing failed."); sys.exit(1)
    if trails_gdf_processed.empty: print("No trails matching initial highway criteria found."); sys.exit(0)
    print(f"Successfully extracted and processed {len(trails_gdf_processed)} potential trails.")
    print(f"Columns after processing: {trails_gdf_processed.columns.tolist()}") # Check for other_tags col

    # --- 2b. Apply Tag Exclusion Filters ---
    print(f"\n--- Step 2b: Applying Tag Exclusion Filters ---")
    filtered_trails_gdf = trails_gdf_processed.copy()
    initial_count = len(filtered_trails_gdf)
    print(f"Starting with {initial_count} trails before tag exclusion.")
    for tag_key, excluded_values in EXCLUDE_TAG_VALUES.items():
        if tag_key in filtered_trails_gdf.columns: # Filter only on explicitly kept columns
            excluded_values_lower = [str(v).lower() for v in excluded_values]
            mask = ~filtered_trails_gdf[tag_key].fillna('---nan---').astype(str).str.lower().isin(excluded_values_lower)
            rows_before = len(filtered_trails_gdf); filtered_trails_gdf = filtered_trails_gdf[mask]; rows_removed = rows_before - len(filtered_trails_gdf)
            if rows_removed > 0: print(f"  Filtered '{tag_key}': Removed {rows_removed} trails (values: {excluded_values}).")
        # Note: Filtering based on keys within the OTHER_TAGS_COLUMN_NAME dictionary is not implemented here
        #       but could be added if needed (would require iterating through the dicts).
        elif tag_key != 'area': # Don't warn about 'area' if it was consolidated
             print(f"  Skipping filter for '{tag_key}': Column not found (may be in '{OTHER_TAGS_COLUMN_NAME}').")
    final_count = len(filtered_trails_gdf)
    print(f"Finished tag exclusion. Kept {final_count}/{initial_count} trails.")
    if filtered_trails_gdf.empty: print("No trails remaining after tag exclusion."); sys.exit(0)
    analysis_input_gdf = filtered_trails_gdf # Use this for subsequent steps

    # --- 3. Analyze Trail Elevations ---
    stats_df = analyze_elevations(analysis_input_gdf, PROJECTED_DEM_PATH, id_column=ID_COLUMN)
    if stats_df is None or stats_df.empty: print("Elevation analysis failed or yielded no results."); sys.exit(1)

    # --- 4. Calculate Distances to Access Points ---
    dist_cols_added = []
    if CALCULATE_ACCESS_DISTANCE:
        print(f"\n--- Step 4: Calculate Distances to Access Points ---")
        # --- 4a. Extract Car Parks ---
        carpark_tags = {"amenity": ["parking"]}
        carparks_gdf_raw = extract_osm_features(PBF_FILE_PATH, bounds_ll, carpark_tags, PATHS["temp_access_geojson"], "car parks", add_id=False)
        carparks_gdf = standardize_and_reproject(carparks_gdf_raw, TARGET_CRS, id_column=None)
        if carparks_gdf is not None and not carparks_gdf.empty:
            print(f"Calculating distance to {len(carparks_gdf)} car parks...")
            min_dist_carparks = calculate_min_distances(analysis_input_gdf[[ID_COLUMN, 'geometry']], carparks_gdf, ID_COLUMN)
            if min_dist_carparks is not None:
                stats_df = stats_df.merge(min_dist_carparks.rename(DIST_CARPARK_COL), on=ID_COLUMN, how='left')
                dist_cols_added.append(DIST_CARPARK_COL); print(f"Added '{DIST_CARPARK_COL}' column.")
            else: print("Failed to calculate car park distances.")
        else: print("No car parks found/extracted.")

        # --- 4b. Extract Access Roads ---
        road_tags = {"highway": ACCESS_ROAD_HIGHWAY_VALUES}
        roads_gdf_raw = extract_osm_features(PBF_FILE_PATH, bounds_ll, road_tags, PATHS["temp_access_geojson"], "access roads", add_id=False)
        roads_gdf = standardize_and_reproject(roads_gdf_raw, TARGET_CRS, id_column=None)
        if roads_gdf is not None and not roads_gdf.empty:
            roads_gdf = roads_gdf[roads_gdf.geometry.geom_type.isin(['LineString', 'MultiLineString'])] # Ensure lines
            if not roads_gdf.empty:
                print(f"Calculating distance to {len(roads_gdf)} access roads...")
                min_dist_roads = calculate_min_distances(analysis_input_gdf[[ID_COLUMN, 'geometry']], roads_gdf, ID_COLUMN)
                if min_dist_roads is not None:
                    stats_df = stats_df.merge(min_dist_roads.rename(DIST_ROAD_COL), on=ID_COLUMN, how='left')
                    dist_cols_added.append(DIST_ROAD_COL); print(f"Added '{DIST_ROAD_COL}' column.")
                else: print("Failed to calculate road distances.")
            else: print("No valid LineString access roads found.")
        else: print("No access roads found/extracted.")
        try: # Cleanup temp geojson
            if os.path.exists(PATHS["temp_access_geojson"]): os.remove(PATHS["temp_access_geojson"])
        except OSError as e: print(f"Warning: Could not remove temp access geojson: {e}")
    else: print("\n--- Step 4: Skipped - Distance calculation disabled ---")

    # --- Merge back OSM tags (incl. other_tags dict) before saving raw stats ---
    print("\n--- Merging Tags and Distances for Raw Stats File ---")
    # Define columns to merge from the filtered GDF (explicit tags + other_tags dict)
    cols_to_merge_from_gdf = [ID_COLUMN] + [tag for tag in OPTIONAL_OSM_TAGS if tag in analysis_input_gdf.columns]
    if OTHER_TAGS_COLUMN_NAME in analysis_input_gdf.columns:
        cols_to_merge_from_gdf.append(OTHER_TAGS_COLUMN_NAME)

    stats_df_merged = stats_df.copy() # Start with stats (which might have distance cols)
    if ID_COLUMN not in analysis_input_gdf.columns: print(f"CRITICAL ERROR: ID column missing."); sys.exit(1)
    if len(cols_to_merge_from_gdf) > 1:
         merge_tags_list = [c for c in cols_to_merge_from_gdf if c != ID_COLUMN]
         print(f"Merging OSM tags ({', '.join(merge_tags_list)})...")
         stats_df_merged[ID_COLUMN] = stats_df_merged[ID_COLUMN].astype(str)
         analysis_input_gdf[ID_COLUMN] = analysis_input_gdf[ID_COLUMN].astype(str)
         tags_only_gdf = analysis_input_gdf[cols_to_merge_from_gdf]
         stats_df_merged = pd.merge(stats_df_merged, tags_only_gdf, on=ID_COLUMN, how='left', suffixes=('', '_tag_dup'))
         dup_cols = [c for c in stats_df_merged if c.endswith('_tag_dup')]
         if dup_cols: print(f"Warning: Dropping duplicate tag columns: {dup_cols}"); stats_df_merged = stats_df_merged.drop(columns=dup_cols)
    else: print("No optional/other OSM tags found in filtered data to merge.")
    try: # Save raw stats
        print(f"Saving raw statistics (incl. distances & tags) to: {PATHS['stats_csv']}")
        raw_osm_cols = [tag for tag in OPTIONAL_OSM_TAGS if tag in stats_df_merged.columns]
        if raw_osm_cols: stats_df_merged[raw_osm_cols] = stats_df_merged[raw_osm_cols].fillna('N/A')
        if OTHER_TAGS_COLUMN_NAME in stats_df_merged.columns: # Ensure other_tags exists and fill NaNs
             stats_df_merged[OTHER_TAGS_COLUMN_NAME] = stats_df_merged[OTHER_TAGS_COLUMN_NAME].apply(lambda x: {} if pd.isna(x) else x)
        for dist_col in dist_cols_added: # Fill NaN distances
            if dist_col in stats_df_merged.columns: stats_df_merged[dist_col] = stats_df_merged[dist_col].fillna(-1)
        # Fill NaN max elevation
        if MAX_ELEVATION_COLUMN in stats_df_merged.columns:
             stats_df_merged[MAX_ELEVATION_COLUMN] = stats_df_merged[MAX_ELEVATION_COLUMN].fillna(-9999)
        # Convert dict column to string for CSV saving
        if OTHER_TAGS_COLUMN_NAME in stats_df_merged.columns:
            stats_df_merged[OTHER_TAGS_COLUMN_NAME] = stats_df_merged[OTHER_TAGS_COLUMN_NAME].astype(str)
        stats_df_merged.to_csv(PATHS['stats_csv'], index=False, float_format='%.4f')
        print("Raw statistics saved.")
    except Exception as e: print(f"Warning: Could not save raw statistics CSV: {e}")

    # --- 5. Rank Trails ---
    ranked_stats_df = rank_trails(
        stats_df, # Rank based on stats_df (which has distances but not tags dict)
        id_col=ID_COLUMN, length_col='length_m', rank_by_col=RANKING_COLUMN,
        rank_col_name=RANK_COLUMN_NAME, net_gain_col=NET_GAIN_COLUMN,
        min_length=MIN_TRAIL_LENGTH_FILTER, absolute_min_net_gain=ABSOLUTE_MIN_NET_GAIN,
        ascending=(not RANK_DESCENDING)
    )
    if ranked_stats_df is None or ranked_stats_df.empty: print("Ranking/filtering failed or yielded no results."); sys.exit(1)
    try: # Save ranked stats
        print(f"Saving ranked statistics to: {PATHS['ranked_stats_csv']}")
        ranked_stats_df.to_csv(PATHS['ranked_stats_csv'], index=False, float_format='%.4f')
        print("Ranked statistics saved.")
    except Exception as e: print(f"Warning: Could not save ranked statistics CSV: {e}")

    # --- 6. Prepare Data for Top N Trails ---
    print(f"\n--- Step 6: Preparing Data for Top {N_TOP_TRAILS} ---")
    if RANK_COLUMN_NAME not in ranked_stats_df.columns: print("Error: Rank column missing."); sys.exit(1)
    top_ranked_ids = ranked_stats_df.head(N_TOP_TRAILS)[ID_COLUMN].tolist()
    # Filter the GDF that has Geometries AND all Tags (analysis_input_gdf)
    if ID_COLUMN not in analysis_input_gdf.columns: print("Error: ID column missing from filtered GDF."); sys.exit(1)
    analysis_input_gdf[ID_COLUMN] = analysis_input_gdf[ID_COLUMN].astype(str)
    top_trails_geom_tags_gdf = analysis_input_gdf[analysis_input_gdf[ID_COLUMN].isin(top_ranked_ids)].copy()

    print("Merging ranked statistics (incl. distances) onto top trail geometries/tags...")
    ranked_stats_df[ID_COLUMN] = ranked_stats_df[ID_COLUMN].astype(str)
    # Merge the ranked data (which includes rank and possibly distances) onto the geometry/tag data
    top_trails_merged_gdf = top_trails_geom_tags_gdf.merge(
        ranked_stats_df, on=ID_COLUMN, how='left', suffixes=('', '_dup_stat')
    )
    dup_cols = [c for c in top_trails_merged_gdf if c.endswith('_dup_stat')]
    if dup_cols: print(f"Warning: Dropping duplicate stat columns: {dup_cols}"); top_trails_merged_gdf = top_trails_merged_gdf.drop(columns=dup_cols)
    top_trails_merged_gdf = top_trails_merged_gdf.dropna(subset=[RANK_COLUMN_NAME])
    top_trails_merged_gdf[RANK_COLUMN_NAME] = top_trails_merged_gdf[RANK_COLUMN_NAME].astype(int)
    top_trails_merged_gdf = top_trails_merged_gdf[top_trails_merged_gdf[RANK_COLUMN_NAME] <= N_TOP_TRAILS]
    top_trails_merged_gdf = top_trails_merged_gdf.sort_values(by=RANK_COLUMN_NAME)

    if top_trails_merged_gdf.empty: print(f"Could not merge data for top {N_TOP_TRAILS} trails."); sys.exit(1)
    print(f"Prepared merged GDF for top {len(top_trails_merged_gdf)} trails.")
    print(f"Columns available for details DF: {top_trails_merged_gdf.columns.tolist()}")

    # --- 7. Create Detailed DataFrame for Top Trails ---
    top_details_df = create_top_trails_details_df(
        top_trails_merged_gdf, n_top=N_TOP_TRAILS, id_col=ID_COLUMN,
        rank_col=RANK_COLUMN_NAME, score_col=RANKING_COLUMN, length_col='length_m',
        center_lat=center_lat, center_lon=center_lon,
        optional_osm_tags=OPTIONAL_OSM_TAGS, # Explicitly requested tag columns
        other_tags_col=OTHER_TAGS_COLUMN_NAME, # Name of the dict column
        dist_cols=dist_cols_added # Names of distance columns added
    )
    if top_details_df is not None and not top_details_df.empty:
        print("\n--- Top Trail Details ---")
        # Convert other_tags dict to string for display if it exists
        display_df = top_details_df.copy()
        if OTHER_TAGS_COLUMN_NAME in display_df.columns:
            # Use json.dumps for potentially more readable string representation
            display_df[OTHER_TAGS_COLUMN_NAME] = display_df[OTHER_TAGS_COLUMN_NAME].apply(json.dumps)

        with pd.option_context('display.max_rows', N_TOP_TRAILS + 5, 'display.max_columns', None, 'display.width', 220, 'display.precision', 4):
            # Add MAX_ELEVATION_COLUMN to display list
            display_cols_core = [ID_COLUMN, RANK_COLUMN_NAME, RANKING_COLUMN]
            if DIST_CARPARK_COL in display_df.columns: display_cols_core.append(DIST_CARPARK_COL)
            if DIST_ROAD_COL in display_df.columns: display_cols_core.append(DIST_ROAD_COL)
            display_cols_core.extend(['nearest_placename', 'distance_from_center_km', 'length_m', NET_GAIN_COLUMN, MAX_ELEVATION_COLUMN]) # Added here
            display_cols_osm = [tag for tag in OPTIONAL_OSM_TAGS] # Explicit tags
            display_cols_other = [OTHER_TAGS_COLUMN_NAME] if OTHER_TAGS_COLUMN_NAME in display_df.columns else [] # Other tags dict
            display_cols_coords = ['representative_latitude', 'representative_longitude']
            desired_display_cols = display_cols_core + display_cols_osm + display_cols_other + display_cols_coords
            display_cols = [col for col in desired_display_cols if col in display_df.columns] # Filter to existing
            print(f"Displaying columns: {display_cols}")
            print(display_df[display_cols]) # Display selected columns
        try: # Save detailed CSV
            print(f"\nSaving detailed top trail data to: {PATHS['top_details_csv']}")
            # Convert dict column back to string using json.dumps for proper CSV storage if not already string
            if OTHER_TAGS_COLUMN_NAME in top_details_df.columns and not pd.api.types.is_string_dtype(top_details_df[OTHER_TAGS_COLUMN_NAME]):
                 top_details_df[OTHER_TAGS_COLUMN_NAME] = top_details_df[OTHER_TAGS_COLUMN_NAME].apply(json.dumps)
            top_details_df.to_csv(PATHS['top_details_csv'], index=False, float_format='%.6f')
            print("Detailed top trail data saved.")
        except Exception as e: print(f"Warning: Could not save detailed CSV: {e}")
    else: print("Failed to create detailed DataFrame for top trails.")

    # --- 8. Visualize Top Trails ---
    visualize_results(
        top_trails_merged_gdf, n_top=N_TOP_TRAILS, place_name=PLACE_NAME,
        rank_col=RANK_COLUMN_NAME, score_col_name=RANKING_COLUMN,
        vis_cmap=VIS_COLORMAP, output_png_path=PATHS["visualization_png"]
    )

    print(f"\n--- Workflow Finished ---")
    print(f"End time: {np.datetime64('now', 's')} UTC")
    print(f"Outputs saved in: {PATHS['output_dir']}")
