#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidated script for finding, analyzing, ranking, and visualizing steep trails
based on OSM data, a DEM raster, and a Land Cover Map (LCM) raster.

Version 12.1 - Added LCM Alignment:
- Takes user-specified LCM_PATH as input.
- Checks if the input LCM needs reprojection/alignment to the TARGET_CRS and AOI bounds.
- If needed, creates a temporary aligned LCM raster in the output directory.
- Uses the correctly aligned LCM (original or temporary) for suitability analysis.
- Integrated land cover suitability score from an LCM raster.
- Added new RANKING_STRATEGY: 'LAND_COVER_LENGTH_SCORE'.
    - Calculates average land cover suitability for each trail.
    - Ranks trails by: trail_length * average_land_cover_suitability.
- Requires an LCM raster path and LAND_COVER_SUITABILITY dictionary in config.
- Output directory and map titles dynamically reflect the chosen ranking strategy.
- All relevant score columns are included in CSV outputs.
- Fixed geometry check before visualization.
- Added diagnostic print for CSV columns.

Based on Streamlined Trail Search v11.4 & v12.0.
Incorporates concepts from Cross_score_steepness_and_terrain_v3 for land cover alignment.

Requires: osmium-tool, geopandas, rasterio, pandas, numpy, matplotlib, contextily, geopy, time, rtree, shapely, pyproj
"""

import os
import sys
import warnings
import subprocess
import shutil
import time
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Point, LineString, MultiPoint, MultiLineString, Polygon, MultiPolygon
from shapely.ops import transform as shapely_transform # For reprojecting single geometries if needed
from functools import partial # For shapely_transform
import pyproj # For shapely_transform
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from geopy.distance import distance as geopy_distance
import math
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import contextily as cx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import json
import traceback

# Suppress specific warnings if needed (e.g., from Shapely or Geopandas)
warnings.filterwarnings('ignore', message='.*Sequential read of iterator was interrupted.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered in cast.*')


# Check for rtree dependency
try:
    import rtree
except ImportError:
    print("Error: The 'rtree' library is required. Install with 'pip install rtree'.")
    sys.exit(1)

# =============================================================================
# --- Configuration ---
# =============================================================================

# --- 1. Search Area ---
#PLACE_NAME = "Ladybower Reservoir, UK"
#PLACE_NAME = "Grimsby, UK"
PLACE_NAME = "Keswick, UK"
RADIUS_KM = 20 # Reduced for potentially faster LCM processing during testing

# --- 2. Input Data Paths ---
PBF_FILE_PATH = "/home/ajay/Python_Projects/steepest_trails_search/united-kingdom-latest.osm.pbf"
PROJECTED_DEM_PATH = "/home/ajay/Python_Projects/steepest_trails_search/outputs_united_kingdom_dem/united_kingdom_dem_proj_27700.tif"
# !!! USER PROVIDED LCM Path !!!
# The script will check its CRS and align it if necessary.
LCM_PATH = "/home/ajay/Python_Projects/steepest_trails_search/LCM_data_and_docs/data/gblcm2023_10m.tif" # USER PROVIDED PATH - Corrected based on user feedback

# --- 3. Coordinate Reference System (CRS) ---
TARGET_CRS = "EPSG:27700" # British National Grid
SOURCE_CRS_GPS = "EPSG:4326" # WGS84 for geocoding

# --- 4. Trail Identification & Tag Extraction ---
TRAIL_TAGS = { "highway": ["bridleway", "track", "path", "footway"] }
OPTIONAL_OSM_TAGS = [
    'highway', 'name', 'surface', 'sac_scale', 'trail_visibility', 'access',
    'designation', 'tracktype', 'smoothness'
]
ID_COLUMN = 'osmid'
OTHER_TAGS_COLUMN_NAME = 'other_osm_tags'

# --- 5. Trail Filtering Criteria ---
EXCLUDE_TAG_VALUES = {
    "surface": ["paved", "asphalt", "concrete", "sett"],
    "highway": ["motorway", "trunk", "primary", "secondary", "tertiary", "residential", "service", "steps", "construction"],
    "access": ["no", "private"],
    "area": ["yes"] # Exclude ways tagged as areas
}
MIN_TRAIL_LENGTH_FILTER = 400 # meters
ABSOLUTE_MIN_NET_GAIN = 0 # meters (applied to absolute gain)
NET_GAIN_COLUMN = 'net_gain_m'
MAX_ELEVATION_COLUMN = 'max_elevation_m'

# --- 6. Remoteness / Access Point Proximity ---
CALCULATE_ACCESS_DISTANCE = True
ACCESS_ROAD_HIGHWAY_VALUES = [
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
    "residential", "motorway_link", "trunk_link", "primary_link", "secondary_link",
    "tertiary_link", "living_street", "service", "road"
]
DIST_CARPARK_COL = 'min_dist_carpark_m'
DIST_ROAD_COL = 'min_dist_road_m'

# --- 7. Trail Ranking Criteria ---
# Choose the ranking strategy. Options:
# 'ABSOLUTE_GRADIENT_DEGREES': Ranks by absolute average gradient in degrees.
# 'TARGET_SLOPE_PERCENT_SCORE': Ranks by a score (0-1) based on proximity to TARGET_SLOPE_PERCENT.
# 'LENGTH_ADJUSTED_TARGET_GRADE_SCORE': Ranks by Length * (Target Grade % - L1_Penalty_from_Target_%).
# 'LAND_COVER_LENGTH_SCORE': Ranks by Length * Average Land Cover Suitability. # !!! NEW STRATEGY !!!
RANKING_STRATEGY = 'LAND_COVER_LENGTH_SCORE' # <-- CHOOSE RANKING STRATEGY HERE

# --- 7a. Settings for 'ABSOLUTE_GRADIENT_DEGREES' strategy ---
ABS_GRAD_DEG_COLUMN = 'absolute_avg_gradient_degrees'

# --- 7b. Settings for 'TARGET_SLOPE_PERCENT_SCORE' and 'LENGTH_ADJUSTED_TARGET_GRADE_SCORE' strategies ---
TARGET_SLOPE_PERCENT = 20.0 # Target grade percentage (e.g., 20.0 for 20%).
TARGET_SLOPE_PENALTY_TYPE_FOR_PCT_SCORE = 'linear' # 'quadratic' or 'linear' for TARGET_SLOPE_PERCENT_SCORE.
MAX_SLOPE_PERCENT_FOR_PCT_SCORE_NORMALIZATION = 50.0 # Max slope % for normalizing penalty in the 0-1 score.

# --- 7c. Settings for 'LAND_COVER_LENGTH_SCORE' strategy ---
# !!! Land Cover Suitability Dictionary !!!
# LAND_COVER_SUITABILITY = {
#     # Codes based on LCM2021/2023 documentation (verify if needed)
#     1: 1.0,  2: 1.0,  3: 0.2,  4: 0.9,  5: 1.0,  6: 1.0,  7: 0.8,
#     8: 0.1,  9: 0.6, 10: 0.7, 11: 0.1, 12: 0.0, 13: 0.0, 14: 0.0,
#    15: 0.0, 16: 0.5, 17: 0.0, 18: 0.3, 19: 0.1, 20: 0.0, 21: 0.0,
#     0: 0.0   # Default for Nodata/Unclassified
# }

LAND_COVER_SUITABILITY = {
    # Codes based on LCM2021/2023 documentation (verify if needed)
    1: 1.0,  # Broadleaved Woodland (leaf litter good, roots/branches caution)
    2: 1.0,  # Coniferous Woodland (needles can be sharp)
    # ---- too risky for descents - too exposed
    3: 1.0,  # Arable (often uneven, potentially sharp stubble)
    4: 1.0,  # Improved Grassland (generally good, watch for stones/thistles)
    5: 1.0,  # Neutral Grassland (often excellent)
    6: 1.0,  # Calcareous Grassland (often excellent, maybe flinty)
    7: 1.0,  # Acid Grassland (good, can be tussocky/boggy patches)
    # ---- too risky 
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


DEFAULT_SUITABILITY_SCORE = 0.0 # Score for LCM values not explicitly listed.
LCM_SAMPLING_DISTANCE = 10 # meters (distance between sample points along trail).
LCM_TARGET_RESOLUTION = (10, 10) # Target resolution for aligned LCM (should match source LCM if possible)
LCM_NODATA_VAL = 0 # Nodata value used in the source LCM (used for alignment profile)

# --- 7d. Column Names for Calculated Scores & Final Rank ---
TARGET_SLOPE_PERCENT_SCORE_COLUMN = 'target_slope_pct_score'
LENGTH_ADJUSTED_TARGET_GRADE_SCORE_COLUMN = 'len_adj_target_grade_score'
AVG_LAND_COVER_SUITABILITY_SCORE_COLUMN = 'avg_land_cover_suitability'
LAND_COVER_LENGTH_RANKING_SCORE_COLUMN = 'land_cover_length_score'

RANK_DESCENDING = True # True for descending rank (higher score/gradient is Rank 1)
RANK_COLUMN_NAME = 'rank' # Name for the final rank column in output

# --- 8. Output & Visualization ---
N_TOP_TRAILS = 20 # Number of top trails to detail and visualize
VIS_COLORMAP = 'viridis_r'
OUTPUT_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CTX_PROVIDER = cx.providers.OpenStreetMap.Mapnik

# =============================================================================
# --- END OF Configuration ---
# =============================================================================

print(f"--- Trail Analysis Workflow v12.1 (LCM Alignment Added) ---") # Version updated
print(f"Selected Ranking Strategy: {RANKING_STRATEGY}")
if RANKING_STRATEGY == 'LAND_COVER_LENGTH_SCORE':
    print(f"  Ranking by: Trail Length * Average Land Cover Suitability")
    print(f"  Input LCM Path: {LCM_PATH}") # Print input path
    if not os.path.exists(LCM_PATH):
         print(f"  WARNING: Input LCM Path does not exist: {LCM_PATH}. Land cover scoring will fail.")
    print(f"  LCM Sampling Distance: {LCM_SAMPLING_DISTANCE}m")
# ... (rest of the print statements remain the same) ...
print(f"Start time: {np.datetime64('now', 's')} UTC")


# --- Dynamic Path Generation ---
def generate_paths(base_dir, place_name, radius_km, ranking_strategy_config, target_slope_pct_config, penalty_type_for_pct_score_config):
    """Generates output directory and file paths based on configuration."""
    place_slug = place_name.replace(' ', '_').replace(',', '').lower()
    strategy_slug = "unknown"
    if ranking_strategy_config == 'ABSOLUTE_GRADIENT_DEGREES':
        strategy_slug = "graddeg"
    elif ranking_strategy_config == 'TARGET_SLOPE_PERCENT_SCORE':
        strategy_slug = f"ts{target_slope_pct_config}{penalty_type_for_pct_score_config[0]}_pctscore"
    elif ranking_strategy_config == 'LENGTH_ADJUSTED_TARGET_GRADE_SCORE':
        strategy_slug = f"ts{target_slope_pct_config}_lenadj"
    elif ranking_strategy_config == 'LAND_COVER_LENGTH_SCORE':
        strategy_slug = "lc_len_score"

    output_dir_name = f"outputs_{place_slug}_r{radius_km}km_{strategy_slug}"
    output_dir = os.path.join(base_dir, output_dir_name)
    paths = {
        "output_dir": output_dir,
        "temp_bbox_pbf": os.path.join(output_dir, "temp_bbox_extract.osm.pbf"),
        "temp_trails_pbf": os.path.join(output_dir, "temp_filtered_trails_initial.osm.pbf"),
        "temp_trails_geojson": os.path.join(output_dir, "temp_filtered_trails.geojson"),
        "temp_access_geojson": os.path.join(output_dir, "temp_access_points.geojson"),
        "aligned_lcm_temp": os.path.join(output_dir, f"{place_slug}_aligned_lcm_temp.tif"), # path for temp aligned LCM
        "stats_csv": os.path.join(output_dir, f"{place_slug}_trail_stats_all.csv"),
        "ranked_stats_csv": os.path.join(output_dir, f"{place_slug}_trail_stats_ranked.csv"),
        "top_details_csv": os.path.join(output_dir, f"{place_slug}_top_{N_TOP_TRAILS}_details.csv"),
        "visualization_png": os.path.join(output_dir, f"{place_slug}_top_{N_TOP_TRAILS}_map.png"),
    }
    return paths

PATHS = generate_paths(OUTPUT_BASE_DIR, PLACE_NAME, RADIUS_KM, RANKING_STRATEGY, TARGET_SLOPE_PERCENT, TARGET_SLOPE_PENALTY_TYPE_FOR_PCT_SCORE)
print(f"Ensuring output directory exists: {PATHS['output_dir']}")
os.makedirs(PATHS['output_dir'], exist_ok=True)
if not os.path.isdir(PATHS['output_dir']):
    print(f"FATAL ERROR: Failed to create output directory: {PATHS['output_dir']}"); sys.exit(1)

# --- Helper Functions ---
# get_bounding_box, run_osmium_command, extract_osm_features, standardize_and_reproject remain the same
def get_bounding_box(latitude, longitude, radius_km):
    """ Calculates bounding box (min_lon, min_lat, max_lon, max_lat) in WGS84."""
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180): raise ValueError("Invalid lat/lon.")
    if radius_km <= 0: raise ValueError("Radius must be positive.")
    earth_radius_km = 6371.0
    lat_delta_deg = math.degrees(radius_km / earth_radius_km)
    clamped_lat_rad = math.radians(max(-89.999, min(89.999, latitude)))
    lon_delta_deg = 180.0 if abs(math.cos(clamped_lat_rad)) < 1e-9 else math.degrees(radius_km / (earth_radius_km * math.cos(clamped_lat_rad)))
    min_lat, max_lat = max(-90.0, latitude - lat_delta_deg), min(90.0, latitude + lat_delta_deg)
    min_lon = (longitude - lon_delta_deg + 540) % 360 - 180
    max_lon = (longitude + lon_delta_deg + 540) % 360 - 180
    if min_lon > max_lon: min_lon, max_lon = max_lon, min_lon
    return (min_lon, min_lat, max_lon, max_lat)

def run_osmium_command(cmd, description):
    """Executes an osmium command line tool command."""
    print(f"  Running osmium: {description}...")
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  Error during '{description}':\n{result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  Error: Osmium command '{description}' timed out after 300 seconds.")
        return False
    except Exception as e:
        print(f"  Failed to run osmium command '{description}': {e}")
        return False

def extract_osm_features(pbf_path, bbox_ll, tags_filter, output_geojson_path, description, add_id=True):
    """Extracts OSM features (ways) matching tags within a bounding box using Osmium."""
    print(f"Extracting {description} features...")
    temp_dir = os.path.dirname(output_geojson_path)
    temp_extract_pbf = os.path.join(temp_dir, f"temp_extract_{description.replace(' ', '_')}.osm.pbf")
    temp_filter_pbf = os.path.join(temp_dir, f"temp_filter_{description.replace(' ', '_')}.osm.pbf")

    bbox_str = f"{bbox_ll[0]},{bbox_ll[1]},{bbox_ll[2]},{bbox_ll[3]}"
    extract_cmd = ["osmium", "extract", "-b", bbox_str, pbf_path, "-o", temp_extract_pbf, "--overwrite"]
    if not run_osmium_command(extract_cmd, f"bbox extract for {description}"): return None

    tag_filter_parts = []
    for key, values in tags_filter.items():
        tag_filter_parts.append(f"w/{key}={','.join(values)}")
    filter_cmd = ["osmium", "tags-filter", temp_extract_pbf] + tag_filter_parts + ["-o", temp_filter_pbf, "--overwrite"]
    if not run_osmium_command(filter_cmd, f"tags filter for {description}"):
        if os.path.exists(temp_extract_pbf): os.remove(temp_extract_pbf)
        return None

    export_cmd_base = ["osmium", "export", temp_filter_pbf]
    if add_id: export_cmd_base.extend(["--add-unique-id=type_id"])
    export_cmd = export_cmd_base + ["-o", output_geojson_path, "--overwrite", "--geometry-types=linestring"]
    if not run_osmium_command(export_cmd, f"geojson export for {description}"):
        for f_path in [temp_extract_pbf, temp_filter_pbf, output_geojson_path]:
            if os.path.exists(f_path):
                try: os.remove(f_path)
                except OSError: pass
        return None

    for f_path in [temp_extract_pbf, temp_filter_pbf]:
        try:
            if os.path.exists(f_path): os.remove(f_path)
        except OSError as e: print(f"  Warning: Could not remove temporary PBF '{f_path}': {e}")

    print(f"  Reading GeoJSON for {description}: {os.path.basename(output_geojson_path)}...")
    if os.path.exists(output_geojson_path) and os.path.getsize(output_geojson_path) > 0:
        try:
            gdf = gpd.read_file(output_geojson_path)
            print(f"  Successfully read {len(gdf)} {description} features.")
            initial_geom_count = len(gdf)
            gdf = gdf[gdf.geometry.geom_type == 'LineString']
            if len(gdf) < initial_geom_count:
                 print(f"    Filtered out {initial_geom_count - len(gdf)} non-LineString geometries.")
            return gdf
        except Exception as read_err:
            print(f"  Error reading GeoJSON file for {description}: {read_err}")
            return None
    else:
        print(f"  GeoJSON file for {description} is empty or does not exist.")
        return gpd.GeoDataFrame()


def standardize_and_reproject(gdf, target_crs, id_column=None, expected_geojson_id_col='id',
                              optional_tags_as_cols=None, other_tags_col_name=None):
    """Standardizes ID column, consolidates other tags, and reprojects a GeoDataFrame."""
    if gdf is None or gdf.empty: return gdf
    print(f"Standardizing and reprojecting GeoDataFrame (Initial rows: {len(gdf)})...")

    if id_column and expected_geojson_id_col and expected_geojson_id_col in gdf.columns:
        if id_column != expected_geojson_id_col:
            gdf = gdf.rename(columns={expected_geojson_id_col: id_column})
        try:
            gdf[id_column] = gdf[id_column].astype(str).str.extract(r'(\d+)$', expand=False).fillna(gdf[id_column]).astype(str)
        except Exception as e:
            print(f"  Warning: Could not extract numeric part from ID column '{id_column}': {e}. Keeping original values.")

    if other_tags_col_name:
        cols_to_keep = ['geometry']
        if id_column and id_column in gdf.columns: cols_to_keep.append(id_column)
        if optional_tags_as_cols:
            cols_to_keep.extend([tag for tag in optional_tags_as_cols if tag in gdf.columns])
        cols_to_consolidate = [col for col in gdf.columns if col not in cols_to_keep]
        if cols_to_consolidate:
            gdf[other_tags_col_name] = gdf[cols_to_consolidate].apply(lambda row: row.dropna().to_dict(), axis=1)
            gdf = gdf.drop(columns=cols_to_consolidate)
            print(f"  Consolidated other tags into '{other_tags_col_name}' column.")
        else:
            gdf[other_tags_col_name] = [{} for _ in range(len(gdf))]

    if gdf.crs is None:
        gdf.crs = SOURCE_CRS_GPS
        print(f"  Warning: Input GeoDataFrame has no CRS defined. Assuming {SOURCE_CRS_GPS}.")
    if gdf.crs.to_string().upper() != target_crs.upper():
        try:
            print(f"  Reprojecting from {gdf.crs.to_string()} to {target_crs}...")
            gdf = gdf.to_crs(target_crs)
            print(f"  Reprojection successful.")
        except Exception as e:
            print(f"  Error during reprojection to {target_crs}: {e}")
            traceback.print_exc()
            return None
    else:
        print(f"  GeoDataFrame already in target CRS ({target_crs}).")

    return gdf

# --- Score Calculation Functions ---
# calculate_target_slope_percent_score, calculate_length_adjusted_target_grade_score remain the same
def calculate_target_slope_percent_score(avg_slope_percent_data,
                                         target_slope_pct_config,
                                         max_slope_pct_for_penalty_norm_config,
                                         penalty_type_config,
                                         output_nodata_val=-1.0):
    """Calculates 0-1 score based on proximity to target slope percentage."""
    if max_slope_pct_for_penalty_norm_config <= 0:
        print("    ERROR: max_slope_pct_for_penalty_norm_config must be positive for TARGET_SLOPE_PERCENT_SCORE.")
        return pd.Series(np.full(len(avg_slope_percent_data), output_nodata_val), dtype=np.float32, index=avg_slope_percent_data.index if isinstance(avg_slope_percent_data, pd.Series) else None)

    slope_values = avg_slope_percent_data.values.astype(np.float32) if isinstance(avg_slope_percent_data, pd.Series) else np.array(avg_slope_percent_data, dtype=np.float32)
    mask = np.isnan(slope_values)
    slope_masked = np.ma.array(slope_values, mask=mask, fill_value=output_nodata_val)

    actual_slope_pct_normalized = np.ma.clip(slope_masked, 0, max_slope_pct_for_penalty_norm_config) / max_slope_pct_for_penalty_norm_config
    target_slope_pct_normalized = np.clip(float(target_slope_pct_config), 0, max_slope_pct_for_penalty_norm_config) / max_slope_pct_for_penalty_norm_config
    difference = actual_slope_pct_normalized - target_slope_pct_normalized

    if penalty_type_config.lower() == 'quadratic':
        score = 1.0 - np.ma.power(difference, 2)
    elif penalty_type_config.lower() == 'linear':
        score = 1.0 - np.ma.abs(difference)
    else:
        score = 1.0 - np.ma.power(difference, 2)
        print(f"    WARNING: Unknown penalty_type '{penalty_type_config}' for TARGET_SLOPE_PERCENT_SCORE. Defaulting to quadratic.")

    score_clipped = np.ma.clip(score, 0, 1)
    result_array = score_clipped.filled(output_nodata_val).astype(np.float32)

    return pd.Series(result_array, index=avg_slope_percent_data.index, name=TARGET_SLOPE_PERCENT_SCORE_COLUMN) if isinstance(avg_slope_percent_data, pd.Series) else result_array


def calculate_length_adjusted_target_grade_score(length_data, avg_slope_percent_data,
                                                 target_slope_pct_config,
                                                 output_nodata_val=np.nan):
    """Calculates Length * (Target Grade % - L1_Penalty_from_Target_%) score."""
    if isinstance(length_data, pd.Series): length_values = length_data.values.astype(np.float32)
    else: length_values = np.array(length_data, dtype=np.float32)
    if isinstance(avg_slope_percent_data, pd.Series): slope_values = avg_slope_percent_data.values.astype(np.float32)
    else: slope_values = np.array(avg_slope_percent_data, dtype=np.float32)

    mask = np.isnan(slope_values) | np.isnan(length_values) | (length_values <= 0)
    slope_masked = np.ma.array(slope_values, mask=mask, fill_value=np.nan)
    length_masked = np.ma.array(length_values, mask=mask, fill_value=np.nan)

    target_grade_pct_val = float(target_slope_pct_config)
    penalty = np.ma.abs(target_grade_pct_val - slope_masked)
    effective_grade = target_grade_pct_val - penalty
    score = length_masked * effective_grade

    result_array = score.filled(output_nodata_val).astype(np.float32)
    return pd.Series(result_array, index=avg_slope_percent_data.index, name=LENGTH_ADJUSTED_TARGET_GRADE_SCORE_COLUMN) if isinstance(avg_slope_percent_data, pd.Series) else result_array

# calculate_average_land_cover_suitability: Uses the path to the aligned raster
def calculate_average_land_cover_suitability(trail_geom, aligned_lcm_raster_path, # Path to ALIGNED raster
                                             land_cover_suitability_dict, default_suitability_score,
                                             sampling_distance_m, trail_id_for_debug="N/A"):
    """
    Calculates the average land cover suitability score for a given trail geometry.
    Samples the ALIGNED LCM raster along the trail and uses the suitability dictionary.
    """
    if not isinstance(trail_geom, LineString) or trail_geom.is_empty:
        return np.nan

    sample_points_coords = []
    trail_length = trail_geom.length
    if trail_length > 0:
        num_segments = max(1, math.ceil(trail_length / sampling_distance_m))
        distances = np.linspace(0, trail_length, num_segments + 1)
        sample_points = [trail_geom.interpolate(dist) for dist in distances]
        sample_points_coords = [(p.x, p.y) for p in sample_points if isinstance(p, Point) and not p.is_empty]
        start_coord = (trail_geom.coords[0][0], trail_geom.coords[0][1])
        end_coord = (trail_geom.coords[-1][0], trail_geom.coords[-1][1])
        if not any(np.allclose(start_coord, sp) for sp in sample_points_coords):
             sample_points_coords.insert(0, start_coord)
        if not any(np.allclose(end_coord, sp) for sp in sample_points_coords):
             sample_points_coords.append(end_coord)
        sample_points_coords = list(dict.fromkeys(sample_points_coords))

    if not sample_points_coords:
        return np.nan

    suitability_scores = []
    try:
        with rasterio.open(aligned_lcm_raster_path) as lcm_src: # Open the aligned raster
            lcm_nodata = lcm_src.nodata
            sampled_values = list(lcm_src.sample(sample_points_coords))

            for val_array in sampled_values:
                lcm_value = val_array[0]
                is_nodata = False
                if lcm_nodata is not None:
                    if np.issubdtype(type(lcm_value), np.floating) and np.issubdtype(type(lcm_nodata), np.floating):
                        is_nodata = np.isnan(lcm_value) or np.isclose(lcm_value, lcm_nodata)
                    else:
                         is_nodata = pd.isna(lcm_value) or lcm_value == lcm_nodata

                if is_nodata:
                    suitability_scores.append(default_suitability_score)
                else:
                    # Ensure lookup key is integer
                    suitability_scores.append(land_cover_suitability_dict.get(int(lcm_value), default_suitability_score))

    except rasterio.RasterioIOError:
        print(f"    ERROR (Trail {trail_id_for_debug}): Could not open or read ALIGNED LCM raster: {aligned_lcm_raster_path}")
        return np.nan
    except Exception as e:
        print(f"    ERROR (Trail {trail_id_for_debug}): Unexpected error sampling ALIGNED LCM: {e}")
        return np.nan

    if not suitability_scores:
        return np.nan

    return np.mean(suitability_scores)

# --- Core Logic Functions ---
# analyze_trail_properties: Accepts path to aligned LCM
def analyze_trail_properties(trails_gdf, dem_raster_path, id_column='osmid',
                             aligned_lcm_path=None, # Renamed parameter
                             land_cover_suitability_dict_config=None,
                             default_suitability_score_config=0.0,
                             lcm_sampling_distance_config=10):
    """
    Analyzes trails for elevation, gradient, and land cover suitability (using aligned LCM).
    """
    print(f"\n--- Step 3: Calculating Trail Properties (Elevation, Gradient, Land Cover) ---")

    output_cols_base = [id_column, 'start_coord_str', 'end_coord_str', 'length_m', 'start_elev_m', 'end_elev_m', NET_GAIN_COLUMN, MAX_ELEVATION_COLUMN, 'avg_uphill_gradient_degrees', ABS_GRAD_DEG_COLUMN, 'absolute_avg_gradient_percent']
    output_cols_scores = [TARGET_SLOPE_PERCENT_SCORE_COLUMN, LENGTH_ADJUSTED_TARGET_GRADE_SCORE_COLUMN]
    lcm_analysis_enabled = aligned_lcm_path is not None and land_cover_suitability_dict_config is not None
    if lcm_analysis_enabled:
        output_cols_scores.extend([AVG_LAND_COVER_SUITABILITY_SCORE_COLUMN, LAND_COVER_LENGTH_RANKING_SCORE_COLUMN])
        print(f"Using aligned LCM for suitability analysis: {aligned_lcm_path}")
    elif RANKING_STRATEGY == 'LAND_COVER_LENGTH_SCORE':
        output_cols_scores.extend([AVG_LAND_COVER_SUITABILITY_SCORE_COLUMN, LAND_COVER_LENGTH_RANKING_SCORE_COLUMN])
        print("Warning: Land cover ranking selected, but no valid aligned LCM path available. Scores will be NaN.")

    output_cols = sorted(list(set(output_cols_base + output_cols_scores)))

    if trails_gdf is None or trails_gdf.empty: return pd.DataFrame(columns=output_cols)
    if not os.path.exists(dem_raster_path): print(f"Error: DEM raster not found: {dem_raster_path}"); return pd.DataFrame(columns=output_cols)

    results = []
    print(f"Opening DEM raster: {dem_raster_path}")
    try:
        with rasterio.open(dem_raster_path) as dem_src:
            dem_crs_str = dem_src.crs.to_string().upper()
            trails_crs_str = trails_gdf.crs.to_string().upper()
            if trails_crs_str != dem_crs_str: print(f"FATAL ERROR: Trail CRS ({trails_gdf.crs}) != DEM CRS ({dem_src.crs})."); return pd.DataFrame(columns=output_cols)
            dem_nodata = dem_src.nodata

            trails_gdf_proc = trails_gdf.reset_index(drop=True)
            print(f"Processing {len(trails_gdf_proc)} trails...")

            for i, row in trails_gdf_proc.iterrows():
                trail_id = str(row.get(id_column, f"UnknownID_{i}"))
                geom = row.geometry
                if not isinstance(geom, LineString) or geom.is_empty: continue

                current_trail_data = {col: np.nan for col in output_cols}
                current_trail_data[id_column] = trail_id
                current_trail_data['length_m'] = geom.length
                current_trail_data['start_coord_str'] = str(geom.coords[0]) if geom.coords else "N/A"
                current_trail_data['end_coord_str'] = str(geom.coords[-1]) if geom.coords else "N/A"

                start_elev, end_elev, net_gain, max_elev_val, grad_deg = np.nan, np.nan, np.nan, np.nan, np.nan
                if geom.coords and len(geom.coords) >= 2:
                    sample_coords_elev = [(geom.coords[0][0], geom.coords[0][1]), (geom.coords[-1][0], geom.coords[-1][1])]
                    try:
                        sampled_elevs_raw = list(dem_src.sample(sample_coords_elev))
                        sampled_elevs = np.array([val[0] for val in sampled_elevs_raw], dtype=np.float32)
                        valid_mask = ~np.isnan(sampled_elevs)
                        if dem_nodata is not None:
                            if np.issubdtype(type(dem_nodata), np.floating): valid_mask &= ~np.isclose(sampled_elevs, dem_nodata)
                            else: valid_mask &= (sampled_elevs != dem_nodata)
                        if valid_mask.sum() == 2:
                            start_elev, end_elev = sampled_elevs[0], sampled_elevs[1]; net_gain = end_elev - start_elev; max_elev_val = max(start_elev, end_elev)
                            if geom.length > 1e-6: grad_deg = np.degrees(np.arctan2(net_gain, geom.length))
                            else: grad_deg = 0.0
                        elif valid_mask.sum() == 1:
                             valid_index = np.where(valid_mask)[0][0]; start_elev = end_elev = max_elev_val = sampled_elevs[valid_index]; net_gain = 0.0; grad_deg = 0.0
                    except Exception as e_elev: print(f"    Warning (Trail {trail_id}): Elevation sampling failed: {e_elev}")

                current_trail_data.update({'start_elev_m': start_elev, 'end_elev_m': end_elev, NET_GAIN_COLUMN: net_gain, MAX_ELEVATION_COLUMN: max_elev_val, 'avg_uphill_gradient_degrees': grad_deg})
                abs_avg_grad_deg = abs(grad_deg) if pd.notna(grad_deg) else np.nan
                abs_avg_grad_pct = math.tan(math.radians(abs_avg_grad_deg)) * 100.0 if pd.notna(abs_avg_grad_deg) else np.nan
                current_trail_data[ABS_GRAD_DEG_COLUMN] = abs_avg_grad_deg
                current_trail_data['absolute_avg_gradient_percent'] = abs_avg_grad_pct

                if pd.notna(current_trail_data['absolute_avg_gradient_percent']):
                    target_score_series = calculate_target_slope_percent_score(pd.Series([current_trail_data['absolute_avg_gradient_percent']]), TARGET_SLOPE_PERCENT, MAX_SLOPE_PERCENT_FOR_PCT_SCORE_NORMALIZATION, TARGET_SLOPE_PENALTY_TYPE_FOR_PCT_SCORE)
                    current_trail_data[TARGET_SLOPE_PERCENT_SCORE_COLUMN] = target_score_series.iloc[0] if not target_score_series.empty else np.nan
                    len_adj_score_series = calculate_length_adjusted_target_grade_score(pd.Series([current_trail_data['length_m']]), pd.Series([current_trail_data['absolute_avg_gradient_percent']]), TARGET_SLOPE_PERCENT)
                    current_trail_data[LENGTH_ADJUSTED_TARGET_GRADE_SCORE_COLUMN] = len_adj_score_series.iloc[0] if not len_adj_score_series.empty else np.nan

                avg_suitability = np.nan; land_cover_rank_score = np.nan
                if lcm_analysis_enabled:
                    avg_suitability = calculate_average_land_cover_suitability(geom, aligned_lcm_path, land_cover_suitability_dict_config, default_suitability_score_config, lcm_sampling_distance_config, trail_id)
                    if pd.notna(avg_suitability) and pd.notna(current_trail_data['length_m']) and current_trail_data['length_m'] > 0:
                        land_cover_rank_score = current_trail_data['length_m'] * avg_suitability
                if AVG_LAND_COVER_SUITABILITY_SCORE_COLUMN in current_trail_data: current_trail_data[AVG_LAND_COVER_SUITABILITY_SCORE_COLUMN] = avg_suitability
                if LAND_COVER_LENGTH_RANKING_SCORE_COLUMN in current_trail_data: current_trail_data[LAND_COVER_LENGTH_RANKING_SCORE_COLUMN] = land_cover_rank_score

                results.append(current_trail_data)

            stats_df = pd.DataFrame(results)
            for col in output_cols:
                if col not in stats_df.columns: stats_df[col] = np.nan
            return stats_df[output_cols]

    except Exception as e: print(f"Error during trail property analysis: {e}"); traceback.print_exc(); return pd.DataFrame(columns=output_cols)

# calculate_min_distances, rank_trails, create_top_trails_details_df, visualize_results remain the same
def calculate_min_distances(trails_gdf, access_points_gdf, id_col):
    """Calculates minimum distance from each trail's representative point to the nearest access point geometry."""
    if trails_gdf is None or trails_gdf.empty or access_points_gdf is None or access_points_gdf.empty:
        print("  Skipping min distance calculation: Empty input GeoDataFrames.")
        return None
    if trails_gdf.crs != access_points_gdf.crs:
        print(f"  Error in min_distances: CRS mismatch between trails ({trails_gdf.crs}) and access points ({access_points_gdf.crs}).")
        return None

    trails_repr_pts = trails_gdf.copy()
    try:
        trails_repr_pts['geometry'] = trails_repr_pts.geometry.representative_point()
    except Exception as e_repr:
         print(f"  Warning: Could not calculate representative points for trails: {e_repr}. Using original geometry (might be inaccurate for distance).")

    try:
        print(f"  Performing spatial join to find nearest access points...")
        joined_gdf = gpd.sjoin_nearest(trails_repr_pts[[id_col, 'geometry']], access_points_gdf[['geometry']],
                                       how='left', distance_col="dist_temp")
        print(f"  Spatial join completed.")

        if 'dist_temp' not in joined_gdf.columns:
             print("  Warning: 'dist_temp' column not found after sjoin_nearest. Distances might be incorrect.")
             raise NotImplementedError("sjoin_nearest failed, manual fallback not fully implemented here for performance reasons.")

    except AttributeError:
         print("  Warning: `sjoin_nearest` not available (requires GeoPandas >= 0.10). Cannot calculate distances efficiently.")
         return None
    except Exception as e_sjoin:
        print(f"  Error during spatial join (sjoin_nearest): {e_sjoin}. Cannot calculate distances.")
        traceback.print_exc()
        return None

    if joined_gdf is not None and not joined_gdf.empty and 'dist_temp' in joined_gdf.columns and id_col in joined_gdf.columns:
        joined_gdf = joined_gdf.dropna(subset=[id_col])
        min_dist_series = joined_gdf.groupby(id_col)['dist_temp'].min()
        return min_dist_series.rename("min_distance")
    else:
        print("  Spatial join result is invalid or empty. Cannot determine minimum distances.")
        return None


def rank_trails(stats_df_input, id_col, length_col, net_gain_col, ranking_strategy_config,
                abs_grad_deg_col_config, target_slope_pct_score_col_config,
                len_adj_target_grade_score_col_config,
                avg_land_cover_suitability_col_config,
                land_cover_length_score_col_config,
                rank_col_name_config, rank_descending_config,
                min_length_filter_config, absolute_min_net_gain_filter_config):
    """Filters trails based on length/gain and ranks them according to the chosen strategy."""
    print("\n--- Step 5: Filtering (Length/Gain) and Ranking Trails ---")
    if stats_df_input is None or stats_df_input.empty:
        print("Input statistics DataFrame is empty. Cannot rank trails.")
        return pd.DataFrame()
    df_processed = stats_df_input.copy()

    rank_by_col_actual = None
    if ranking_strategy_config == 'ABSOLUTE_GRADIENT_DEGREES': rank_by_col_actual = abs_grad_deg_col_config
    elif ranking_strategy_config == 'TARGET_SLOPE_PERCENT_SCORE': rank_by_col_actual = target_slope_pct_score_col_config
    elif ranking_strategy_config == 'LENGTH_ADJUSTED_TARGET_GRADE_SCORE': rank_by_col_actual = len_adj_target_grade_score_col_config
    elif ranking_strategy_config == 'LAND_COVER_LENGTH_SCORE': rank_by_col_actual = land_cover_length_score_col_config
    else: print(f"Error: Unknown RANKING_STRATEGY '{ranking_strategy_config}' specified."); return pd.DataFrame()

    required_cols_for_ranking = [id_col, length_col]
    if rank_by_col_actual: required_cols_for_ranking.append(rank_by_col_actual)
    else: print(f"Error: Could not determine ranking column."); return pd.DataFrame()

    if absolute_min_net_gain_filter_config is not None and absolute_min_net_gain_filter_config > 0:
        if net_gain_col in df_processed.columns: required_cols_for_ranking.append(net_gain_col)
        else: print(f"Warning: Net gain column '{net_gain_col}' not found for filtering.")

    missing_cols = [col for col in required_cols_for_ranking if col not in df_processed.columns]
    if missing_cols: print(f"Error: Missing required columns: {missing_cols}."); return pd.DataFrame()

    print(f"Filtering and ranking based on column: '{rank_by_col_actual}'")

    initial_rows = len(df_processed)
    cols_to_check_nan = [length_col, rank_by_col_actual]
    if net_gain_col in required_cols_for_ranking: cols_to_check_nan.append(net_gain_col)
    df_processed.dropna(subset=cols_to_check_nan, inplace=True)

    if min_length_filter_config is not None and min_length_filter_config > 0:
        df_processed = df_processed[df_processed[length_col] >= min_length_filter_config]

    if absolute_min_net_gain_filter_config is not None and absolute_min_net_gain_filter_config > 0 and net_gain_col in df_processed.columns:
        df_processed = df_processed[df_processed[net_gain_col].abs() >= absolute_min_net_gain_filter_config]

    if df_processed.empty: print("No trails remaining after applying filters."); return pd.DataFrame()

    ranked_df = df_processed.sort_values(by=rank_by_col_actual, ascending=(not rank_descending_config))
    ranked_df[rank_col_name_config] = range(1, len(ranked_df) + 1)

    print(f"Ranking complete. Final ranked DataFrame has {len(ranked_df)} trails (ranked from {initial_rows} initially analyzed).")
    return ranked_df


def create_top_trails_details_df(top_trails_merged_gdf, n_top, id_col, rank_col,
                                 actual_ranking_score_col,
                                 length_col, center_lat, center_lon,
                                 optional_osm_tags_config, other_tags_col_config, dist_cols_list):
    """Creates a detailed DataFrame for the top N trails, including geocoding and distance from center."""
    print(f"\n--- Step 7: Creating Detailed Output DataFrame for Top {n_top} ---")
    if top_trails_merged_gdf is None or top_trails_merged_gdf.empty: print("  Input GDF empty."); return pd.DataFrame()

    details = []
    all_possible_score_cols = [ABS_GRAD_DEG_COLUMN, 'absolute_avg_gradient_percent', TARGET_SLOPE_PERCENT_SCORE_COLUMN, LENGTH_ADJUSTED_TARGET_GRADE_SCORE_COLUMN, AVG_LAND_COVER_SUITABILITY_SCORE_COLUMN, LAND_COVER_LENGTH_RANKING_SCORE_COLUMN]
    base_stat_cols = [id_col, rank_col, actual_ranking_score_col, length_col, 'start_elev_m', 'end_elev_m', NET_GAIN_COLUMN, MAX_ELEVATION_COLUMN]
    existing_score_cols = [sc for sc in all_possible_score_cols if sc in top_trails_merged_gdf.columns]
    existing_base_stat_cols = [bsc for bsc in base_stat_cols if bsc in top_trails_merged_gdf.columns]
    existing_dist_cols = [dc for dc in dist_cols_list if dc in top_trails_merged_gdf.columns]
    existing_osm_tags = [otc for otc in optional_osm_tags_config if otc in top_trails_merged_gdf.columns]
    has_other_tags_col = other_tags_col_config and other_tags_col_config in top_trails_merged_gdf.columns

    geom_col_name = top_trails_merged_gdf.geometry.name
    geolocator = Nominatim(user_agent=f"trail_detail_v12_{PLACE_NAME.replace(' ','_')}", timeout=10)
    center_coords_wgs84 = (center_lat, center_lon)
    top_trails_wgs84 = None
    try:
        if top_trails_merged_gdf.crs.to_string().upper() != SOURCE_CRS_GPS.upper(): top_trails_wgs84 = top_trails_merged_gdf[[geom_col_name]].to_crs(SOURCE_CRS_GPS)
        else: top_trails_wgs84 = top_trails_merged_gdf[[geom_col_name]].copy()
    except Exception as e_reproject: print(f"  Warning: Could not reproject top trails to WGS84: {e_reproject}")

    print(f"  Processing {len(top_trails_merged_gdf)} trails for detailed output...")
    for index, row in top_trails_merged_gdf.iterrows():
        trail_data = {}
        for col in existing_base_stat_cols + existing_score_cols + existing_dist_cols + existing_osm_tags: trail_data[col] = row.get(col)
        if has_other_tags_col: trail_data[other_tags_col_config] = row.get(other_tags_col_config)

        lat, lon, placename, dist_center_km = np.nan, np.nan, "Geocoding Failed", np.nan
        if top_trails_wgs84 is not None and index in top_trails_wgs84.index:
            geom_wgs84 = top_trails_wgs84.loc[index, geom_col_name]
            if isinstance(geom_wgs84, LineString) and not geom_wgs84.is_empty:
                try:
                    repr_pt_wgs84 = geom_wgs84.representative_point()
                    lat, lon = repr_pt_wgs84.y, repr_pt_wgs84.x
                    if all(pd.notna(c) for c in [lat, lon, center_lat, center_lon]):
                        try: dist_center_km = geopy_distance(center_coords_wgs84, (lat, lon)).km
                        except Exception as e_dist: print(f"    Warning (Trail {row.get(id_col, index)}): Dist calc failed: {e_dist}")
                    if pd.notna(lat) and pd.notna(lon):
                        try:
                            location = geolocator.reverse(f"{lat:.6f}, {lon:.6f}", exactly_one=True, language='en', timeout=5)
                            placename = location.address if location else "Placename not found"
                        except GeocoderTimedOut: placename = "Geocoding Timed Out"
                        except GeocoderServiceError as e_geo_service: placename = f"Geocoding Service Error: {e_geo_service}"
                        except Exception as e_geo: placename = f"Geocoding Error: {e_geo}"
                        finally: time.sleep(1.1)
                except Exception as e_repr_pt: print(f"    Warning (Trail {row.get(id_col, index)}): Repr point failed: {e_repr_pt}")

        trail_data.update({'representative_latitude': lat, 'representative_longitude': lon, 'nearest_placename': placename, 'distance_from_center_km': dist_center_km})
        details.append(trail_data)

    if not details: print("  No details generated."); return pd.DataFrame()
    details_df = pd.DataFrame(details)

    final_cols_order = [id_col, rank_col, actual_ranking_score_col]
    for sc_col in existing_score_cols:
        if sc_col != actual_ranking_score_col and sc_col not in final_cols_order: final_cols_order.append(sc_col)
    location_cols = ['nearest_placename', 'distance_from_center_km', 'representative_latitude', 'representative_longitude']
    final_cols_order.extend([lc for lc in location_cols if lc in details_df.columns])
    core_stats = [length_col, NET_GAIN_COLUMN, MAX_ELEVATION_COLUMN, 'start_elev_m', 'end_elev_m']
    final_cols_order.extend([cs for cs in core_stats if cs in details_df.columns])
    final_cols_order.extend([dc for dc in existing_dist_cols if dc not in final_cols_order])
    final_cols_order.extend([ot for ot in existing_osm_tags if ot not in final_cols_order])
    if has_other_tags_col and other_tags_col_config not in final_cols_order: final_cols_order.append(other_tags_col_config)
    remaining_cols = [col for col in details_df.columns if col not in final_cols_order]
    final_cols_order.extend(remaining_cols)
    details_df = details_df[final_cols_order]

    for tag_col in existing_osm_tags: details_df[tag_col] = details_df[tag_col].fillna('N/A')
    for dist_c in existing_dist_cols: details_df[dist_c] = details_df[dist_c].fillna(-1.0)
    if MAX_ELEVATION_COLUMN in details_df.columns: details_df[MAX_ELEVATION_COLUMN] = details_df[MAX_ELEVATION_COLUMN].fillna(-9999.0)
    if has_other_tags_col: details_df[other_tags_col_config] = details_df[other_tags_col_config].apply(lambda x: x if isinstance(x, dict) else ({} if pd.isna(x) else str(x)))

    cols_to_round = existing_score_cols + [length_col, NET_GAIN_COLUMN, MAX_ELEVATION_COLUMN, 'start_elev_m', 'end_elev_m', 'distance_from_center_km'] + existing_dist_cols
    for num_col in cols_to_round:
        if num_col in details_df.columns: details_df[num_col] = pd.to_numeric(details_df[num_col], errors='coerce').round(4)
    if 'representative_latitude' in details_df.columns: details_df['representative_latitude'] = pd.to_numeric(details_df['representative_latitude'], errors='coerce').round(6)
    if 'representative_longitude' in details_df.columns: details_df['representative_longitude'] = pd.to_numeric(details_df['representative_longitude'], errors='coerce').round(6)

    print(f"  Successfully created detailed DataFrame for {len(details_df)} trails.")
    return details_df


def visualize_results(top_trails_gdf, n_top, place_name, rank_col,
                      actual_ranking_score_col_name,
                      vis_cmap_config, output_png_path,
                      ranking_strategy_for_title,
                      target_slope_for_title=None,
                      penalty_type_for_title=None):
    """Visualizes the top N trails on a map with contextily basemap."""
    print(f"\n--- Step 8: Visualizing Top {n_top} Trails ---")
    if top_trails_gdf is None or top_trails_gdf.empty: print("Input GDF empty."); return
    # Check specifically for geometry column and non-empty/non-null geometries
    if 'geometry' not in top_trails_gdf.columns or top_trails_gdf.geometry.isna().all():
        print("Input GDF has no valid geometries to plot.")
        return

    try:
        fig, ax = plt.subplots(1, 1, figsize=(14, 12))
        plot_column = rank_col
        top_trails_gdf.plot(column=plot_column, cmap=vis_cmap_config, ax=ax,
                            legend=True, legend_kwds={'label': f"Trail Rank (Top {min(n_top, len(top_trails_gdf))}, 1=Best)", 'shrink': 0.6},
                            linewidth=2.5, aspect='equal')

        try:
            print("  Adding basemap...")
            cx.add_basemap(ax, crs=top_trails_gdf.crs.to_string(), source=CTX_PROVIDER, zoom='auto', attribution_size=7)
            print("  Basemap added.")
        except Exception as e_basemap: print(f"  Warning: Contextily basemap failed: {e_basemap}.")

        title_detail = ""
        if ranking_strategy_for_title == 'ABSOLUTE_GRADIENT_DEGREES': title_detail = f"Abs. Avg. Gradient ({ABS_GRAD_DEG_COLUMN.replace('_',' ').title()})"
        elif ranking_strategy_for_title == 'TARGET_SLOPE_PERCENT_SCORE': title_detail = f"Target Slope {target_slope_for_title}% Score ({penalty_type_for_title.title()} Pen.)"
        elif ranking_strategy_for_title == 'LENGTH_ADJUSTED_TARGET_GRADE_SCORE': title_detail = f"Length-Adj. Target Grade {target_slope_for_title}% Score"
        elif ranking_strategy_for_title == 'LAND_COVER_LENGTH_SCORE': title_detail = f"Land Cover Suitability * Length Score"
        else: title_detail = f"Score: {actual_ranking_score_col_name}"
        ax.set_title(f"Top {min(n_top, len(top_trails_gdf))} Trails near {place_name}\n(Ranked by: {title_detail})", fontsize=14)
        ax.set_axis_off()

        print("  Adding rank labels to trails...")
        text_kwargs = dict(ha='center', va='bottom', fontsize=7, color='black', path_effects=[PathEffects.withStroke(linewidth=1.5, foreground='white')])
        for idx, row in top_trails_gdf.iterrows():
            # Check geometry is valid before getting representative point
            if pd.notna(row[rank_col]) and row.geometry and not row.geometry.is_empty:
                try:
                    label_point = row.geometry.representative_point()
                    xlim = ax.get_xlim(); ylim = ax.get_ylim()
                    if xlim[0] <= label_point.x <= xlim[1] and ylim[0] <= label_point.y <= ylim[1]:
                         ax.text(label_point.x, label_point.y, str(int(row[rank_col])), **text_kwargs)
                except Exception as e_text: print(f"  Warning: Could not place text label for rank {row[rank_col]}: {e_text}")

        if not top_trails_gdf.empty:
            minx, miny, maxx, maxy = top_trails_gdf.total_bounds
            if np.all(np.isfinite([minx, miny, maxx, maxy])):
                x_range = max(maxx - minx, 1); y_range = max(maxy - miny, 1)
                x_pad = max(x_range * 0.1, 500); y_pad = max(y_range * 0.1, 500)
                ax.set_xlim(minx - x_pad, maxx + x_pad)
                ax.set_ylim(miny - y_pad, maxy + y_pad)
            else: print("  Warning: Invalid bounds for map extent.")

        plt.tight_layout(pad=0.5)
        plt.savefig(output_png_path, dpi=250, bbox_inches='tight')
        plt.close(fig)
        print(f"  Visualization saved: {os.path.basename(output_png_path)}")
    except Exception as e_viz:
        print(f"  Error during visualization: {e_viz}")
        traceback.print_exc()
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)

# align_lcm_raster function remains the same as v12.1
def align_lcm_raster(input_lcm_path, target_crs_str, target_bounds_tuple,
                     target_resolution_tuple, output_aligned_lcm_path,
                     resampling_method=Resampling.nearest, nodata_val=None):
    """
    Aligns (reprojects and clips) an input LCM raster to the target CRS, bounds, and resolution.
    """
    print(f"  Aligning LCM: {os.path.basename(input_lcm_path)} -> {os.path.basename(output_aligned_lcm_path)}")
    try:
        with rasterio.open(input_lcm_path) as src:
            dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
                src.crs, target_crs_str, src.width, src.height, *src.bounds,
                dst_width=int(np.ceil((target_bounds_tuple[2] - target_bounds_tuple[0]) / target_resolution_tuple[0])),
                dst_height=int(np.ceil((target_bounds_tuple[3] - target_bounds_tuple[1]) / abs(target_resolution_tuple[1]))),
                dst_bounds=target_bounds_tuple,
                resolution=target_resolution_tuple
            )
            dst_profile = src.profile.copy()
            dst_profile.update({
                'crs': target_crs_str, 'transform': dst_transform, 'width': dst_width, 'height': dst_height,
                'nodata': nodata_val if nodata_val is not None else src.nodata, 'dtype': src.dtype
            })
            with rasterio.open(output_aligned_lcm_path, 'w', **dst_profile) as dst:
                rasterio.warp.reproject(
                    source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=dst_profile['transform'], dst_crs=dst_profile['crs'],
                    resampling=resampling_method, dst_nodata=dst_profile['nodata']
                )
        print("  LCM Alignment successful.")
        return True
    except FileNotFoundError: print(f"  Error aligning LCM: Input file not found at {input_lcm_path}"); return False
    except Exception as e:
        print(f"  Error during LCM alignment: {e}"); traceback.print_exc()
        if os.path.exists(output_aligned_lcm_path):
            try: os.remove(output_aligned_lcm_path)
            except OSError: pass
        return False

# =============================================================================
# --- Main Execution Workflow ---
# =============================================================================
if __name__ == "__main__":
    start_time_main = time.time()

    # --- 1. Geocoding and Bounding Box ---
    print(f"\n--- Step 1: Geocoding and Bounding Box ---")
    geolocator = Nominatim(user_agent=f"trail_search_v12_{PLACE_NAME.replace(' ','_')}")
    target_bounds_proj = None
    try:
        print(f"  Geocoding '{PLACE_NAME}'...")
        location = geolocator.geocode(PLACE_NAME, timeout=20)
        if not location: print(f"Error: Could not geocode '{PLACE_NAME}'."); sys.exit(1)
        center_lat, center_lon = location.latitude, location.longitude
        print(f"  Coordinates found: Lat={center_lat:.4f}, Lon={center_lon:.4f}")
        bounds_ll = get_bounding_box(center_lat, center_lon, RADIUS_KM)
        print(f"  Bounding box (WGS84): {bounds_ll}")
        target_bounds_proj = rasterio.warp.transform_bounds(SOURCE_CRS_GPS, TARGET_CRS, *bounds_ll)
        print(f"  Bounding box ({TARGET_CRS}): {target_bounds_proj}")
    except (GeocoderTimedOut, GeocoderServiceError) as e: print(f"Geocoding error: {e}."); sys.exit(1)
    except Exception as e: print(f"Error during geocoding/bounds: {e}"); traceback.print_exc(); sys.exit(1)

    # --- 1b. Check and Align LCM Raster ---
    print(f"\n--- Step 1b: Check and Align LCM Raster ---")
    lcm_path_for_analysis = None
    if not os.path.exists(LCM_PATH):
        print(f"  Error: Input LCM file not found at specified path: {LCM_PATH}")
        if RANKING_STRATEGY == 'LAND_COVER_LENGTH_SCORE': print("  WARNING: Ranking by land cover selected, but input LCM is missing. Scores will be NaN.")
    else:
        try:
            with rasterio.open(LCM_PATH) as lcm_src:
                lcm_crs = lcm_src.crs
                print(f"  Input LCM CRS: {lcm_crs}")
                if lcm_crs and lcm_crs.to_string().upper() == TARGET_CRS.upper():
                    print(f"  Input LCM is already in the target CRS ({TARGET_CRS}). Using original file.")
                    lcm_path_for_analysis = LCM_PATH # Use the original path
                    # Optional: Could add a check here to see if the extent/resolution roughly matches the AOI if needed
                else:
                    print(f"  Input LCM CRS ({lcm_crs}) does not match target CRS ({TARGET_CRS}). Alignment required.")
                    aligned_lcm_output_path = PATHS["aligned_lcm_temp"]
                    alignment_successful = align_lcm_raster(
                        input_lcm_path=LCM_PATH, target_crs_str=TARGET_CRS, target_bounds_tuple=target_bounds_proj,
                        target_resolution_tuple=LCM_TARGET_RESOLUTION, output_aligned_lcm_path=aligned_lcm_output_path,
                        resampling_method=Resampling.nearest, nodata_val=LCM_NODATA_VAL
                    )
                    if alignment_successful: lcm_path_for_analysis = aligned_lcm_output_path
                    else: print(f"  Error: Failed to align LCM raster.")
        except rasterio.RasterioIOError as e_rio: print(f"  Error opening input LCM file '{LCM_PATH}': {e_rio}")
        except Exception as e_lcm_check: print(f"  Unexpected error during LCM check/alignment: {e_lcm_check}"); traceback.print_exc()
        if lcm_path_for_analysis is None and RANKING_STRATEGY == 'LAND_COVER_LENGTH_SCORE': print("  WARNING: Cannot analyze land cover.")

    # --- 2. Extract & Process Trail Geometries & Tags ---
    print(f"\n--- Step 2: Extract & Process Trail Geometries & Tags ---")
    trails_gdf_raw = extract_osm_features(PBF_FILE_PATH, bounds_ll, TRAIL_TAGS, PATHS["temp_trails_geojson"], "trails")
    analysis_input_gdf = standardize_and_reproject(trails_gdf_raw, TARGET_CRS, ID_COLUMN, 'id', OPTIONAL_OSM_TAGS, OTHER_TAGS_COLUMN_NAME)
    if analysis_input_gdf is None or analysis_input_gdf.empty: print("Trail extraction/processing failed."); sys.exit(1)

    # --- 2b. Apply Tag Exclusion Filters ---
    initial_count = len(analysis_input_gdf)
    print(f"\n--- Step 2b: Applying Tag Exclusion Filters (Initial: {initial_count} trails) ---")
    gdf_filtered = analysis_input_gdf.copy()
    for tag_key, excluded_vals in EXCLUDE_TAG_VALUES.items():
        if tag_key in gdf_filtered.columns:
            exclude_mask = gdf_filtered[tag_key].fillna('').astype(str).str.lower().isin([str(v).lower() for v in excluded_vals])
            gdf_filtered = gdf_filtered[~exclude_mask]
    print(f"Trails remaining after tag exclusion: {len(gdf_filtered)}")
    if gdf_filtered.empty: print("No trails remaining after tag filters."); sys.exit(0)

    # --- 3. Analyze Trail Properties ---
    stats_df = analyze_trail_properties(
        gdf_filtered, PROJECTED_DEM_PATH, ID_COLUMN,
        aligned_lcm_path=lcm_path_for_analysis, # Pass the path to aligned (or original) LCM
        land_cover_suitability_dict_config=LAND_COVER_SUITABILITY,
        default_suitability_score_config=DEFAULT_SUITABILITY_SCORE,
        lcm_sampling_distance_config=LCM_SAMPLING_DISTANCE
    )
    if stats_df is None or stats_df.empty: print("Trail property analysis failed."); sys.exit(1)

    # --- 4. Calculate Distances to Access Points ---
    dist_cols_added = []
    if CALCULATE_ACCESS_DISTANCE:
        print(f"\n--- Step 4: Calculate Distances to Access Points ---")
        for ap_type, ap_tags, ap_col in [("car parks", {"amenity": ["parking"]}, DIST_CARPARK_COL), ("access roads", {"highway": ACCESS_ROAD_HIGHWAY_VALUES}, DIST_ROAD_COL)]:
            print(f"  Processing access points: {ap_type}")
            ap_gdf_raw = extract_osm_features(PBF_FILE_PATH, bounds_ll, ap_tags, PATHS["temp_access_geojson"], ap_type, add_id=False)
            ap_gdf = standardize_and_reproject(ap_gdf_raw, TARGET_CRS)
            if ap_gdf is not None and not ap_gdf.empty:
                if ap_type == "access roads": ap_gdf = ap_gdf[ap_gdf.geometry.geom_type.isin(['LineString', 'MultiLineString'])]
                if not ap_gdf.empty:
                    trails_for_dist_calc = gdf_filtered[[ID_COLUMN, 'geometry']].copy()
                    if trails_for_dist_calc.empty: print(f"    No trail geometries for distance to {ap_type}."); stats_df[ap_col] = np.nan
                    else:
                        min_dist_series = calculate_min_distances(trails_for_dist_calc, ap_gdf, ID_COLUMN)
                        if min_dist_series is not None and not min_dist_series.empty:
                            stats_df[ID_COLUMN] = stats_df[ID_COLUMN].astype(str)
                            min_dist_series = min_dist_series.reset_index(); min_dist_series[ID_COLUMN] = min_dist_series[ID_COLUMN].astype(str)
                            stats_df = stats_df.merge(min_dist_series.rename(columns={'min_distance': ap_col}), on=ID_COLUMN, how='left')
                            dist_cols_added.append(ap_col); print(f"    Added '{ap_col}' column.")
                        else: print(f"    No min distances calculated for {ap_type}."); stats_df[ap_col] = np.nan
                else: print(f"    No valid {ap_type} geometries found."); stats_df[ap_col] = np.nan
            else: print(f"    No {ap_type} features extracted."); stats_df[ap_col] = np.nan
        if os.path.exists(PATHS["temp_access_geojson"]):
            try: os.remove(PATHS["temp_access_geojson"])
            except OSError as e_rm: print(f"  Warning: Could not remove temp access GeoJSON: {e_rm}")
    else: print("\nSkipping distance calculation.")

    # --- Merge Full OSM Tags ---
    print("\n--- Merging Full OSM Tags into Statistics DataFrame ---")
    merge_cols_from_input = [ID_COLUMN] + [col for col in OPTIONAL_OSM_TAGS if col in gdf_filtered.columns] + ([OTHER_TAGS_COLUMN_NAME] if OTHER_TAGS_COLUMN_NAME in gdf_filtered.columns else [])
    stats_df[ID_COLUMN] = stats_df[ID_COLUMN].astype(str)
    gdf_filtered[ID_COLUMN] = gdf_filtered[ID_COLUMN].astype(str)
    if not gdf_filtered[list(set(merge_cols_from_input))].empty:
        stats_df_full = pd.merge(stats_df, gdf_filtered[list(set(merge_cols_from_input))], on=ID_COLUMN, how='left', suffixes=('', '_osm_dup'))
        dup_osm_cols = [c for c in stats_df_full.columns if c.endswith('_osm_dup')]
        if dup_osm_cols: stats_df_full.drop(columns=dup_osm_cols, inplace=True)
        print("  Successfully merged OSM tags.")
    else: print("  Warning: No OSM tag columns to merge."); stats_df_full = stats_df.copy()

    # --- Save Comprehensive Statistics ---
    try:
        print(f"\nSaving comprehensive statistics to: {PATHS['stats_csv']}")
        df_to_save_all = stats_df_full.copy()
        for tag in OPTIONAL_OSM_TAGS:
            if tag in df_to_save_all.columns: df_to_save_all[tag] = df_to_save_all[tag].fillna('N/A')
        if OTHER_TAGS_COLUMN_NAME in df_to_save_all.columns:
             df_to_save_all[OTHER_TAGS_COLUMN_NAME] = df_to_save_all[OTHER_TAGS_COLUMN_NAME].apply(lambda x: json.dumps(x) if isinstance(x, dict) else (x if pd.notna(x) else '{}')).astype(str)
        for dcol in dist_cols_added:
            if dcol in df_to_save_all.columns: df_to_save_all[dcol] = df_to_save_all[dcol].fillna(-1.0)
        numeric_cols_to_round = [c for c in df_to_save_all.columns if df_to_save_all[c].dtype in ['float64', 'float32', 'int64', 'int32']]
        numeric_cols_to_round = [c for c in numeric_cols_to_round if c not in [ID_COLUMN, RANK_COLUMN_NAME]]
        for num_col in numeric_cols_to_round: df_to_save_all[num_col] = pd.to_numeric(df_to_save_all[num_col], errors='coerce').round(4)
        df_to_save_all.to_csv(PATHS['stats_csv'], index=False, float_format='%.4f')
    except Exception as e_csv_all: print(f"  Warning: Could not save comprehensive stats CSV: {e_csv_all}"); traceback.print_exc()

    # --- 5. Filter and Rank Trails ---
    ranked_stats_df = rank_trails(stats_df_full, ID_COLUMN, 'length_m', NET_GAIN_COLUMN, RANKING_STRATEGY, ABS_GRAD_DEG_COLUMN, TARGET_SLOPE_PERCENT_SCORE_COLUMN, LENGTH_ADJUSTED_TARGET_GRADE_SCORE_COLUMN, AVG_LAND_COVER_SUITABILITY_SCORE_COLUMN, LAND_COVER_LENGTH_RANKING_SCORE_COLUMN, RANK_COLUMN_NAME, RANK_DESCENDING, MIN_TRAIL_LENGTH_FILTER, ABSOLUTE_MIN_NET_GAIN)
    if ranked_stats_df is None or ranked_stats_df.empty: print("Ranking/filtering yielded no results."); sys.exit(1)

    # --- Save Ranked Statistics ---
    try:
        print(f"\nSaving filtered and ranked statistics to: {PATHS['ranked_stats_csv']}")
        df_to_save_ranked = ranked_stats_df.copy()
        if OTHER_TAGS_COLUMN_NAME in df_to_save_ranked.columns and not pd.api.types.is_string_dtype(df_to_save_ranked[OTHER_TAGS_COLUMN_NAME]): df_to_save_ranked[OTHER_TAGS_COLUMN_NAME] = df_to_save_ranked[OTHER_TAGS_COLUMN_NAME].astype(str)
        # numeric_cols_to_round was defined above for stats_csv, can reuse
        for num_col in numeric_cols_to_round:
            if num_col in df_to_save_ranked.columns: df_to_save_ranked[num_col] = pd.to_numeric(df_to_save_ranked[num_col], errors='coerce').round(4)
        df_to_save_ranked.to_csv(PATHS['ranked_stats_csv'], index=False, float_format='%.4f')
    except Exception as e_csv_ranked: print(f"  Warning: Could not save ranked statistics CSV: {e_csv_ranked}"); traceback.print_exc()

    # --- 6. Prepare Data for Top N Trails ---
    print(f"\n--- Step 6: Preparing Full Data for Top {N_TOP_TRAILS} Trails ---")
    top_n_from_ranked_df = ranked_stats_df.head(N_TOP_TRAILS).copy()
    if ID_COLUMN in gdf_filtered.columns and 'geometry' in gdf_filtered.columns:
        top_n_from_ranked_df[ID_COLUMN] = top_n_from_ranked_df[ID_COLUMN].astype(str)
        gdf_filtered[ID_COLUMN] = gdf_filtered[ID_COLUMN].astype(str)
        top_trails_final_gdf = pd.merge(top_n_from_ranked_df, gdf_filtered[[ID_COLUMN, 'geometry']], on=ID_COLUMN, how='left')
        top_trails_final_gdf = gpd.GeoDataFrame(top_trails_final_gdf, geometry='geometry', crs=TARGET_CRS)
        top_trails_final_gdf.sort_values(by=RANK_COLUMN_NAME, inplace=True)
        print(f"  Prepared GeoDataFrame for top {len(top_trails_final_gdf)} trails.")
    else: print(f"Error: Cannot merge geometries for top trails."); top_trails_final_gdf = gpd.GeoDataFrame(top_n_from_ranked_df, crs=TARGET_CRS)
    if top_trails_final_gdf.empty: print(f"Could not prepare GDF for top {N_TOP_TRAILS}."); sys.exit(1)

    # --- 7. Create Detailed CSV for Top Trails ---
    actual_rank_score_col = "UNKNOWN_RANK_COL"
    if RANKING_STRATEGY == 'ABSOLUTE_GRADIENT_DEGREES': actual_rank_score_col = ABS_GRAD_DEG_COLUMN
    elif RANKING_STRATEGY == 'TARGET_SLOPE_PERCENT_SCORE': actual_rank_score_col = TARGET_SLOPE_PERCENT_SCORE_COLUMN
    elif RANKING_STRATEGY == 'LENGTH_ADJUSTED_TARGET_GRADE_SCORE': actual_rank_score_col = LENGTH_ADJUSTED_TARGET_GRADE_SCORE_COLUMN
    elif RANKING_STRATEGY == 'LAND_COVER_LENGTH_SCORE': actual_rank_score_col = LAND_COVER_LENGTH_RANKING_SCORE_COLUMN
    if actual_rank_score_col not in top_trails_final_gdf.columns:
         print(f"Warning: Ranking score column '{actual_rank_score_col}' not in top trails GDF.")
         if RANK_COLUMN_NAME in top_trails_final_gdf.columns : actual_rank_score_col = RANK_COLUMN_NAME; print(f" Using '{RANK_COLUMN_NAME}' as fallback.")
         else: actual_rank_score_col = None; print(" Cannot determine fallback.")

    if actual_rank_score_col:
        top_details_df = create_top_trails_details_df(top_trails_final_gdf, N_TOP_TRAILS, ID_COLUMN, RANK_COLUMN_NAME, actual_rank_score_col, 'length_m', center_lat, center_lon, OPTIONAL_OSM_TAGS, OTHER_TAGS_COLUMN_NAME, dist_cols_added)
    else: print("Skipping detailed CSV creation."); top_details_df = pd.DataFrame()

    if top_details_df is not None and not top_details_df.empty:
        print("\n--- Top Trail Details (Sample) ---")
        print(top_details_df.head(min(N_TOP_TRAILS, 5)).to_string())
        try:
            print(f"\nSaving detailed top trail data to: {PATHS['top_details_csv']}")
            # Diagnostic print of columns being saved to top_details_csv
            print(f"  Columns in top_details_df being saved: {top_details_df.columns.tolist()}")
            if OTHER_TAGS_COLUMN_NAME in top_details_df.columns and top_details_df[OTHER_TAGS_COLUMN_NAME].apply(lambda x: isinstance(x, dict)).any():
                 top_details_df[OTHER_TAGS_COLUMN_NAME] = top_details_df[OTHER_TAGS_COLUMN_NAME].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
            top_details_df.to_csv(PATHS['top_details_csv'], index=False, float_format='%.6f')
        except Exception as e_csv_top: print(f"  Warning: Could not save detailed top trails CSV: {e_csv_top}"); traceback.print_exc()
    else: print("No detailed data generated/saved.")

    # --- 8. Visualize Top Trails ---
    # Corrected check for valid geometries
    if ('geometry' in top_trails_final_gdf.columns and
        not top_trails_final_gdf.empty and
        not top_trails_final_gdf['geometry'].isna().all()): # Check if NOT ALL geometries are NaN/None
        visualize_results(top_trails_final_gdf, N_TOP_TRAILS, PLACE_NAME, RANK_COLUMN_NAME,
                          actual_rank_score_col if actual_rank_score_col else RANK_COLUMN_NAME,
                          VIS_COLORMAP, PATHS["visualization_png"],
                          RANKING_STRATEGY, TARGET_SLOPE_PERCENT, TARGET_SLOPE_PENALTY_TYPE_FOR_PCT_SCORE)
    else:
        print("\nSkipping visualization (no valid geometries).")


    # --- Cleanup Temporary Files ---
    print("\n--- Cleaning up temporary files ---")
    temp_files_to_remove = ["temp_bbox_pbf", "temp_trails_pbf", "temp_trails_geojson", "temp_access_geojson", "aligned_lcm_temp"]
    for key in temp_files_to_remove:
        if key in PATHS and os.path.exists(PATHS[key]):
            try: os.remove(PATHS[key]); print(f"  Removed: {os.path.basename(PATHS[key])}")
            except OSError as e_remove: print(f"  Warning: Could not remove '{PATHS[key]}': {e_remove}")

    end_time_main = time.time()
    print(f"\n--- Workflow Finished ---")
    print(f"Total execution time: {end_time_main - start_time_main:.2f} seconds")
    print(f"End time: {np.datetime64('now', 's')} UTC")
    print(f"Outputs saved in: {PATHS['output_dir']}")
