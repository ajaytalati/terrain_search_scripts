#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidated script for finding, analyzing, ranking, and visualizing steep trails
based on OSM data and a DEM raster.

Version 11.4 - Length-Adjusted Target Grade Score Ranking:
- Introduced RANKING_STRATEGY config to choose between:
    - 'ABSOLUTE_GRADIENT_DEGREES'
    - 'TARGET_SLOPE_PERCENT_SCORE' (0-1 score based on target % and penalty)
    - 'LENGTH_ADJUSTED_TARGET_GRADE_SCORE' (Length * (Target Grade % - Deviation Penalty))
- Added calculation for the new length-adjusted score.
- Output directory and map titles dynamically reflect the chosen ranking strategy.
- All relevant score columns are included in CSV outputs.

Version 11.3 (Corrected):
- Fixed an error in the osmium export command.
- Added alternative ranking method based on a target average slope percentage.

Requires: osmium-tool, geopandas, rasterio, pandas, numpy, matplotlib, contextily, geopy, time, rtree
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
PLACE_NAME = "Ladybower Reservoir, UK"
RADIUS_KM = 20

# --- 2. Input Data Paths ---
PBF_FILE_PATH = "/home/ajay/Python_Projects/steepest_trails_search/united-kingdom-latest.osm.pbf"
PROJECTED_DEM_PATH = "/home/ajay/Python_Projects/steepest_trails_search/outputs_united_kingdom_dem/united_kingdom_dem_proj_27700.tif"

# --- 3. Coordinate Reference System (CRS) ---
TARGET_CRS = "EPSG:27700"

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
    "area": ["yes"]
}
MIN_TRAIL_LENGTH_FILTER = 1000 # meters
ABSOLUTE_MIN_NET_GAIN = 25 # meters
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
# 'ABSOLUTE_GRADIENT_DEGREES': Ranks by absolute average gradient in degrees (steeper is better).
# 'TARGET_SLOPE_PERCENT_SCORE': Ranks by a score (0-1) based on proximity to TARGET_SLOPE_PERCENT.
# 'LENGTH_ADJUSTED_TARGET_GRADE_SCORE': Ranks by Length * (Target Grade % - Penalty for deviation from Target Grade %).
RANKING_STRATEGY = 'LENGTH_ADJUSTED_TARGET_GRADE_SCORE' # <-- CHOOSE RANKING STRATEGY HERE

# --- 7a. Settings for 'ABSOLUTE_GRADIENT_DEGREES' strategy ---
# This column is always calculated, but only used for ranking if strategy is set above.
ABS_GRAD_DEG_COLUMN = 'absolute_avg_gradient_degrees'

# --- 7b. Settings for 'TARGET_SLOPE_PERCENT_SCORE' and 'LENGTH_ADJUSTED_TARGET_GRADE_SCORE' strategies ---
# These settings are used if RANKING_STRATEGY involves a target slope.
TARGET_SLOPE_PERCENT = 14.0  # Target grade percentage (e.g., 12.0 for 12%). Used by both target-based strategies.
# Settings specific to 'TARGET_SLOPE_PERCENT_SCORE' strategy:
TARGET_SLOPE_PENALTY_TYPE_FOR_PCT_SCORE = 'linear' # 'quadratic' or 'linear'.
MAX_SLOPE_PERCENT_FOR_PCT_SCORE_NORMALIZATION = 50.0 # Max slope % for normalizing penalty in the 0-1 score.

# --- 7c. Column Names for Calculated Scores & Final Rank ---
# These columns will be added to the stats DataFrame if the corresponding strategy is active or components are calculated.
TARGET_SLOPE_PERCENT_SCORE_COLUMN = 'target_slope_pct_score' # Holds the 0-1 score for 'TARGET_SLOPE_PERCENT_SCORE'
LENGTH_ADJUSTED_TARGET_GRADE_SCORE_COLUMN = 'len_adj_target_grade_score' # Holds score for 'LENGTH_ADJUSTED_TARGET_GRADE_SCORE'

RANK_DESCENDING = True # True for descending (higher score/gradient is Rank 1)
RANK_COLUMN_NAME = 'rank' # Name for the final rank column in output

# --- 8. Output & Visualization ---
N_TOP_TRAILS = 20
VIS_COLORMAP = 'viridis_r'
OUTPUT_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# --- END OF Configuration ---
# =============================================================================

print(f"--- Consolidated Trail Analysis Workflow v11.4 ---")
print(f"Selected Ranking Strategy: {RANKING_STRATEGY}")
if RANKING_STRATEGY == 'TARGET_SLOPE_PERCENT_SCORE':
    print(f"  Target Slope: {TARGET_SLOPE_PERCENT}%, Penalty: {TARGET_SLOPE_PENALTY_TYPE_FOR_PCT_SCORE}, Max Norm: {MAX_SLOPE_PERCENT_FOR_PCT_SCORE_NORMALIZATION}% for 0-1 score")
elif RANKING_STRATEGY == 'LENGTH_ADJUSTED_TARGET_GRADE_SCORE':
    print(f"  Target Slope for Length-Adjusted Score: {TARGET_SLOPE_PERCENT}% (Penalty: L1 norm from target)")
elif RANKING_STRATEGY == 'ABSOLUTE_GRADIENT_DEGREES':
    print(f"  Ranking by: {ABS_GRAD_DEG_COLUMN}")
else:
    print(f"Error: Unknown RANKING_STRATEGY '{RANKING_STRATEGY}'. Exiting.")
    sys.exit(1)
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

    output_dir_name = f"outputs_{place_slug}_r{radius_km}km_{strategy_slug}"
    output_dir = os.path.join(base_dir, output_dir_name)
    paths = {
        "output_dir": output_dir,
        "temp_bbox_pbf": os.path.join(output_dir, "temp_bbox_extract.osm.pbf"),
        "temp_trails_pbf": os.path.join(output_dir, "temp_filtered_trails_initial.osm.pbf"),
        "temp_trails_geojson": os.path.join(output_dir, "temp_filtered_trails.geojson"),
        "temp_access_geojson": os.path.join(output_dir, "temp_access_points.geojson"),
        "stats_csv": os.path.join(output_dir, f"{place_slug}_trail_stats_all.csv"), # Comprehensive stats
        "ranked_stats_csv": os.path.join(output_dir, f"{place_slug}_trail_stats_ranked.csv"), # Filtered and ranked
        "top_details_csv": os.path.join(output_dir, f"{place_slug}_top_{N_TOP_TRAILS}_details.csv"),
        "visualization_png": os.path.join(output_dir, f"{place_slug}_top_{N_TOP_TRAILS}_map.png"),
    }
    return paths

PATHS = generate_paths(OUTPUT_BASE_DIR, PLACE_NAME, RADIUS_KM, RANKING_STRATEGY, TARGET_SLOPE_PERCENT, TARGET_SLOPE_PENALTY_TYPE_FOR_PCT_SCORE)
print(f"Ensuring output directory exists: {PATHS['output_dir']}")
os.makedirs(PATHS['output_dir'], exist_ok=True)
if not os.path.isdir(PATHS['output_dir']):
    print(f"FATAL ERROR: Failed to create output directory: {PATHS['output_dir']}"); sys.exit(1)

# --- Helper Functions (get_bounding_box, run_osmium_command, extract_osm_features, standardize_and_reproject) ---
def get_bounding_box(latitude, longitude, radius_km):
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
    print(f"  Running osmium: {description}...")
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  Error during '{description}':\n{result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  Error: Osmium command '{description}' timed out.")
        return False
    except Exception as e:
        print(f"  Failed to run osmium command '{description}': {e}")
        return False

def extract_osm_features(pbf_path, bbox_ll, tags_filter, output_geojson_path, description, add_id=True):
    print(f"Extracting {description} features...")
    temp_dir = os.path.dirname(output_geojson_path)
    temp_extract_pbf = os.path.join(temp_dir, f"temp_extract_{description.replace(' ', '_')}.osm.pbf")
    temp_filter_pbf = os.path.join(temp_dir, f"temp_filter_{description.replace(' ', '_')}.osm.pbf")

    bbox_str = f"{bbox_ll[0]},{bbox_ll[1]},{bbox_ll[2]},{bbox_ll[3]}"
    extract_cmd = ["osmium", "extract", "-b", bbox_str, pbf_path, "-o", temp_extract_pbf, "--overwrite"]
    if not run_osmium_command(extract_cmd, f"bbox extract for {description}"): return None

    tag_filter_parts = []
    for key, values in tags_filter.items():
        tag_filter_parts.append(f"nwr/{key}={','.join(values)}")
    filter_cmd = ["osmium", "tags-filter", temp_extract_pbf] + tag_filter_parts + ["-o", temp_filter_pbf, "--overwrite"]
    if not run_osmium_command(filter_cmd, f"tags filter for {description}"):
        if os.path.exists(temp_extract_pbf): os.remove(temp_extract_pbf)
        return None

    export_cmd_base = ["osmium", "export", temp_filter_pbf]
    if add_id: export_cmd_base.extend(["--add-unique-id=type_id"])
    export_cmd = export_cmd_base + ["-o", output_geojson_path, "--overwrite"] # Corrected
    if not run_osmium_command(export_cmd, f"geojson export for {description}"):
        for f_path in [temp_extract_pbf, temp_filter_pbf, output_geojson_path]:
            if os.path.exists(f_path): os.remove(f_path)
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
            return gdf
        except Exception as read_err:
            print(f"  Error reading GeoJSON file for {description}: {read_err}")
            return None
    else:
        print(f"  GeoJSON file for {description} is empty or does not exist.")
        return gpd.GeoDataFrame()

def standardize_and_reproject(gdf, target_crs, id_column=None, expected_geojson_id_col='id',
                              optional_tags_as_cols=None, other_tags_col_name=None):
    if gdf is None or gdf.empty: return gdf
    print(f"Standardizing and reprojecting GeoDataFrame (Initial rows: {len(gdf)})...")
    if id_column and expected_geojson_id_col and expected_geojson_id_col in gdf.columns:
        if id_column != expected_geojson_id_col: gdf = gdf.rename(columns={expected_geojson_id_col: id_column})
        try: gdf[id_column] = gdf[id_column].astype(str).str.extract(r'(\d+)$', expand=False).fillna(gdf[id_column]).astype(str)
        except Exception as e: print(f"  Warning: Could not extract numeric part from ID '{id_column}': {e}.")
    if other_tags_col_name:
        cols_to_keep = ['geometry'] + ([id_column] if id_column and id_column in gdf.columns else []) + \
                       ([tag for tag in optional_tags_as_cols if tag in gdf.columns] if optional_tags_as_cols else [])
        cols_to_consolidate = [col for col in gdf.columns if col not in cols_to_keep]
        if cols_to_consolidate:
            gdf[other_tags_col_name] = gdf[cols_to_consolidate].apply(lambda row: row.dropna().to_dict(), axis=1)
            gdf = gdf.drop(columns=cols_to_consolidate)
        else: gdf[other_tags_col_name] = [{} for _ in range(len(gdf))]
    if gdf.crs is None: gdf.crs = "EPSG:4326"; print(f"  Warning: Input GDF has no CRS. Assuming EPSG:4326.")
    if gdf.crs.to_string().upper() != target_crs.upper():
        try: gdf = gdf.to_crs(target_crs); print(f"  Reprojected to {target_crs}.")
        except Exception as e: print(f"  Error during reprojection to {target_crs}: {e}"); return None
    return gdf

# --- Score Calculation Functions ---
def calculate_target_slope_percent_score(avg_slope_percent_data,
                                         target_slope_pct_config,
                                         max_slope_pct_for_penalty_norm_config,
                                         penalty_type_config,
                                         output_nodata_val=-1.0):
    """ Calculates 0-1 score based on proximity to target slope percentage. """
    if max_slope_pct_for_penalty_norm_config <= 0:
        print("    ERROR: max_slope_pct_for_penalty_norm_config must be positive for TARGET_SLOPE_PERCENT_SCORE.")
        return pd.Series(np.full(len(avg_slope_percent_data), output_nodata_val), dtype=np.float32, index=avg_slope_percent_data.index if isinstance(avg_slope_percent_data, pd.Series) else None)
    slope_values = avg_slope_percent_data.values.astype(np.float32) if isinstance(avg_slope_percent_data, pd.Series) else np.array(avg_slope_percent_data, dtype=np.float32)
    mask = np.isnan(slope_values)
    slope_masked = np.ma.array(slope_values, mask=mask, fill_value=output_nodata_val)
    actual_slope_pct_normalized = np.ma.clip(slope_masked, 0, max_slope_pct_for_penalty_norm_config) / max_slope_pct_for_penalty_norm_config
    target_slope_pct_normalized = np.clip(float(target_slope_pct_config), 0, max_slope_pct_for_penalty_norm_config) / max_slope_pct_for_penalty_norm_config
    difference = actual_slope_pct_normalized - target_slope_pct_normalized
    if penalty_type_config.lower() == 'quadratic': score = 1.0 - np.ma.power(difference, 2)
    elif penalty_type_config.lower() == 'linear': score = 1.0 - np.ma.abs(difference)
    else: score = 1.0 - np.ma.power(difference, 2); print(f"    WARNING: Unknown penalty_type '{penalty_type_config}' for TARGET_SLOPE_PERCENT_SCORE. Defaulting to quadratic.")
    score_clipped = np.ma.clip(score, 0, 1)
    result_array = score_clipped.filled(output_nodata_val).astype(np.float32)
    return pd.Series(result_array, index=avg_slope_percent_data.index, name=TARGET_SLOPE_PERCENT_SCORE_COLUMN) if isinstance(avg_slope_percent_data, pd.Series) else result_array

def calculate_length_adjusted_target_grade_score(length_data, avg_slope_percent_data,
                                                 target_slope_pct_config,
                                                 output_nodata_val=np.nan): # Use NaN for this score's nodata
    """ Calculates Length * (Target Grade % - L1_Penalty_from_Target_%) score. """
    if isinstance(length_data, pd.Series): length_values = length_data.values.astype(np.float32)
    else: length_values = np.array(length_data, dtype=np.float32)
    if isinstance(avg_slope_percent_data, pd.Series): slope_values = avg_slope_percent_data.values.astype(np.float32)
    else: slope_values = np.array(avg_slope_percent_data, dtype=np.float32)

    mask = np.isnan(slope_values) | np.isnan(length_values) | (length_values <= 0) # Also mask if length is invalid
    # Note: avg_slope_percent_data should be absolute for this calculation.
    slope_masked = np.ma.array(slope_values, mask=mask, fill_value=np.nan) # Use NaN as fill for intermediate
    length_masked = np.ma.array(length_values, mask=mask, fill_value=np.nan)

    target_grade_pct_val = float(target_slope_pct_config)
    penalty = np.ma.abs(target_grade_pct_val - slope_masked)
    effective_grade = target_grade_pct_val - penalty # This can be negative
    score = length_masked * effective_grade
    
    result_array = score.filled(output_nodata_val).astype(np.float32)
    return pd.Series(result_array, index=avg_slope_percent_data.index, name=LENGTH_ADJUSTED_TARGET_GRADE_SCORE_COLUMN) if isinstance(avg_slope_percent_data, pd.Series) else result_array

# --- Core Logic Functions ---
def analyze_elevations(trails_gdf, dem_raster_path, id_column='osmid'):
    print(f"\n--- Step 3: Calculating Elevation Statistics & Slope Metrics ---")
    output_cols = [id_column, 'start_coord_str', 'end_coord_str', 'length_m', 'start_elev_m', 'end_elev_m',
                   NET_GAIN_COLUMN, MAX_ELEVATION_COLUMN, 'avg_uphill_gradient_degrees', ABS_GRAD_DEG_COLUMN,
                   'absolute_avg_gradient_percent']
    if RANKING_STRATEGY == 'TARGET_SLOPE_PERCENT_SCORE': output_cols.append(TARGET_SLOPE_PERCENT_SCORE_COLUMN)
    if RANKING_STRATEGY == 'LENGTH_ADJUSTED_TARGET_GRADE_SCORE': output_cols.append(LENGTH_ADJUSTED_TARGET_GRADE_SCORE_COLUMN)

    if trails_gdf is None or trails_gdf.empty: return pd.DataFrame(columns=output_cols)
    if not os.path.exists(dem_raster_path): print(f"Error: DEM not found: {dem_raster_path}"); return None
    results = []
    print(f"Opening DEM raster: {dem_raster_path}")
    try:
        with rasterio.open(dem_raster_path) as dem_src:
            if trails_gdf.crs != dem_src.crs: print(f"FATAL: Trail CRS ({trails_gdf.crs}) != DEM CRS ({dem_src.crs})."); return None
            dem_nodata = dem_src.nodata
            trails_gdf_proc = trails_gdf.reset_index(drop=True)
            print(f"Processing {len(trails_gdf_proc)} trails for elevation...")
            for i, row in trails_gdf_proc.iterrows():
                geom = row.geometry
                if not isinstance(geom, LineString) or geom.is_empty: continue
                if id_column not in row or pd.isna(row[id_column]): continue
                
                # Simplified get_elevation_stats_single_trail logic inline
                start_pt, end_pt = geom.interpolate(0), geom.interpolate(geom.length)
                sample_coords = [(p.x, p.y) for p in [start_pt, end_pt] if isinstance(p, Point)]
                start_elev, end_elev, net_gain, max_elev_val, grad_deg = np.nan, np.nan, np.nan, np.nan, np.nan
                if len(sample_coords) == 2:
                    try:
                        sampled_elevs = np.array([val[0] for val in dem_src.sample(sample_coords)], dtype=np.float32)
                        valid_mask = ~np.isnan(sampled_elevs)
                        if dem_nodata is not None:
                            valid_mask &= (~np.isclose(sampled_elevs, dem_nodata) if np.issubdtype(type(dem_nodata), np.floating) else (sampled_elevs != dem_nodata))
                        if valid_mask.sum() == 2:
                            start_elev, end_elev = sampled_elevs[0], sampled_elevs[1]
                            net_gain = end_elev - start_elev
                            grad_deg = 0.0 if geom.length < 1e-6 else np.degrees(np.arctan2(net_gain, geom.length))
                            max_elev_val = np.nanmax(sampled_elevs[valid_mask])
                    except Exception: pass

                abs_avg_grad_deg = abs(grad_deg) if pd.notna(grad_deg) else np.nan
                abs_avg_grad_pct = math.tan(math.radians(abs_avg_grad_deg)) * 100.0 if pd.notna(abs_avg_grad_deg) else np.nan
                
                results.append({
                    id_column: str(row[id_column]), 'start_coord_str': str(geom.coords[0]), 'end_coord_str': str(geom.coords[-1]),
                    'length_m': geom.length, 'start_elev_m': start_elev, 'end_elev_m': end_elev, NET_GAIN_COLUMN: net_gain,
                    MAX_ELEVATION_COLUMN: max_elev_val, 'avg_uphill_gradient_degrees': grad_deg, ABS_GRAD_DEG_COLUMN: abs_avg_grad_deg,
                    'absolute_avg_gradient_percent': abs_avg_grad_pct
                })
            stats_df = pd.DataFrame(results)
            if stats_df.empty: return pd.DataFrame(columns=output_cols)

            if RANKING_STRATEGY == 'TARGET_SLOPE_PERCENT_SCORE':
                if 'absolute_avg_gradient_percent' in stats_df.columns:
                    stats_df[TARGET_SLOPE_PERCENT_SCORE_COLUMN] = calculate_target_slope_percent_score(
                        stats_df['absolute_avg_gradient_percent'], TARGET_SLOPE_PERCENT,
                        MAX_SLOPE_PERCENT_FOR_PCT_SCORE_NORMALIZATION, TARGET_SLOPE_PENALTY_TYPE_FOR_PCT_SCORE)
                else: stats_df[TARGET_SLOPE_PERCENT_SCORE_COLUMN] = np.nan
            
            if RANKING_STRATEGY == 'LENGTH_ADJUSTED_TARGET_GRADE_SCORE':
                if 'length_m' in stats_df.columns and 'absolute_avg_gradient_percent' in stats_df.columns:
                    stats_df[LENGTH_ADJUSTED_TARGET_GRADE_SCORE_COLUMN] = calculate_length_adjusted_target_grade_score(
                        stats_df['length_m'], stats_df['absolute_avg_gradient_percent'], TARGET_SLOPE_PERCENT)
                else: stats_df[LENGTH_ADJUSTED_TARGET_GRADE_SCORE_COLUMN] = np.nan
            
            final_cols_present = [col for col in output_cols if col in stats_df.columns]
            return stats_df[final_cols_present]

    except Exception as e: print(f"Error in analyze_elevations: {e}"); traceback.print_exc(); return None


def calculate_min_distances(trails_gdf, access_points_gdf, id_col):
    if trails_gdf is None or trails_gdf.empty or access_points_gdf is None or access_points_gdf.empty: return None
    if trails_gdf.crs != access_points_gdf.crs: return None
    trails_centroids = trails_gdf.copy(); trails_centroids['geometry'] = trails_centroids.geometry.centroid
    try:
        joined_gdf = gpd.sjoin_nearest(trails_centroids[[id_col, 'geometry']], access_points_gdf[['geometry']], how='left', distance_col="dist_temp")
        if 'dist_temp' not in joined_gdf.columns: # Fallback for older geopandas
             joined_gdf['dist_temp'] = joined_gdf.geometry.distance(gpd.GeoSeries(access_points_gdf.loc[joined_gdf['index_right'], 'geometry'].values, index=joined_gdf.index))
    except Exception: return None
    return joined_gdf.groupby(id_col)['dist_temp'].min().rename("min_distance")

def rank_trails(stats_df_input, id_col, length_col, net_gain_col, ranking_strategy_config,
                abs_grad_deg_col_config, target_slope_pct_score_col_config, len_adj_target_grade_score_col_config,
                rank_col_name_config, rank_descending_config,
                min_length_filter_config, absolute_min_net_gain_filter_config):
    print("\n--- Step 5: Filtering (Length/Gain) and Ranking Trails ---")
    if stats_df_input is None or stats_df_input.empty: print("Input stats empty for ranking."); return pd.DataFrame()
    df_processed = stats_df_input.copy()
    
    if ranking_strategy_config == 'ABSOLUTE_GRADIENT_DEGREES': rank_by_col_actual = abs_grad_deg_col_config
    elif ranking_strategy_config == 'TARGET_SLOPE_PERCENT_SCORE': rank_by_col_actual = target_slope_pct_score_col_config
    elif ranking_strategy_config == 'LENGTH_ADJUSTED_TARGET_GRADE_SCORE': rank_by_col_actual = len_adj_target_grade_score_col_config
    else: print(f"Error: Unknown RANKING_STRATEGY '{ranking_strategy_config}' in rank_trails."); return None
    print(f"Ranking by column: '{rank_by_col_actual}'")

    required_cols = [id_col, length_col, rank_by_col_actual] + ([net_gain_col] if absolute_min_net_gain_filter_config and absolute_min_net_gain_filter_config > 0 else [])
    if any(col not in df_processed.columns for col in required_cols): print(f"Error: Missing required columns for ranking: {required_cols}"); return None
    
    df_processed.dropna(subset=[length_col, rank_by_col_actual] + ([net_gain_col] if net_gain_col in df_processed.columns else []), inplace=True)
    if min_length_filter_config and min_length_filter_config > 0: df_processed = df_processed[df_processed[length_col] >= min_length_filter_config]
    if absolute_min_net_gain_filter_config and absolute_min_net_gain_filter_config > 0 and net_gain_col in df_processed.columns:
        df_processed = df_processed[df_processed[net_gain_col].abs() >= absolute_min_net_gain_filter_config]
    if df_processed.empty: print("No data remaining after filters."); return pd.DataFrame()

    ranked_df = df_processed.sort_values(by=rank_by_col_actual, ascending=(not rank_descending_config))
    ranked_df[rank_col_name_config] = range(1, len(ranked_df) + 1)
    print(f"Ranking complete. Final ranked DataFrame has {len(ranked_df)} rows.")
    return ranked_df

def create_top_trails_details_df(top_trails_merged_gdf, n_top, id_col, rank_col,
                                 actual_ranking_score_col, # The specific score column used for this run's ranking
                                 length_col, center_lat, center_lon,
                                 optional_osm_tags_config, other_tags_col_config, dist_cols_list):
    print(f"\n--- Step 7: Creating Detailed Output DataFrame for Top {n_top} ---")
    if top_trails_merged_gdf is None or top_trails_merged_gdf.empty: return pd.DataFrame()
    details = []
    # Define all potentially relevant score columns
    all_score_cols = [ABS_GRAD_DEG_COLUMN, 'absolute_avg_gradient_percent', TARGET_SLOPE_PERCENT_SCORE_COLUMN, LENGTH_ADJUSTED_TARGET_GRADE_SCORE_COLUMN]
    base_stat_cols = [id_col, rank_col, actual_ranking_score_col, length_col, 'start_elev_m', 'end_elev_m', NET_GAIN_COLUMN, MAX_ELEVATION_COLUMN]
    # Add other score columns if they exist and are not the primary ranking score for this run
    for sc in all_score_cols:
        if sc in top_trails_merged_gdf.columns and sc != actual_ranking_score_col and sc not in base_stat_cols:
            base_stat_cols.append(sc)
    base_stat_cols.extend(dist_cols_list)
    if other_tags_col_config and other_tags_col_config in top_trails_merged_gdf.columns: base_stat_cols.append(other_tags_col_config)
    
    available_base_cols = [col for col in base_stat_cols if col in top_trails_merged_gdf.columns]
    available_osm_tags = [col for col in optional_osm_tags_config if col in top_trails_merged_gdf.columns]
    geom_col_name = top_trails_merged_gdf.geometry.name
    geolocator = Nominatim(user_agent=f"trail_detail_v11.4_{PLACE_NAME.replace(' ','_')}", timeout=10)
    center_coords_wgs84 = (center_lat, center_lon)
    try: top_trails_wgs84 = top_trails_merged_gdf[[geom_col_name]].to_crs("EPSG:4326")
    except Exception: top_trails_wgs84 = None

    for index, row in top_trails_merged_gdf.iterrows():
        trail_data = {col: row.get(col) for col in available_base_cols}
        trail_data.update({col: row.get(col) for col in available_osm_tags})
        lat, lon, placename, dist_center_km = np.nan, np.nan, "Geocoding Failed", np.nan
        if top_trails_wgs84 is not None and index in top_trails_wgs84.index:
            geom_wgs84 = top_trails_wgs84.loc[index, geom_col_name]
            if isinstance(geom_wgs84, LineString) and not geom_wgs84.is_empty:
                centroid = geom_wgs84.centroid; lat, lon = centroid.y, centroid.x
                if all(pd.notna(c) for c in [lat, lon, center_lat, center_lon]):
                    try: dist_center_km = geopy_distance(center_coords_wgs84, (lat, lon)).km
                    except Exception: pass
                if pd.notna(lat) and pd.notna(lon):
                    try:
                        location = geolocator.reverse(f"{lat:.6f}, {lon:.6f}", exactly_one=True, language='en')
                        placename = location.address if location else "Placename not found"
                    except Exception: placename = "Geocoding Error" # Simplified error
                    finally: time.sleep(1.1)
        trail_data.update({'representative_latitude': lat, 'representative_longitude': lon, 'nearest_placename': placename, 'distance_from_center_km': dist_center_km})
        details.append(trail_data)
    details_df = pd.DataFrame(details)
    # Column ordering for CSV
    final_cols_order = [id_col, rank_col, actual_ranking_score_col]
    # Add other scores
    for sc_col in all_score_cols:
        if sc_col in details_df.columns and sc_col != actual_ranking_score_col: final_cols_order.append(sc_col)
    final_cols_order.extend(dist_cols_list)
    final_cols_order.extend(['nearest_placename', 'distance_from_center_km', length_col, NET_GAIN_COLUMN, MAX_ELEVATION_COLUMN])
    final_cols_order.extend([tag for tag in optional_osm_tags_config if tag in details_df.columns])
    if other_tags_col_config and other_tags_col_config in details_df.columns: final_cols_order.append(other_tags_col_config)
    # Add any remaining base columns not yet included
    final_cols_order.extend([b_col for b_col in available_base_cols if b_col not in final_cols_order and b_col in details_df.columns])
    final_cols_order = [fc for i, fc in enumerate(final_cols_order) if fc in details_df.columns and fc not in final_cols_order[:i]] # Unique, existing
    details_df = details_df[final_cols_order]
    # Fill NaNs
    for tag in optional_osm_tags_config:
        if tag in details_df.columns: details_df[tag] = details_df[tag].fillna('N/A')
    for d_col in dist_cols_list:
        if d_col in details_df.columns: details_df[d_col] = details_df[d_col].fillna(-1).round(1)
    if MAX_ELEVATION_COLUMN in details_df.columns: details_df[MAX_ELEVATION_COLUMN] = details_df[MAX_ELEVATION_COLUMN].fillna(-9999).round(1)
    if other_tags_col_config and other_tags_col_config in details_df.columns:
         details_df[other_tags_col_config] = details_df[other_tags_col_config].apply(lambda x: {} if pd.isna(x) else x)
    return details_df

def visualize_results(top_trails_gdf, n_top, place_name, rank_col,
                      actual_ranking_score_col_name, # Specific score col name for title
                      vis_cmap_config, output_png_path):
    print(f"\n--- Step 8: Visualizing Top {n_top} Trails ---")
    if top_trails_gdf is None or top_trails_gdf.empty: print("Input GDF empty for visualization."); return
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        top_trails_gdf.plot(column=rank_col, cmap=vis_cmap_config, ax=ax, legend=False, linewidth=2.5, aspect='equal')
        norm = Normalize(vmin=top_trails_gdf[rank_col].min() - 0.5, vmax=top_trails_gdf[rank_col].max() + 0.5)
        fig.colorbar(ScalarMappable(cmap=vis_cmap_config, norm=norm), ax=ax, label=f"Trail Rank (Top {n_top}, 1=Best)", shrink=0.6)
        try: cx.add_basemap(ax, crs=top_trails_gdf.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik, zoom='auto')
        except Exception as e: print(f"  Warning: Basemap failed: {e}.")
        
        title_suffix = ""
        if RANKING_STRATEGY == 'ABSOLUTE_GRADIENT_DEGREES': title_suffix = f"Abs. Avg. Gradient ({ABS_GRAD_DEG_COLUMN.replace('_',' ').title()})"
        elif RANKING_STRATEGY == 'TARGET_SLOPE_PERCENT_SCORE': title_suffix = f"Target Slope {TARGET_SLOPE_PERCENT}% Score ({TARGET_SLOPE_PENALTY_TYPE_FOR_PCT_SCORE.title()} Pen.)"
        elif RANKING_STRATEGY == 'LENGTH_ADJUSTED_TARGET_GRADE_SCORE': title_suffix = f"Length-Adj. Target Grade {TARGET_SLOPE_PERCENT}% Score"
        ax.set_title(f"Top {n_top} Trails near {place_name}\n(Ranked by: {title_suffix})", fontsize=14)
        ax.set_axis_off()

        text_kwargs = dict(ha='center', va='center', fontsize=8, color='black', path_effects=[PathEffects.withStroke(linewidth=2, foreground='white')])
        for idx, row in top_trails_gdf.iterrows():
            ax.text(row.geometry.representative_point().x, row.geometry.representative_point().y, str(row[rank_col]), **text_kwargs)
        
        minx, miny, maxx, maxy = top_trails_gdf.total_bounds
        if np.all(np.isfinite([minx, miny, maxx, maxy])):
            x_pad, y_pad = (maxx-minx)*0.05 or 1000, (maxy-miny)*0.05 or 1000
            ax.set_xlim(minx-x_pad, maxx+x_pad); ax.set_ylim(miny-y_pad, maxy+y_pad)
        plt.tight_layout()
        plt.savefig(output_png_path, dpi=300, bbox_inches='tight'); plt.close(fig)
        print(f"  Visualization saved: {os.path.basename(output_png_path)}")
    except Exception as e: print(f"  Error during visualization: {e}"); traceback.print_exc(); plt.close('all')

# =============================================================================
# --- Main Execution Workflow ---
# =============================================================================
if __name__ == "__main__":
    # --- 1. Geocoding and Bounding Box ---
    print(f"\n--- Step 1: Geocoding and Bounding Box ---")
    geolocator = Nominatim(user_agent=f"trail_search_v11.4_{PLACE_NAME.replace(' ','_')}")
    try:
        location = geolocator.geocode(PLACE_NAME, timeout=20)
        if not location: print(f"Error: Could not geocode '{PLACE_NAME}'."); sys.exit(1)
        center_lat, center_lon = location.latitude, location.longitude
        print(f"Coordinates for '{PLACE_NAME}': Lat={center_lat:.4f}, Lon={center_lon:.4f}")
        bounds_ll = get_bounding_box(center_lat, center_lon, RADIUS_KM)
    except Exception as e: print(f"Geocoding/BBox error: {e}"); traceback.print_exc(); sys.exit(1)

    # --- 2. Extract & Process Trail Geometries & Tags ---
    print(f"\n--- Step 2: Extract & Process Trail Geometries & Tags ---")
    trails_gdf_raw = extract_osm_features(PBF_FILE_PATH, bounds_ll, TRAIL_TAGS, PATHS["temp_trails_geojson"], "trails")
    analysis_input_gdf = standardize_and_reproject(trails_gdf_raw, TARGET_CRS, ID_COLUMN, 'id', OPTIONAL_OSM_TAGS, OTHER_TAGS_COLUMN_NAME)
    if analysis_input_gdf is None or analysis_input_gdf.empty: print("Trail extraction/processing failed. Exiting."); sys.exit(1)
    
    # --- 2b. Apply Tag Exclusion Filters ---
    initial_count = len(analysis_input_gdf)
    print(f"\n--- Step 2b: Applying Tag Exclusion Filters (Initial: {initial_count} trails) ---")
    for tag_key, excluded_vals in EXCLUDE_TAG_VALUES.items():
        if tag_key in analysis_input_gdf.columns:
            mask = ~analysis_input_gdf[tag_key].fillna('').astype(str).str.lower().isin([str(v).lower() for v in excluded_vals])
            analysis_input_gdf = analysis_input_gdf[mask]
    print(f"Trails remaining after tag exclusion: {len(analysis_input_gdf)}")
    if analysis_input_gdf.empty: print("No trails after tag filters. Exiting."); sys.exit(0)

    # --- 3. Analyze Elevations & Calculate Scores ---
    stats_df = analyze_elevations(analysis_input_gdf, PROJECTED_DEM_PATH, ID_COLUMN)
    if stats_df is None or stats_df.empty: print("Elevation analysis failed. Exiting."); sys.exit(1)

    # --- 4. Calculate Distances to Access Points ---
    dist_cols_added = []
    if CALCULATE_ACCESS_DISTANCE:
        print(f"\n--- Step 4: Calculate Distances to Access Points ---")
        for ap_type, ap_tags, ap_col in [("car parks", {"amenity": ["parking"]}, DIST_CARPARK_COL),
                                         ("access roads", {"highway": ACCESS_ROAD_HIGHWAY_VALUES}, DIST_ROAD_COL)]:
            ap_gdf_raw = extract_osm_features(PBF_FILE_PATH, bounds_ll, ap_tags, PATHS["temp_access_geojson"], ap_type, add_id=False)
            ap_gdf = standardize_and_reproject(ap_gdf_raw, TARGET_CRS)
            if ap_gdf is not None and not ap_gdf.empty:
                if ap_type == "access roads": ap_gdf = ap_gdf[ap_gdf.geometry.geom_type.isin(['LineString', 'MultiLineString'])]
                if not ap_gdf.empty:
                    min_dist_series = calculate_min_distances(analysis_input_gdf[[ID_COLUMN, 'geometry']], ap_gdf, ID_COLUMN)
                    if min_dist_series is not None:
                        stats_df = stats_df.merge(min_dist_series.rename(ap_col), on=ID_COLUMN, how='left')
                        dist_cols_added.append(ap_col); print(f"  Added '{ap_col}' column.")
        if os.path.exists(PATHS["temp_access_geojson"]): os.remove(PATHS["temp_access_geojson"])

    # --- Merge all OSM tags from analysis_input_gdf to stats_df for comprehensive output ---
    print("\n--- Merging Full OSM Tags into Statistics DataFrame ---")
    merge_cols = [ID_COLUMN] + [col for col in OPTIONAL_OSM_TAGS if col in analysis_input_gdf.columns] + \
                 ([OTHER_TAGS_COLUMN_NAME] if OTHER_TAGS_COLUMN_NAME in analysis_input_gdf.columns else [])
    stats_df[ID_COLUMN] = stats_df[ID_COLUMN].astype(str) # Ensure consistent ID type
    analysis_input_gdf[ID_COLUMN] = analysis_input_gdf[ID_COLUMN].astype(str)
    stats_df_full = pd.merge(stats_df, analysis_input_gdf[list(set(merge_cols))], on=ID_COLUMN, how='left', suffixes=('', '_tag_dup'))
    dup_tag_cols = [c for c in stats_df_full.columns if c.endswith('_tag_dup')]
    if dup_tag_cols: stats_df_full.drop(columns=dup_tag_cols, inplace=True)
    
    # Save comprehensive stats (all trails, all calculated metrics and tags)
    try:
        print(f"Saving comprehensive statistics (all trails, pre-filter/rank) to: {PATHS['stats_csv']}")
        # Prepare for CSV: fill NaNs in tag columns, convert dicts to strings
        for tag in OPTIONAL_OSM_TAGS:
            if tag in stats_df_full.columns: stats_df_full[tag] = stats_df_full[tag].fillna('N/A')
        if OTHER_TAGS_COLUMN_NAME in stats_df_full.columns:
             stats_df_full[OTHER_TAGS_COLUMN_NAME] = stats_df_full[OTHER_TAGS_COLUMN_NAME].apply(lambda x: {} if pd.isna(x) else x).astype(str)
        for dcol in dist_cols_added:
            if dcol in stats_df_full.columns: stats_df_full[dcol] = stats_df_full[dcol].fillna(-1.0)
        stats_df_full.to_csv(PATHS['stats_csv'], index=False, float_format='%.4f')
    except Exception as e: print(f"  Warning: Could not save comprehensive stats CSV: {e}"); traceback.print_exc()

    # --- 5. Filter and Rank Trails ---
    ranked_stats_df = rank_trails(
        stats_df_full, ID_COLUMN, 'length_m', NET_GAIN_COLUMN, RANKING_STRATEGY,
        ABS_GRAD_DEG_COLUMN, TARGET_SLOPE_PERCENT_SCORE_COLUMN, LENGTH_ADJUSTED_TARGET_GRADE_SCORE_COLUMN,
        RANK_COLUMN_NAME, RANK_DESCENDING, MIN_TRAIL_LENGTH_FILTER, ABSOLUTE_MIN_NET_GAIN)
    if ranked_stats_df is None or ranked_stats_df.empty: print("Ranking/filtering yielded no results. Exiting."); sys.exit(1)
    try:
        print(f"Saving filtered and ranked statistics to: {PATHS['ranked_stats_csv']}")
        # Ensure 'other_osm_tags' is string if it exists (should be from stats_df_full)
        if OTHER_TAGS_COLUMN_NAME in ranked_stats_df.columns and not pd.api.types.is_string_dtype(ranked_stats_df[OTHER_TAGS_COLUMN_NAME]):
            ranked_stats_df[OTHER_TAGS_COLUMN_NAME] = ranked_stats_df[OTHER_TAGS_COLUMN_NAME].astype(str)
        ranked_stats_df.to_csv(PATHS['ranked_stats_csv'], index=False, float_format='%.4f')
    except Exception as e: print(f"  Warning: Could not save ranked statistics CSV: {e}"); traceback.print_exc()

    # --- 6. Prepare Data for Top N Trails (Geometries + All Stats/Tags from ranked_stats_df) ---
    print(f"\n--- Step 6: Preparing Full Data for Top {N_TOP_TRAILS} Trails ---")
    top_n_from_ranked_df = ranked_stats_df.head(N_TOP_TRAILS).copy() # This already has all stats and stringified tags
    # Merge geometry back onto this top N selection
    top_trails_final_gdf = pd.merge(top_n_from_ranked_df, analysis_input_gdf[[ID_COLUMN, 'geometry']], on=ID_COLUMN, how='left')
    top_trails_final_gdf = gpd.GeoDataFrame(top_trails_final_gdf, geometry='geometry', crs=TARGET_CRS)
    top_trails_final_gdf.sort_values(by=RANK_COLUMN_NAME, inplace=True) # Ensure sorted by rank
    if top_trails_final_gdf.empty: print(f"Could not prepare GDF for top {N_TOP_TRAILS} trails. Exiting."); sys.exit(1)

    # --- 7. Create Detailed CSV for Top Trails ---
    # Determine the actual score column that was used for ranking to pass to details function
    if RANKING_STRATEGY == 'ABSOLUTE_GRADIENT_DEGREES': actual_rank_score_col = ABS_GRAD_DEG_COLUMN
    elif RANKING_STRATEGY == 'TARGET_SLOPE_PERCENT_SCORE': actual_rank_score_col = TARGET_SLOPE_PERCENT_SCORE_COLUMN
    elif RANKING_STRATEGY == 'LENGTH_ADJUSTED_TARGET_GRADE_SCORE': actual_rank_score_col = LENGTH_ADJUSTED_TARGET_GRADE_SCORE_COLUMN
    else: actual_rank_score_col = "UNKNOWN_RANK_COL" # Should not happen

    top_details_df = create_top_trails_details_df(
        top_trails_final_gdf, N_TOP_TRAILS, ID_COLUMN, RANK_COLUMN_NAME, actual_rank_score_col,
        'length_m', center_lat, center_lon, OPTIONAL_OSM_TAGS, OTHER_TAGS_COLUMN_NAME, dist_cols_added)
    if top_details_df is not None and not top_details_df.empty:
        print("\n--- Top Trail Details (Sample) ---")
        # Display logic for top_details_df (simplified for brevity here, full logic in function)
        print(top_details_df.head(min(N_TOP_TRAILS, 5))) # Print first few rows
        try:
            print(f"\nSaving detailed top trail data to: {PATHS['top_details_csv']}")
            # Ensure 'other_osm_tags' is string for CSV (should be from ranked_stats_df)
            if OTHER_TAGS_COLUMN_NAME in top_details_df.columns and not pd.api.types.is_string_dtype(top_details_df[OTHER_TAGS_COLUMN_NAME]):
                 top_details_df[OTHER_TAGS_COLUMN_NAME] = top_details_df[OTHER_TAGS_COLUMN_NAME].apply(json.dumps)
            top_details_df.to_csv(PATHS['top_details_csv'], index=False, float_format='%.6f')
        except Exception as e: print(f"  Warning: Could not save detailed CSV: {e}"); traceback.print_exc()

    # --- 8. Visualize Top Trails ---
    visualize_results(top_trails_final_gdf, N_TOP_TRAILS, PLACE_NAME, RANK_COLUMN_NAME, actual_rank_score_col, VIS_COLORMAP, PATHS["visualization_png"])

    print(f"\n--- Workflow Finished ---")
    print(f"End time: {np.datetime64('now', 's')} UTC")
    print(f"Outputs saved in: {PATHS['output_dir']}")
