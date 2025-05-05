#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to find the minimum distance from full trail geometries (provided in a CSV)
to the nearest water source/amenity within different categories identified in OSM data.

Version 2.9:
- Fixed ambiguous truth value error in calculate_min_distance_to_features
  by explicitly checking possible_matches_index.empty.
Version 2.8:
- Simplified standardize_and_reproject: Removed all buffer(0) calls.
  Checks validity only *after* reprojection. Reports invalid geometries
  with explain_validity but keeps them in the GDF.
- calculate_min_distance_to_features skips invalid geometries.
Version 2.7:
- Modified standardize_and_reproject: Reproject first, then check validity,
  then attempt buffer(0) fix only on invalid geometries. Keep invalid geometries
  in the GDF but print warnings.
- Added check for invalid trail geometry within calculate_min_distance_to_features.
Version 2.6:
- Corrected ID handling in extract_osm_features: Look for '@id' column when using --attributes=id.
Version 2.5:
- Added '--attributes=id' to osmium export command for trails to ensure ID is included.
[...]

Requires: osmium-tool, geopandas, pandas, numpy, rtree, shapely
Install required libraries:
pip install geopandas pandas numpy osmium rtree shapely
"""

import os
import sys
import subprocess
import shutil
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Point, LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import explain_validity # Import explain_validity
import math
import json # For saving other_tags if present
import time

# Check for rtree dependency
try:
    import rtree
except ImportError:
    print("Error: The 'rtree' library is required for spatial indexing but is not installed.")
    print("Please install it, e.g., using 'pip install rtree' or 'conda install rtree'.")
    sys.exit(1)

# =============================================================================
# --- Configuration ---
# =============================================================================

# --- Input/Output Files ---
#INPUT_CSV_PATH = "kendal_uk_top_20_details.csv" # <- UPDATE PATH AS NEEDED
#INPUT_CSV_PATH = "/home/ajay/Python_Projects/steepest_trails_search/outputs_kendal_uk_radius25km_consolidated/kendal_uk_top_20_details.csv" # <- UPDATE PATH AS NEEDED
INPUT_CSV_PATH = "/home/ajay/Python_Projects/steepest_trails_search/outputs_edale_uk_radius25km_consolidated/edale_uk_top_20_details.csv" # <- UPDATE PATH AS NEEDED

OUTPUT_CSV_PATH = os.path.join(os.path.dirname(INPUT_CSV_PATH), "top_20_details_full_geom_dist_categorized_v2.9.csv") # Output in the same folder (updated version)

# --- OSM Data ---
PBF_FILE_PATH = "united-kingdom-latest.osm.pbf" # <- UPDATE PATH AS NEEDED
if not os.path.exists(PBF_FILE_PATH): print(f"Error: OSM PBF file not found at {PBF_FILE_PATH}"); sys.exit(1)

# --- CRS ---
TARGET_CRS = "EPSG:27700"
SOURCE_CRS = "EPSG:4326"

# --- Column Names ---
ID_COLUMN = 'osmid' # This is the target column name in the final DataFrame
DIST_RELIABLE_PUBLIC_WATER_COL = 'min_dist_reliable_public_water_m'
DIST_POTENTIAL_PUBLIC_WATER_COL = 'min_dist_potential_public_water_m'
DIST_NATURAL_WATER_COL = 'min_dist_natural_water_m'
DIST_SHOP_COL = 'min_dist_shop_m'
DIST_AMENITY_COL = 'min_dist_amenity_m'
DIST_TOURISM_COL = 'min_dist_tourism_m'
DISTANCE_COLUMNS = [
    DIST_RELIABLE_PUBLIC_WATER_COL, DIST_POTENTIAL_PUBLIC_WATER_COL,
    DIST_NATURAL_WATER_COL, DIST_SHOP_COL, DIST_AMENITY_COL, DIST_TOURISM_COL
]

# Error/Fallback Values
NO_SOURCE_FOUND_VALUE = -1.0
CALCULATION_ERROR_VALUE = -2.0

# --- Trail Identification ---
TRAIL_HIGHWAY_TAGS = ["path", "footway", "track", "steps", "bridleway", "cycleway"]

# --- Source Feature Identification Tags ---
# (Source tags remain the same as v2.3)
RELIABLE_PUBLIC_WATER_TAGS = { "amenity": ["drinking_water"], "amenity_conditional": ["fountain"], "man_made_conditional": ["water_tap"] }
CONDITIONAL_TAG = "drinking_water"
POTENTIAL_PUBLIC_WATER_TAGS = { "amenity": ["fountain", "toilets"], "man_made": ["water_tap"] }
NATURAL_WATER_TAGS = { "natural": ["spring", "stream"], "waterway": ["stream"] }
SHOP_TAGS = { "shop": ["convenience", "supermarket", "general", "kiosk", "farm", "greengrocer", "bakery", "butcher", "newsagent", "stationery", "outdoor", "sports", "bicycle"] }
AMENITY_TAGS = { "amenity": ["pub", "cafe", "restaurant", "fast_food", "community_centre", "bar", "biergarten", "food_court", "visitor_centre", "place_of_worship", "library", "post_office", "pharmacy", "fuel"] }
TOURISM_TAGS = { "tourism": ["camp_site", "caravan_site", "guest_house", "hotel", "hostel", "information", "picnic_site", "alpine_hut", "motel", "wilderness_hut", "chalet"] }

# Combine all source tags for efficient extraction
ALL_SOURCE_TAGS = {}
ALL_SOURCE_TAGS.update(RELIABLE_PUBLIC_WATER_TAGS); ALL_SOURCE_TAGS.update(POTENTIAL_PUBLIC_WATER_TAGS); ALL_SOURCE_TAGS.update(NATURAL_WATER_TAGS); ALL_SOURCE_TAGS.update(SHOP_TAGS); ALL_SOURCE_TAGS.update(AMENITY_TAGS); ALL_SOURCE_TAGS.update(TOURISM_TAGS)
combined_tags_for_extraction = {}
for key, values in ALL_SOURCE_TAGS.items():
    actual_key = key.replace("_conditional", "")
    if actual_key not in combined_tags_for_extraction: combined_tags_for_extraction[actual_key] = []
    combined_tags_for_extraction[actual_key].extend(values)
for key in combined_tags_for_extraction: combined_tags_for_extraction[key] = list(set(combined_tags_for_extraction[key]))
print(f"Combined tags for source extraction: {combined_tags_for_extraction}")

# --- Bounding Box Buffer ---
BBOX_BUFFER_DEGREES = 0.05

# =============================================================================
# --- Helper Functions ---
# =============================================================================

def run_osmium_command(cmd, description):
    # (Same as v2.6)
    print(f"Running osmium command: {description}")
    print(f"Command: {' '.join(cmd)}")
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0: print(f"Error during '{description}':\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"); return False
        elapsed = time.time() - start_time
        print(f"Osmium command '{description}' completed successfully in {elapsed:.2f} seconds.")
        return True
    except FileNotFoundError: print(f"Error: 'osmium' command not found. Is osmium-tool installed and in your PATH?"); return False
    except Exception as e: elapsed = time.time() - start_time; print(f"Failed to run osmium command '{description}' after {elapsed:.2f} seconds: {e}"); return False

def extract_osm_features(pbf_path, bbox_ll, tags_filter, element_type, output_geojson_path, description, add_id=False, id_list=None, attributes=None):
    # (Same as v2.6)
    print(f"\n--- Extracting {description} ---")
    output_dir = os.path.dirname(output_geojson_path)
    os.makedirs(output_dir, exist_ok=True)
    safe_desc = description.replace(" ", "_").lower()
    temp_extract_pbf = os.path.join(output_dir, f"temp_extract_{safe_desc}.osm.pbf")
    temp_filter_pbf = os.path.join(output_dir, f"temp_filter_{safe_desc}.osm.pbf")
    final_pbf_for_export = temp_extract_pbf

    # Step 1: Extract BBox
    bbox_str = f"{bbox_ll[0]:.7f},{bbox_ll[1]:.7f},{bbox_ll[2]:.7f},{bbox_ll[3]:.7f}"
    extract_cmd = ["osmium", "extract", "-b", bbox_str, pbf_path, "-o", temp_extract_pbf, "--overwrite", "--strategy=complete_ways"]
    if not run_osmium_command(extract_cmd, f"bbox extract for {description}"):
        if os.path.exists(temp_extract_pbf): os.remove(temp_extract_pbf); return None

    # Step 2: Filter by Tags
    use_tags_filter_step = bool(tags_filter)
    if use_tags_filter_step:
        filter_cmd_base = ["osmium", "tags-filter", temp_extract_pbf]
        filter_parts = []
        if tags_filter:
            for key, values in tags_filter.items(): filter_parts.append(f"{element_type}/{key}={','.join(map(str, values))}")
        if not filter_parts: print(f"Warning: No valid tag filters generated for {description}. Skipping filter step."); use_tags_filter_step = False
        else:
            filter_cmd = filter_cmd_base + filter_parts + ["-o", temp_filter_pbf, "--overwrite"]
            if not run_osmium_command(filter_cmd, f"tags filter for {description}"):
                if os.path.exists(temp_extract_pbf): os.remove(temp_extract_pbf)
                if os.path.exists(temp_filter_pbf): os.remove(temp_filter_pbf); return None
            final_pbf_for_export = temp_filter_pbf

    # Step 3: Export to GeoJSON
    export_cmd_base = ["osmium", "export", final_pbf_for_export]
    export_cmd_opts = []
    if attributes: print(f"  Including attributes: {attributes}"); export_cmd_opts.append(f"--attributes={','.join(attributes)}")
    else: print(f"  Note: Exporting features without extra attributes.")
    if element_type == 'w': export_cmd_opts.extend(["--geometry-types=linestring"])
    elif element_type == 'n': export_cmd_opts.extend(["--geometry-types=point"])
    elif element_type == 'nwr': export_cmd_opts.extend(["--geometry-types=point,linestring,polygon"])
    export_cmd = export_cmd_base + export_cmd_opts + ["-o", output_geojson_path, "--overwrite"]
    if not run_osmium_command(export_cmd, f"geojson export for {description}"):
        if os.path.exists(temp_extract_pbf): os.remove(temp_extract_pbf)
        if use_tags_filter_step and os.path.exists(temp_filter_pbf): os.remove(temp_filter_pbf)
        if os.path.exists(output_geojson_path): os.remove(output_geojson_path); return None

    # Step 4: Cleanup
    try:
        if os.path.exists(temp_extract_pbf): os.remove(temp_extract_pbf)
        if use_tags_filter_step and final_pbf_for_export == temp_filter_pbf and os.path.exists(temp_filter_pbf): os.remove(temp_filter_pbf)
    except OSError as e: print(f"Warning: Could not remove temporary PBF file for {description}: {e}")

    # Step 5: Read GeoJSON & Find/Rename ID Column
    print(f"Reading GeoJSON for {description}: {output_geojson_path}...")
    if os.path.exists(output_geojson_path) and os.path.getsize(output_geojson_path) > 0:
        try:
            gdf = gpd.read_file(output_geojson_path)
            id_col_target = ID_COLUMN
            print(f"  GeoDataFrame columns read: {gdf.columns.tolist()}")
            if add_id:
                id_col_osm = '@id'
                if id_col_osm in gdf.columns:
                    print(f"  Found '{id_col_osm}' column. Assuming it's the OSM ID and renaming to '{id_col_target}'.")
                    gdf[id_col_target] = pd.to_numeric(gdf[id_col_osm], errors='coerce').astype('Int64')
                    if id_col_osm != id_col_target: gdf = gdf.drop(columns=[id_col_osm])
                elif 'id' in gdf.columns: # Fallback check
                    print(f"  Warning: '{id_col_osm}' not found, but found 'id'. Using 'id' and renaming to '{id_col_target}'.")
                    source_id_col = 'id'; gdf[id_col_target] = pd.to_numeric(gdf[source_id_col], errors='coerce').astype('Int64')
                    if source_id_col != id_col_target: gdf = gdf.drop(columns=[source_id_col])
                else: print(f"Warning: Could not find '@id' or 'id' property in GeoJSON for {description} to use as '{id_col_target}'.")

            print(f"Successfully read {len(gdf)} {description} features from GeoJSON.")
            if not gdf.empty: print(f"  CRS detected: {gdf.crs}\n  Geometry types found: {gdf.geom_type.value_counts().to_dict()}")
            return gdf
        except Exception as read_err: print(f"Error reading GeoJSON file for {description}: {read_err}"); return None
    else: print(f"GeoJSON file for {description} is empty or does not exist."); return gpd.GeoDataFrame()

def standardize_and_reproject(gdf, target_crs, source_crs=None, id_column_for_warnings=None):
    # (Same as v2.8)
    if gdf is None or gdf.empty: print("GeoDataFrame is empty or None, cannot reproject."); return gdf
    print(f"Standardizing {len(gdf)} features...")
    if gdf.crs is None:
        if source_crs: print(f"Warning: GeoDataFrame has no CRS. Assuming source CRS: {source_crs}"); gdf.crs = source_crs
        else: print("Error: GeoDataFrame has no CRS and no source_crs was provided."); return None
    print(f"  Initial CRS: {gdf.crs}")
    gdf_proj = gdf
    if str(gdf.crs).lower() != str(target_crs).lower():
        print(f"Reprojecting from {gdf.crs} to {target_crs}..."); start_time = time.time()
        try: gdf_proj = gdf.to_crs(target_crs); elapsed = time.time() - start_time; print(f"Reprojection complete in {elapsed:.2f} seconds. New CRS: {gdf_proj.crs}")
        except Exception as e: print(f"Error during reprojection: {e}"); return None
    else: print("Data already in target CRS."); gdf_proj = gdf.copy()
    print("Checking geometry validity post-reprojection...")
    invalid_mask = ~gdf_proj.geometry.is_valid; num_invalid = invalid_mask.sum()
    if num_invalid > 0:
        print(f"Warning: Found {num_invalid} invalid geometries after reprojection.")
        invalid_geoms = gdf_proj[invalid_mask]
        for index, row in invalid_geoms.iterrows():
            geom_id = row[id_column_for_warnings] if id_column_for_warnings and id_column_for_warnings in row else f"Index {index}"
            try: reason = explain_validity(row['geometry']); print(f"  - ID {geom_id}: Invalid - {reason}")
            except Exception as explain_err: print(f"  - ID {geom_id}: Invalid - Error explaining validity: {explain_err}")
    empty_mask = gdf_proj.geometry.is_empty; num_empty = empty_mask.sum(); initial_count = len(gdf_proj)
    if num_empty > 0: print(f"Warning: Removing {num_empty} empty geometries."); gdf_proj = gdf_proj[~empty_mask].copy()
    if gdf_proj.empty and initial_count > 0: print("GeoDataFrame became empty after removing empty geometries.")
    print(f"Standardization complete. Returning {len(gdf_proj)} features (may include some invalid geometries).")
    return gdf_proj


def calculate_min_distance_to_features(trails_gdf, features_gdf, trail_id_col):
    """
    Calculates the minimum distance from each trail LineString to the nearest feature
    in features_gdf. Uses spatial index for efficiency. Skips trails with invalid geometry.
    """
    n_trails = len(trails_gdf)
    n_features = len(features_gdf)
    print(f"Calculating minimum distances from {n_trails} trails to {n_features} features...")

    if trails_gdf is None or trails_gdf.empty: print("Input trails GDF is empty or None."); return pd.Series(dtype=float, name="min_distance")
    if features_gdf is None or features_gdf.empty: print("Input features GDF is empty or None. No sources found."); return pd.Series(NO_SOURCE_FOUND_VALUE, index=trails_gdf[trail_id_col], name="min_distance")
    if trails_gdf.crs != features_gdf.crs: print(f"Error: CRS mismatch between trails ({trails_gdf.crs}) and features ({features_gdf.crs})."); return pd.Series(CALCULATION_ERROR_VALUE, index=trails_gdf[trail_id_col], name="min_distance")

    print("Building spatial index for features (if needed)..."); start_time_idx = time.time()
    try: features_gdf_sindex = features_gdf.sindex; assert features_gdf_sindex is not None; print(f"Using spatial index for features (built/verified in {time.time() - start_time_idx:.2f}s).")
    except Exception as e: print(f"Warning: Failed to build or use spatial index for features ({e}). Might be slow."); features_gdf_sindex = None

    print("Calculating distances for each trail..."); start_time_dist = time.time()
    min_distances = {}
    feature_geoms = features_gdf.geometry
    MAX_SEARCH_RADIUS = 10000 # 10 km

    for index, trail in trails_gdf.iterrows():
        trail_id = trail[trail_id_col]
        trail_geom = trail.geometry
        min_dist = np.inf

        if not trail_geom.is_valid: min_distances[trail_id] = CALCULATION_ERROR_VALUE; continue
        if trail_geom.is_empty: min_distances[trail_id] = CALCULATION_ERROR_VALUE; continue

        try:
            possible_matches_index = None # Initialize as None
            if features_gdf_sindex:
                search_bounds = trail_geom.buffer(MAX_SEARCH_RADIUS).bounds
                hits = list(features_gdf_sindex.intersection(search_bounds))
                if hits:
                    possible_matches_index = features_gdf.iloc[hits].index # Assign the Index object

            # *** CHANGE: Explicitly check if the index is None or empty ***
            if possible_matches_index is None or possible_matches_index.empty:
                # This block handles two cases:
                # 1. features_gdf_sindex was None (no index) -> calculate against all
                # 2. features_gdf_sindex existed, but no hits were found -> min_dist is NO_SOURCE_FOUND_VALUE
                if not features_gdf_sindex:
                     # No index case: calculate against all features
                     print(f"Warning: No spatial index, calculating distance to all {len(feature_geoms)} features for trail {trail_id} (slow).")
                     distances_to_all = trail_geom.distance(feature_geoms)
                     min_dist = NO_SOURCE_FOUND_VALUE if distances_to_all.empty else distances_to_all.min()
                else:
                     # Index existed but no hits found within radius
                     # print(f"Trail {trail_id}: No features found within {MAX_SEARCH_RADIUS}m radius using spatial index.") # Optional verbose log
                     min_dist = NO_SOURCE_FOUND_VALUE
            else:
                # Index exists and found candidates (possible_matches_index is a non-empty Index)
                nearby_features = feature_geoms.loc[possible_matches_index]
                distances_to_nearby = trail_geom.distance(nearby_features)
                min_dist = NO_SOURCE_FOUND_VALUE if distances_to_nearby.empty else distances_to_nearby.min()

            # Final distance assignment
            if np.isinf(min_dist): min_distances[trail_id] = NO_SOURCE_FOUND_VALUE
            elif min_dist == NO_SOURCE_FOUND_VALUE: min_distances[trail_id] = NO_SOURCE_FOUND_VALUE
            else: min_distances[trail_id] = round(min_dist, 1)

        except Exception as e: print(f"Error calculating distance for trail {trail_id}: {e}"); min_distances[trail_id] = CALCULATION_ERROR_VALUE

    elapsed_dist = time.time() - start_time_dist; print(f"Finished calculating distances for {len(min_distances)} trails in {elapsed_dist:.2f} seconds.")
    min_dist_series = pd.Series(min_distances, name="min_distance"); min_dist_series = min_dist_series.reindex(trails_gdf[trail_id_col]); min_dist_series.fillna(CALCULATION_ERROR_VALUE, inplace=True)
    return min_dist_series


def filter_sources_by_category(all_sources_gdf):
    # (Same as v2.3 - unchanged)
    print("\n--- Filtering Sources into Categories ---")
    categories = {}
    if not isinstance(all_sources_gdf, gpd.GeoDataFrame): print("Error: Input to filter_sources_by_category is not a GeoDataFrame."); return categories # Return empty dict
    # Filter logic remains the same...
    mask_reliable = pd.Series(False, index=all_sources_gdf.index); #... (rest of filtering logic) ...
    if 'amenity' in all_sources_gdf.columns: mask_reliable |= all_sources_gdf['amenity'].fillna('').isin(RELIABLE_PUBLIC_WATER_TAGS.get('amenity', []))
    cond_keys = {'amenity': RELIABLE_PUBLIC_WATER_TAGS.get('amenity_conditional', []), 'man_made': RELIABLE_PUBLIC_WATER_TAGS.get('man_made_conditional', [])}
    has_drinking_water_yes = pd.Series(False, index=all_sources_gdf.index)
    if CONDITIONAL_TAG in all_sources_gdf.columns: has_drinking_water_yes = all_sources_gdf[CONDITIONAL_TAG].astype(str).str.lower().fillna('no') == 'yes'
    for key, values in cond_keys.items():
        if key in all_sources_gdf.columns and values: mask_key_in_values = all_sources_gdf[key].fillna('').isin(values); mask_reliable |= (mask_key_in_values & has_drinking_water_yes)
    categories[DIST_RELIABLE_PUBLIC_WATER_COL] = all_sources_gdf[mask_reliable].copy(); print(f"  {len(categories[DIST_RELIABLE_PUBLIC_WATER_COL]):>5} Reliable Public Water sources"); reliable_indices = categories[DIST_RELIABLE_PUBLIC_WATER_COL].index
    mask_potential_tags = pd.Series(False, index=all_sources_gdf.index)
    for key, values in POTENTIAL_PUBLIC_WATER_TAGS.items():
        if key in all_sources_gdf.columns: mask_potential_tags |= all_sources_gdf[key].fillna('').isin(values)
    mask_potential_final = mask_potential_tags & ~all_sources_gdf.index.isin(reliable_indices); categories[DIST_POTENTIAL_PUBLIC_WATER_COL] = all_sources_gdf[mask_potential_final].copy(); print(f"  {len(categories[DIST_POTENTIAL_PUBLIC_WATER_COL]):>5} Potential Public Water sources (excluding reliable)")
    mask_natural = pd.Series(False, index=all_sources_gdf.index)
    for key, values in NATURAL_WATER_TAGS.items():
        if key in all_sources_gdf.columns: mask_natural |= all_sources_gdf[key].fillna('').isin(values)
    categories[DIST_NATURAL_WATER_COL] = all_sources_gdf[mask_natural].copy(); print(f"  {len(categories[DIST_NATURAL_WATER_COL]):>5} Natural Water sources (Treatment required!)")
    mask_shop = pd.Series(False, index=all_sources_gdf.index)
    for key, values in SHOP_TAGS.items():
        if key in all_sources_gdf.columns: mask_shop |= all_sources_gdf[key].fillna('').isin(values)
    categories[DIST_SHOP_COL] = all_sources_gdf[mask_shop].copy(); print(f"  {len(categories[DIST_SHOP_COL]):>5} Shop sources")
    mask_amenity = pd.Series(False, index=all_sources_gdf.index)
    for key, values in AMENITY_TAGS.items():
        if key in all_sources_gdf.columns: mask_amenity |= all_sources_gdf[key].fillna('').isin(values)
    categories[DIST_AMENITY_COL] = all_sources_gdf[mask_amenity].copy(); print(f"  {len(categories[DIST_AMENITY_COL]):>5} Amenity sources")
    mask_tourism = pd.Series(False, index=all_sources_gdf.index)
    for key, values in TOURISM_TAGS.items():
        if key in all_sources_gdf.columns: mask_tourism |= all_sources_gdf[key].fillna('').isin(values)
    categories[DIST_TOURISM_COL] = all_sources_gdf[mask_tourism].copy(); print(f"  {len(categories[DIST_TOURISM_COL]):>5} Tourism sources")
    return categories

# =============================================================================
# --- Main Script Logic ---
# =============================================================================

if __name__ == "__main__":
    print("--- Trail Distance to Features Finder (v2.9) ---") # Updated version
    overall_start_time = time.time()

    # --- 1. Read Input Trail Data & Get Trail IDs ---
    print(f"\n--- Step 1: Reading Input Trail Data ---")
    # (Input reading logic unchanged)
    print(f"Reading input trail data from: {INPUT_CSV_PATH}")
    try:
        input_trails_df = pd.read_csv(INPUT_CSV_PATH)
        required_cols = [ID_COLUMN, 'representative_latitude', 'representative_longitude']
        if not all(col in input_trails_df.columns for col in required_cols): print(f"Error: Input CSV must contain columns: {required_cols}"); sys.exit(1)
        input_trails_df[ID_COLUMN] = pd.to_numeric(input_trails_df[ID_COLUMN], errors='coerce'); input_trails_df.dropna(subset=[ID_COLUMN], inplace=True); input_trails_df[ID_COLUMN] = input_trails_df[ID_COLUMN].astype(int)
        for coord_col in ['representative_latitude', 'representative_longitude']: input_trails_df[coord_col] = pd.to_numeric(input_trails_df[coord_col], errors='coerce'); input_trails_df.dropna(subset=['representative_latitude', 'representative_longitude'], inplace=True)
        trail_osmids = input_trails_df[ID_COLUMN].unique().tolist()
        if not trail_osmids: print(f"Error: No valid trail OSM IDs found after cleaning {INPUT_CSV_PATH}"); sys.exit(1)
        print(f"Found {len(trail_osmids)} unique valid trail OSM IDs to process: {trail_osmids[:5]}...")
        min_lon, min_lat = input_trails_df['representative_longitude'].min(), input_trails_df['representative_latitude'].min()
        max_lon, max_lat = input_trails_df['representative_longitude'].max(), input_trails_df['representative_latitude'].max()
        query_bbox_ll = (min_lon - BBOX_BUFFER_DEGREES, min_lat - BBOX_BUFFER_DEGREES, max_lon + BBOX_BUFFER_DEGREES, max_lat + BBOX_BUFFER_DEGREES)
        print(f"Bounding box for OSM query (WGS84): {query_bbox_ll}")
    except FileNotFoundError: print(f"Error: Input CSV file not found at {INPUT_CSV_PATH}"); sys.exit(1)
    except KeyError as e: print(f"Error: Required column {e} not found in {INPUT_CSV_PATH}"); sys.exit(1)
    except Exception as e: print(f"Error reading or processing input CSV: {e}"); sys.exit(1)

    # --- 2. Extract *All* Relevant Trail Geometries & Filter ---
    print(f"\n--- Step 2: Extracting Trail Geometries from OSM ---")
    output_dir = os.path.dirname(OUTPUT_CSV_PATH)
    temp_trails_geojson = os.path.join(output_dir, "temp_all_trails_in_bbox.geojson")

    # Use attributes=['id'] to force ID inclusion
    trails_gdf_raw_all = extract_osm_features(
        pbf_path=PBF_FILE_PATH,
        bbox_ll=query_bbox_ll,
        tags_filter={"highway": TRAIL_HIGHWAY_TAGS},
        element_type='w',
        output_geojson_path=temp_trails_geojson,
        description="all trails in bbox",
        add_id=True,      # Set to True to trigger ID finding logic inside function
        id_list=None,
        attributes=['id'] # Explicitly ask for the 'id' attribute
    )

    # --- Validate Trail Extraction & Filter by Input IDs ---
    if trails_gdf_raw_all is None: print(f"Error: Failed to extract trail geometries by tag. Check osmium errors."); sys.exit(1)
    if trails_gdf_raw_all.empty: print(f"Warning: No ways with tags {TRAIL_HIGHWAY_TAGS} found in the bbox.")
    elif ID_COLUMN not in trails_gdf_raw_all.columns:
        print(f"Error: Expected ID column '{ID_COLUMN}' not found after reading/parsing trail geometries.")
        print(f"       Available columns: {trails_gdf_raw_all.columns.tolist()}")
        print(f"       Check if --attributes=id worked correctly and if '@id' or 'id' is in the columns list above.")
        sys.exit(1)
    else:
        print(f"Extracted {len(trails_gdf_raw_all)} trail segments in bbox. Filtering for the {len(trail_osmids)} IDs from input CSV...")
        trails_gdf_raw = trails_gdf_raw_all[trails_gdf_raw_all[ID_COLUMN].isin(trail_osmids)].copy()
        print(f"Matched {len(trails_gdf_raw)} trail segments corresponding to input IDs.")
        found_ids = trails_gdf_raw[ID_COLUMN].unique().tolist()
        missing_ids = list(set(trail_osmids) - set(found_ids))
        if missing_ids: print(f"Warning: Could not find geometries for {len(missing_ids)} trail IDs within the extracted ways: {missing_ids}")
        if trails_gdf_raw.empty: print("Warning: No matching trail geometries found after filtering by ID.")

    # Reproject the *filtered* trails
    if 'trails_gdf_raw' in locals() and not trails_gdf_raw.empty:
        trails_gdf_proj = standardize_and_reproject(trails_gdf_raw, TARGET_CRS, source_crs=SOURCE_CRS, id_column_for_warnings=ID_COLUMN)
    else:
        print("No matching trail geometries found or extracted, cannot reproject trails.")
        trails_gdf_proj = None

    if trails_gdf_proj is None and 'trails_gdf_raw' in locals() and not trails_gdf_raw.empty : print("Error: Trail geometry reprojection failed.");

    try:
        if os.path.exists(temp_trails_geojson): os.remove(temp_trails_geojson)
    except OSError as e: print(f"Warning: Could not remove temp trails geojson: {e}")


    # --- Step 3, 4 (Source Extraction & Processing): Unchanged ---
    print(f"\n--- Step 3: Extracting ALL Potential Source Features from OSM ---")
    temp_sources_geojson = os.path.join(output_dir, "temp_all_sources.geojson")
    all_sources_gdf_raw = extract_osm_features( pbf_path=PBF_FILE_PATH, bbox_ll=query_bbox_ll, tags_filter=combined_tags_for_extraction, element_type='nwr', output_geojson_path=temp_sources_geojson, description="all potential source features", add_id=False, attributes=None)
    print(f"\n--- Step 4: Processing Source Features ---")
    categorized_sources_proj = {}
    if all_sources_gdf_raw is None: print("Error: Failed to extract source features.");
    elif all_sources_gdf_raw.empty: print("Warning: No potential source features extracted.");
    else:
        all_sources_gdf_proj = standardize_and_reproject(all_sources_gdf_raw, TARGET_CRS, source_crs=SOURCE_CRS)
        if all_sources_gdf_proj is None: print("Warning: Source feature reprojection failed.") ;
        elif all_sources_gdf_proj.empty: print("Warning: All source features invalid/empty after standardization.");
        else: categorized_sources_proj = filter_sources_by_category(all_sources_gdf_proj)
    for col_name in DISTANCE_COLUMNS: # Ensure keys exist
        if col_name not in categorized_sources_proj: categorized_sources_proj[col_name] = gpd.GeoDataFrame(columns=['geometry'], crs=TARGET_CRS)
    try:
        if os.path.exists(temp_sources_geojson): os.remove(temp_sources_geojson)
    except OSError as e: print(f"Warning: Could not remove temp sources geojson: {e}")

    # --- Step 5 (Distance Calculation): Logic updated to handle potentially invalid trails_gdf_proj ---
    print(f"\n--- Step 5: Calculating Distances from Trails to Sources by Category ---")
    results_df = input_trails_df.copy()
    if trails_gdf_proj is None or trails_gdf_proj.empty:
         print("Error/Warning: Projected trail data is missing or empty after standardization. Setting all distances to Calculation Error.")
         for col in DISTANCE_COLUMNS: results_df[col] = CALCULATION_ERROR_VALUE
    else:
        if ID_COLUMN not in trails_gdf_proj.columns:
             print(f"Critical Error: ID column '{ID_COLUMN}' missing from projected trails GDF before distance calculation. Aborting.")
             for col in DISTANCE_COLUMNS: results_df[col] = CALCULATION_ERROR_VALUE
        else:
            for category_col, sources_gdf in categorized_sources_proj.items():
                print(f"\nCalculating distances for category: {category_col}...")
                if not isinstance(sources_gdf, gpd.GeoDataFrame):
                    print(f"  Error: Sources data for {category_col} is not a GeoDataFrame.")
                    min_dist_series = pd.Series(CALCULATION_ERROR_VALUE, index=trails_gdf_proj[ID_COLUMN])
                elif sources_gdf.empty:
                    print(f"  No valid sources found for this category.")
                    min_dist_series = pd.Series(NO_SOURCE_FOUND_VALUE, index=trails_gdf_proj[ID_COLUMN])
                else:
                    min_dist_series = calculate_min_distance_to_features(trails_gdf_proj, sources_gdf, ID_COLUMN)

                # Merge results
                min_dist_series.name = category_col
                min_dist_df = min_dist_series.reset_index()
                if min_dist_df.columns[0] != ID_COLUMN: min_dist_df.rename(columns={min_dist_df.columns[0]: ID_COLUMN}, inplace=True)
                results_df = pd.merge(results_df, min_dist_df, on=ID_COLUMN, how='left')
                # *** CHANGE fillna location & method ***
                # results_df[category_col].fillna(CALCULATION_ERROR_VALUE, inplace=True) # Remove inplace=True
                results_df[category_col] = results_df[category_col].fillna(CALCULATION_ERROR_VALUE)
                print(f"  Finished distance calculation and merge for {category_col}.")

    # --- Step 6 (Save): Unchanged ---
    print(f"\n--- Step 6: Saving Enriched Data ---")
    print(f"Saving enriched data to: {OUTPUT_CSV_PATH}")
    try: # (Saving logic remains the same)
        for col in DISTANCE_COLUMNS:
            if col not in results_df.columns: print(f"Warning: Distance column {col} was missing."); results_df[col] = CALCULATION_ERROR_VALUE
        other_tags_col = 'other_osm_tags'
        if other_tags_col in results_df.columns and not pd.api.types.is_string_dtype(results_df[other_tags_col]):
             print(f"Converting '{other_tags_col}' column to JSON string format.")
             def safe_json_dumps(x):
                 if pd.isna(x): return ''
                 try:
                     if isinstance(x, str):
                          try: json.loads(x); return x
                          except json.JSONDecodeError: pass
                     return json.dumps(x)
                 except (TypeError, OverflowError) as e: print(f"Warning: Could not JSON dump value '{x}'. Converting to string. Error: {e}"); return json.dumps(str(x))
             results_df[other_tags_col] = results_df[other_tags_col].apply(safe_json_dumps)
        float_format_spec = '%.1f'
        results_df.to_csv(OUTPUT_CSV_PATH, index=False, float_format=float_format_spec)
        print("Enriched data saved successfully.")
    except Exception as e: print(f"Error saving output CSV: {e}")

    overall_elapsed = time.time() - overall_start_time
    print(f"\n--- Script Finished in {overall_elapsed:.2f} seconds ---")

