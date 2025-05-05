#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone script to fetch Strava running segments for defined areas,
save details, visualize segments per area, generate a comparison summary,
make individual DataFrames available, and create a final overview map.

Version 2.3 (Overview Map):
- Handles multiple areas defined in `AREAS_TO_PROCESS`.
- Runs geocoding, fetching, processing, plotting, saving for each area.
- Generates separate output directories and files per area.
- Saves detailed segment info to CSV per area.
- Creates plots per area: segments only and segments on basemap.
- Calculates mean values for numeric attributes and saves a comparison CSV.
- Assigns area DataFrames to global variables (e.g., df_A).
- **NEW:** Creates and saves an overview map showing the locations of all processed areas.
- Uses hardcoded API credentials.

Requires: stravalib, rasterio, numpy, geopy, math, os, sys, shapely, pandas, matplotlib, contextily, geopandas, scipy
"""

import os
import sys
import rasterio
import rasterio.warp
import numpy as np
import pandas as pd
from shapely.geometry import box, Point, LineString
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import math
import warnings
import time
from datetime import datetime, timedelta
import traceback

# --- Attempt to import required libraries ---
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle
except ImportError: print("ERROR: Missing 'matplotlib'. Install: pip install matplotlib"); sys.exit(1)
try:
    import contextily as cx
except ImportError: print("ERROR: Missing 'contextily'. Install: pip install contextily"); sys.exit(1)
try:
    import geopandas as gpd
except ImportError: print("ERROR: Missing 'geopandas'. Install: pip install geopandas"); sys.exit(1)
try:
    from stravalib.client import Client
    from stravalib.exc import RateLimitExceeded, AccessUnauthorized, RateLimitTimeout
    from stravalib.model import LatLon
    try: from stravalib.exc import NotFound
    except ImportError:
        try: from stravalib.exc import ObjectNotFoundError as NotFound
        except ImportError: NotFound = None; print("WARNING: Could not import Strava NotFound exception.")
except ImportError: print("ERROR: Missing 'stravalib'. Install: pip install stravalib"); sys.exit(1)
try:
    from scipy import stats
except ImportError: print("ERROR: Missing 'scipy'. Install: pip install scipy"); sys.exit(1)

# Optional import for better label placement on overview map
try:
    from adjustText import adjust_text
    ADJUST_TEXT_AVAILABLE = True
except ImportError:
    ADJUST_TEXT_AVAILABLE = False
    print("NOTE: 'adjustText' library not found (pip install adjustText). Labels on overview map might overlap.")


# =============================================================================
# --- Configuration ---
# =============================================================================

# --- 1. Area of Interest Definitions ---
# Define parameters for the areas you want to compare in a list of dictionaries
AREAS_TO_PROCESS = [
    
    # {
    #     "place_name": "Alford, Lincolnshire, UK",   # <- Define Place Name for Area B
    #     "radius_km": 5,                 # <- Define Radius (km) for Area B
    #     "label": "A"                    # Label used in filenames/titles
    # },
    # {
    #      "place_name": "Laceby, Lincolnshire, UK",
    #      "radius_km": 5,
    #      "label": "B"
    # },
    # {
    #      "place_name": "Grimsby, Lincolnshire, UK",
    #      "radius_km": 5,
    #      "label": "C"
    # },
    # {
    #      "place_name": "Claxby, Lincolnshire. UK",
    #      "radius_km": 5,
    #      "label": "D"
    # },
    
    #=========
    
    {
        "place_name": "Ecclesall, Sheffield, UK",   # <- Define Place Name for Area A
        "radius_km": 5,                 # <- Define Radius (km) for Area A
        "label": "A"                    # Label used in filenames/titles
    },
    {
         "place_name": "Hathersage, UK",
         "radius_km": 5,
         "label": "B"
    },
    {
         "place_name": "Edale, UK",
         "radius_km": 5,
         "label": "C"
    },
    {
         "place_name": "Totley, UK",
         "radius_km": 5,
         "label": "D"
    },
    {
         "place_name": "Nether Green, Sheffield, UK",
         "radius_km": 5,
         "label": "E"
    },
    {
         "place_name": "Upper Padley, UK",
         "radius_km": 5,
         "label": "F"
    },
    {
         "place_name": "Hayfield, UK",
         "radius_km": 5,
         "label": "G"
    },
    
    {
         "place_name": "Bamford, UK", 
         "radius_km": 5,
         "label": "H"
    },
    {
         "place_name": "Ashopton, UK", 
         "radius_km": 5,
         "label": "I"
    },
    {
         "place_name": "Upper Booth, UK", 
         "radius_km": 5,
         "label": "J"
    },
    {
         "place_name": "Derwent Dam, UK", 
         "radius_km": 5,
         "label": "K"
    },
    
    {
         "place_name": "High Neb, UK", 
         "radius_km": 5,
         "label": "L"
    },
    
    {
         "place_name": "Oughtibridge, UK", # North West Sheffield - Forests
         "radius_km": 5,
         "label": "L"
    },
    
    
    # {
    #      "place_name": "Bakewll, UK", - TOO FLAT
    #      "radius_km": 5,
    #      "label": "G"
    # },
    # {
    #      "place_name": "Sherwood Forest, UK", - TOO FLAT
    #      "radius_km": 5,
    #      "label": "H"
    # },

    #=========

    # {
    #      "place_name": "Wendover, UK", # Near Aylesbury, Oxford, Cambridge
    #      "radius_km": 5,
    #      "label": "K"
    # },
    
    # {
    #      "place_name": "Berkhamsted, UK", # Near Aylesbury, Oxford, Cambridge
    #      "radius_km": 5,
    #      "label": "L"
    # },

    #=========
    
    # {
    #      "place_name": "Keswick, UK",
    #      "radius_km": 10,
    #      "label": "M"
    # },

    # {
    #      "place_name": "Sedbergh, UK",
    #      "radius_km": 5,
    #      "label": "N"
    # },
    # {
    #      "place_name": "Great Gable, Keswick, UK",
    #      "radius_km": 5,
    #      "label": "O"
    # },
    # {
    #      "place_name": "Ambleside, UK",
    #      "radius_km": 5,
    #      "label": "P"
    # },
    
    # # {
    # #      "place_name": "Kendal, UK", - TOO FLAT - BUT VERY POPULAR
    # #      "radius_km": 5,
    # #      "label": "J"
    # # },
        
    #=========
    
    
    # Add more dictionaries here to compare more areas, e.g.:
    # {
    #     "place_name": "Hathersage, UK",
    #     "radius_km": 3,
    #     "label": "C"
    # },
    
    # Add more dictionaries here to compare more areas, e.g.:
    # {
    #     "place_name": "Hathersage, UK",
    #     "radius_km": 3,
    #     "label": "C"
    # },
]
    
    
# --- 2. Strava API Configuration ---
STRAVA_CLIENT_ID = 157447
STRAVA_CLIENT_SECRET = "25a9e9a6af5a64653a17f56e60dd87540f1d537f"
STRAVA_REFRESH_TOKEN = "333ff0884e3eade9485faf903006db6c36fbc735"
STRAVA_ACCESS_TOKEN_FILE = "strava_access_token.txt"

# --- 3. Output Configuration ---
OUTPUT_BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
TARGET_CRS = "EPSG:27700" # For plotting
SOURCE_CRS_GPS = "EPSG:4326" # For geocoding/API
COMPARISON_CSV_FILENAME = "comparison_summary.csv"
OVERVIEW_MAP_FILENAME = "overview_map_areas.png" # Filename for the new map

# --- 4. Visualization Configuration ---
VIS_CMAP_SEGMENTS = 'plasma'
VIS_LINEWIDTH = 1.5
VIS_SEGMENT_ALPHA = 0.8
VIS_LABEL_FONTSIZE = 6
VIS_LABEL_COLOR = 'black'
VIS_LABEL_BACKGROUND = 'white'
VIS_LABEL_ALPHA = 0.6
CTX_PROVIDER = cx.providers.OpenStreetMap.Mapnik
CTX_ALPHA_BASEMAP = 1.0
BBOX_COLOR = 'red'
BBOX_LINEWIDTH = 1.5
# Overview Map Specifics
OVERVIEW_MARKER_COLOR = 'red'
OVERVIEW_MARKER_SIZE = 50
OVERVIEW_LABEL_FONTSIZE = 8
OVERVIEW_LABEL_COLOR = 'black'

# =============================================================================
# --- END OF Configuration ---
# =============================================================================

print(f"--- Strava Segment Plotter v2.3 (Overview Map) ---") # Updated version
overall_start_time = time.time()

# =============================================================================
# --- Helper Functions (Unchanged from v2.2) ---
# =============================================================================

def get_bounding_box_wgs84(latitude, longitude, radius_km):
    """Calculates WGS84 bounding box."""
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180): raise ValueError("Invalid lat/lon.")
    if radius_km <= 0: raise ValueError("Radius must be positive.")
    earth_radius_km = 6371.0088
    lat_delta_rad = radius_km / earth_radius_km
    lat_delta_deg = math.degrees(lat_delta_rad)
    clamped_latitude_rad = math.radians(max(-89.999999, min(89.999999, latitude)))
    cos_latitude = math.cos(clamped_latitude_rad)
    if abs(cos_latitude) < 1e-12: lon_delta_deg = 180.0
    else: lon_delta_rad = radius_km / (earth_radius_km * cos_latitude); lon_delta_deg = math.degrees(lon_delta_rad)
    min_lat = max(-90.0, latitude - lat_delta_deg); max_lat = min(90.0, latitude + lat_delta_deg)
    min_lon = longitude - lon_delta_deg; max_lon = longitude + lon_delta_deg
    min_lon = (min_lon + 180) % 360 - 180; max_lon = (max_lon + 180) % 360 - 180
    return (min_lon, min_lat, max_lon, max_lat)

def get_strava_client(base_dir):
    """Authenticates with Strava API using hardcoded credentials."""
    client = Client()
    access_token = None
    token_file_path = os.path.join(base_dir, STRAVA_ACCESS_TOKEN_FILE)
    if os.path.exists(token_file_path):
        try:
            with open(token_file_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                if len(lines) >= 2:
                    token, expires_at_str = lines[0], lines[1]
                    try:
                        expires_at_dt = datetime.fromisoformat(expires_at_str)
                        if datetime.now() < (expires_at_dt - timedelta(hours=1)):
                            access_token = token; client.access_token = access_token
                            print(f"  Using cached Strava access token (expires {expires_at_dt}).")
                        else: print("  Cached token expired/near expiry.")
                    except ValueError: print(f"  Warning: Invalid expiry date format in {token_file_path}.")
                else: print(f"  Warning: Invalid format in {token_file_path}.")
        except Exception as read_err: print(f"  Warning: Could not read token cache ({token_file_path}): {read_err}")
    if not access_token:
        if not all([STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, STRAVA_REFRESH_TOKEN]):
            print("  ERROR: Hardcoded Strava credentials missing."); return None
        print("  Attempting to refresh Strava access token...")
        try:
            client_id_int = int(STRAVA_CLIENT_ID)
            token_response = client.refresh_access_token(
                client_id=client_id_int, client_secret=STRAVA_CLIENT_SECRET, refresh_token=STRAVA_REFRESH_TOKEN)
            access_token = token_response['access_token']
            new_refresh_token = token_response['refresh_token']
            expires_at_dt = datetime.fromtimestamp(token_response['expires_at'])
            client.access_token = access_token
            print(f"  Successfully refreshed Strava token (expires {expires_at_dt}).")
            try:
                with open(token_file_path, 'w') as f: f.write(f"{access_token}\n{expires_at_dt.isoformat()}\n")
                print(f"  Saved new access token to cache file: {token_file_path}")
            except Exception as write_err: print(f"  Warning: Could not save token cache ({token_file_path}): {write_err}")
            if new_refresh_token != STRAVA_REFRESH_TOKEN:
                print("\n  *** IMPORTANT: Strava Refresh Token updated! Update script: ***")
                print(f"  *** NEW TOKEN: {new_refresh_token} ***\n")
        except AccessUnauthorized as e: print(f"  ERROR: Strava Auth Error refreshing token: {e}. Check REFRESH_TOKEN."); return None
        except RateLimitExceeded as e: print(f"  ERROR: Strava Rate Limit Exceeded refreshing token: {e}."); return None
        except RateLimitTimeout as e: print(f"  ERROR: Strava Rate Limit Timeout refreshing token: {e}."); return None
        except Exception as e: print(f"  ERROR refreshing token: {type(e).__name__} - {e}"); traceback.print_exc(); return None
    if not client.access_token: print(" ERROR: Failed to get Strava access token."); return None
    return client

def fetch_strava_segments_details(client, bounds_wgs84, area_label):
    """Fetches segment summaries and then details for a given area."""
    if client is None: print(f"  ERROR (Area {area_label}): Strava client not authenticated."); return []
    west, south, east, north = bounds_wgs84[0], bounds_wgs84[1], bounds_wgs84[2], bounds_wgs84[3]
    print(f"  Exploring Strava segments (Area {area_label}) for bounds: SW=({south:.5f},{west:.5f}), NE=({north:.5f},{east:.5f})")
    segment_summaries = []
    try:
        explore_results = client.explore_segments(bounds=(south, west, north, east), activity_type='running')
        segment_summaries = list(explore_results)
        print(f"  Found {len(segment_summaries)} segment summaries (Area {area_label}).")
        if len(segment_summaries) == 10: print("  WARNING (Area {area_label}): Retrieved exactly 10 segments (API limit?).")
    except RateLimitExceeded as e: print(f"  ERROR (Area {area_label}): Rate Limit during exploration: {e}"); return []
    except AccessUnauthorized as e: print(f"  ERROR (Area {area_label}): Auth Error during exploration: {e}"); return []
    except Exception as e: print(f"  ERROR (Area {area_label}) during exploration: {type(e).__name__} - {e}"); traceback.print_exc(); return []

    detailed_segments = []
    if not segment_summaries: print(f"  No segment summaries found (Area {area_label})."); return []
    total_to_fetch = len(segment_summaries)
    print(f"  Fetching full details for {total_to_fetch} segments (Area {area_label})...")
    fetch_errors, rate_limit_hit, nfe, authe, othe = 0, False, 0, 0, 0
    delay = 0.7 # Delay between get_segment calls
    for i, summary in enumerate(segment_summaries):
        segment_id = getattr(summary, 'id', 'UNKNOWN_ID')
        time.sleep(delay)
        print(f"    Area {area_label}: Fetching {i+1}/{total_to_fetch} (ID: {segment_id})...", end='\r', flush=True)
        try:
            detailed = client.get_segment(segment_id=segment_id)
            if detailed: detailed_segments.append(detailed)
            else: print(f"\n    Warning (Area {area_label}): get_segment returned None for {segment_id}."); fetch_errors+=1; othe+=1
        except NotFound: fetch_errors+=1; nfe+=1; continue
        except RateLimitExceeded as e: print(f"\n    ERROR (Area {area_label}): Rate Limit fetching {segment_id} ({e}). Aborting."); fetch_errors+=(total_to_fetch-i); rate_limit_hit=True; break
        except AccessUnauthorized as e: print(f"\n    ERROR (Area {area_label}): Auth Error fetching {segment_id}: {e}. Skipping."); fetch_errors+=1; authe+=1; continue
        except Exception as e:
            sc = getattr(getattr(e, 'response', None), 'status_code', None)
            if sc == 404: fetch_errors+=1; nfe+=1; continue
            else: print(f"\n    ERROR (Area {area_label}) fetching {segment_id}: {type(e).__name__} - {e}"); fetch_errors+=1; othe+=1
    print() # Newline after progress
    print(f"  Successfully fetched details for {len(detailed_segments)} segments (Area {area_label}).")
    if fetch_errors > 0:
        ed = []
        if nfe: ed.append(f"{nfe} Not Found")
        if authe: ed.append(f"{authe} Auth Errors")
        if othe: ed.append(f"{othe} Other Errors")
        if rate_limit_hit: ed.append("Rate Limit Hit")
        print(f"  Could not fetch {fetch_errors} segment details (Area {area_label}) ({', '.join(ed)}).")
    return detailed_segments

def plot_segments_styled(segments_gdf, target_bounds, output_png_path, title, value_col='effort_count'):
    """Plots ONLY styled segments: color by effort percentile, constant width, labeled."""
    print(f"  Generating Segment-Only Plot: {title} -> {os.path.basename(output_png_path)}")
    if segments_gdf is None or segments_gdf.empty: print(f"    Skipping plot: No data."); return
    if value_col not in segments_gdf.columns:
        print(f"    WARNING: Styling column '{value_col}' missing. Plotting grey.")
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            minx, miny, maxx, maxy = target_bounds
            ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
            segments_gdf.plot(ax=ax, color='grey', linewidth=VIS_LINEWIDTH, alpha=VIS_SEGMENT_ALPHA)
            ax.set_title(f"{title}\n(Styling column '{value_col}' missing)", fontsize=12)
            ax.set_aspect('equal', adjustable='box'); ax.set_axis_off()
            plt.tight_layout(); plt.savefig(output_png_path, dpi=200, bbox_inches='tight'); plt.close(fig)
            print(f"    Segment-only plot saved (uncolored): {output_png_path}")
        except Exception as plot_err: print(f"    Error plotting uncolored: {plot_err}"); traceback.print_exc()
        return
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        minx, miny, maxx, maxy = target_bounds
        ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
        ax.set_aspect('equal', adjustable='box'); ax.set_axis_off()
        values = segments_gdf[value_col].fillna(0).astype(float)
        valid_values = values[values > 0]
        if not valid_values.empty and len(valid_values.unique()) > 1:
            percentiles = values.apply(lambda x: stats.percentileofscore(valid_values, x, kind='weak') / 100.0 if x > 0 else 0.0)
        elif not valid_values.empty: percentiles = values.apply(lambda x: 0.5 if x > 0 else 0.0)
        else: percentiles = pd.Series([0.0] * len(segments_gdf), index=segments_gdf.index)
        cmap = plt.get_cmap(VIS_CMAP_SEGMENTS); colors = cmap(percentiles)
        segments_gdf.plot(ax=ax, color=colors, linewidth=VIS_LINEWIDTH, alpha=VIS_SEGMENT_ALPHA, zorder=10)
        label_bg_props = None
        if VIS_LABEL_BACKGROUND and VIS_LABEL_BACKGROUND.lower() != 'none':
             label_bg_props = dict(boxstyle='round,pad=0.15', fc=VIS_LABEL_BACKGROUND, ec='none', alpha=VIS_LABEL_ALPHA)
        plot_box = box(minx, miny, maxx, maxy)
        for idx, row in segments_gdf.iterrows():
            if row.geometry and row.geometry.is_valid and not row.geometry.is_empty:
                 centroid = row.geometry.centroid
                 if centroid.within(plot_box):
                     ax.text(centroid.x, centroid.y, str(idx), fontsize=VIS_LABEL_FONTSIZE, color=VIS_LABEL_COLOR, ha='center', va='center', bbox=label_bg_props, zorder=20)
        norm = plt.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=15, pad=0.02)
        cbar_label = f'{value_col.replace("_"," ").title()} (Percentile Rank)'
        cbar.set_label(cbar_label, size=10); cbar.ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=14, pad=10)
        plt.tight_layout(rect=[0, 0, 0.92, 1])
        plt.savefig(output_png_path, dpi=200, bbox_inches='tight'); plt.close(fig)
        print(f"    Segment-only plot saved successfully: {output_png_path}")
    except Exception as e:
        print(f"    ERROR during segment-only plot: {type(e).__name__} - {e}"); traceback.print_exc()
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)

def plot_segments_styled_with_basemap(segments_gdf, target_bounds, target_crs_str, output_png_path, title, value_col='effort_count'):
    """Plots styled segments overlaid on a contextily basemap, with AOI box."""
    print(f"  Generating Segment + Basemap Plot: {title} -> {os.path.basename(output_png_path)}")
    if segments_gdf is None or segments_gdf.empty: print(f"    Skipping plot: No data."); return
    if segments_gdf.crs is None: print(f"    ERROR: GDF missing CRS. Cannot add basemap."); return
    if value_col not in segments_gdf.columns:
        print(f"    WARNING: Styling column '{value_col}' missing. Plotting grey on basemap.")
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            minx, miny, maxx, maxy = target_bounds
            ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
            segments_gdf.plot(ax=ax, color='grey', linewidth=VIS_LINEWIDTH, alpha=VIS_SEGMENT_ALPHA, zorder=10)
            rect = Rectangle((minx, miny), maxx - minx, maxy - miny, linewidth=BBOX_LINEWIDTH, edgecolor=BBOX_COLOR, facecolor='none', zorder=20)
            ax.add_patch(rect)
            try: cx.add_basemap(ax, crs=segments_gdf.crs, source=CTX_PROVIDER, alpha=CTX_ALPHA_BASEMAP, zorder=1)
            except Exception as ctx_e: print(f"      ERROR adding basemap: {ctx_e}")
            ax.set_title(f"{title}\n(Styling column '{value_col}' missing)", fontsize=12)
            ax.set_aspect('equal', adjustable='box'); ax.set_axis_off(); plt.tight_layout()
            plt.savefig(output_png_path, dpi=200, bbox_inches='tight'); plt.close(fig)
            print(f"    Segments + Basemap plot saved (uncolored): {output_png_path}")
        except Exception as plot_err: print(f"    Error plotting uncolored on basemap: {plot_err}"); traceback.print_exc()
        return
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        minx, miny, maxx, maxy = target_bounds
        ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
        ax.set_aspect('equal', adjustable='box'); ax.set_axis_off()
        values = segments_gdf[value_col].fillna(0).astype(float)
        valid_values = values[values > 0]
        if not valid_values.empty and len(valid_values.unique()) > 1: percentiles = values.apply(lambda x: stats.percentileofscore(valid_values, x, kind='weak') / 100.0 if x > 0 else 0.0)
        elif not valid_values.empty: percentiles = values.apply(lambda x: 0.5 if x > 0 else 0.0)
        else: percentiles = pd.Series([0.0] * len(segments_gdf), index=segments_gdf.index)
        cmap = plt.get_cmap(VIS_CMAP_SEGMENTS); colors = cmap(percentiles)
        try: cx.add_basemap(ax, crs=segments_gdf.crs, source=CTX_PROVIDER, zoom='auto', alpha=CTX_ALPHA_BASEMAP, zorder=1)
        except Exception as ctx_e: print(f"    ERROR adding basemap: {type(ctx_e).__name__} - {ctx_e}")
        segments_gdf.plot(ax=ax, color=colors, linewidth=VIS_LINEWIDTH, alpha=VIS_SEGMENT_ALPHA, zorder=10)
        rect = Rectangle((minx, miny), maxx - minx, maxy - miny, linewidth=BBOX_LINEWIDTH, edgecolor=BBOX_COLOR, facecolor='none', zorder=20)
        ax.add_patch(rect)
        label_bg_props = None
        if VIS_LABEL_BACKGROUND and VIS_LABEL_BACKGROUND.lower() != 'none':
             label_bg_props = dict(boxstyle='round,pad=0.15', fc=VIS_LABEL_BACKGROUND, ec='none', alpha=VIS_LABEL_ALPHA)
        plot_box = box(minx, miny, maxx, maxy)
        for idx, row in segments_gdf.iterrows():
             if row.geometry and row.geometry.is_valid and not row.geometry.is_empty:
                centroid = row.geometry.centroid
                if centroid.within(plot_box):
                     ax.text(centroid.x, centroid.y, str(idx), fontsize=VIS_LABEL_FONTSIZE, color=VIS_LABEL_COLOR, ha='center', va='center', bbox=label_bg_props, zorder=25)
        norm = plt.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=15, pad=0.02)
        cbar_label = f'{value_col.replace("_"," ").title()} (Percentile Rank)'
        cbar.set_label(cbar_label, size=10); cbar.ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=14, pad=10)
        plt.tight_layout(rect=[0, 0, 0.92, 1])
        plt.savefig(output_png_path, dpi=200, bbox_inches='tight', transparent=False); plt.close(fig)
        print(f"    Segments + Basemap plot saved successfully: {output_png_path}")
    except Exception as e:
        print(f"    ERROR during segments + basemap plot: {type(e).__name__} - {e}"); traceback.print_exc()
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)

# =============================================================================
# --- Main Processing Function for One Area ---
# =============================================================================

def process_area(strava_client, area_config):
    """
    Runs the full workflow (geocode, fetch, process, save, plot) for a single area.

    Args:
        strava_client (stravalib.Client): Authenticated Strava client.
        area_config (dict): Dictionary containing 'place_name', 'radius_km', 'label'.

    Returns:
        tuple: (pandas_df, projected_gdf, center_coords_wgs84) for the processed area,
               or (None, None, None) on failure.
               center_coords_wgs84 is a tuple (latitude, longitude).
    """
    place_name = area_config['place_name']
    radius_km = area_config['radius_km']
    area_label = area_config['label']
    print(f"\n===== Processing Area {area_label}: {place_name} (Radius: {radius_km}km) =====")
    area_start_time = time.time()

    # --- Dynamic Path Generation for this Area ---
    try:
        place_slug = place_name.replace(' ', '_').replace(',', '').lower()
        place_slug = "".join(c for c in place_slug if c.isalnum() or c in ('_', '-')).rstrip('_')
        if not place_slug: place_slug = f"area_{area_label.lower()}"
        output_subdir_name = f"outputs_{place_slug}_radius{radius_km}km_area_{area_label}"
        output_dir = os.path.join(OUTPUT_BASE_DIR, output_subdir_name)
        print(f"Output directory for Area {area_label}: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.isdir(output_dir) or not os.access(output_dir, os.W_OK):
             raise OSError(f"Failed to create/access output directory: {output_dir}")
        print(f"Output directory verified.")
    except OSError as e: print(f"FATAL ERROR (Area {area_label}) setting up output directory: {e}"); return None, None, None
    except Exception as e: print(f"FATAL ERROR (Area {area_label}) during path setup: {type(e).__name__} - {e}"); return None, None, None

    # Define Output File Paths for this Area
    base_filename = f"{place_slug}_radius{radius_km}km_area_{area_label}"
    output_segments_csv_path = os.path.join(output_dir, f"{base_filename}_details.csv")
    vis_segments_only_path = os.path.join(output_dir, f"{base_filename}_plot_segments_only.png")
    vis_segments_with_basemap_path = os.path.join(output_dir, f"{base_filename}_plot_segments_map.png")

    # Initialize results for this area
    target_bounds_proj = None; segments_gdf_proj = None; segments_gdf = None; bounds_wgs84 = None
    segments_df = None; center_coords_wgs84 = None

    try:
        # --- Step 1: Define Area of Interest (AOI) ---
        print(f"\n--- Step 1 (Area {area_label}): Defining AOI ---")
        print(f"  Geocoding '{place_name}'...")
        geolocator = Nominatim(user_agent=f"strava_segment_plotter_v2.3_{area_label}_{time.time()}")
        center_lat, center_lon = None, None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                location = geolocator.geocode(place_name, timeout=25)
                if location:
                    center_lat, center_lon = location.latitude, location.longitude
                    center_coords_wgs84 = (center_lat, center_lon) # Store WGS84 coords
                    print(f"  Geocoding successful: Lat={center_lat:.5f}, Lon={center_lon:.5f}"); break
                else: print(f"  Geocoding attempt {attempt+1}/{max_retries} failed: Place '{place_name}' not found."); time.sleep(3)
            except (GeocoderTimedOut, GeocoderServiceError) as e: print(f"  Geocoding attempt {attempt+1}/{max_retries} failed: {type(e).__name__} - {e}"); time.sleep(5)
            except Exception as e: print(f"FATAL ERROR (Area {area_label}) during geocoding: {type(e).__name__} - {e}"); traceback.print_exc(); return None, None, None
            if attempt == max_retries - 1: print(f"FATAL ERROR (Area {area_label}): Could not geocode '{place_name}'."); return None, None, None
        if center_lat is None or center_lon is None: print(f"FATAL ERROR (Area {area_label}): Failed to get coords."); return None, None, None

        try: # Calculate & Transform Bounds
            bounds_wgs84 = get_bounding_box_wgs84(center_lat, center_lon, radius_km)
            print(f"  Calculated BBox (WGS84): {bounds_wgs84}")
            try: target_crs_obj = rasterio.crs.CRS.from_string(TARGET_CRS)
            except rasterio.errors.CRSError as e: print(f"FATAL ERROR: Invalid Target CRS '{TARGET_CRS}': {e}"); return None, None, None
            target_bounds_proj = rasterio.warp.transform_bounds(SOURCE_CRS_GPS, TARGET_CRS, *bounds_wgs84)
            print(f"  Transformed BBox ({TARGET_CRS}): {target_bounds_proj}")
        except ValueError as e: print(f"FATAL ERROR (Area {area_label}) calculating bbox: {e}"); return None, None, None
        except rasterio.errors.CRSError as e: print(f"FATAL ERROR (Area {area_label}) transforming CRS: {e}"); return None, None, None
        except Exception as e: print(f"FATAL ERROR (Area {area_label}) during AOI processing: {type(e).__name__} - {e}"); traceback.print_exc(); return None, None, None

        # --- Step 2: Fetch Strava Segments ---
        print(f"\n--- Step 2 (Area {area_label}): Fetching Strava Segments ---")
        segments = fetch_strava_segments_details(strava_client, bounds_wgs84, area_label)

        # --- Step 3: Process Segment Details & Create GeoDataFrame ---
        print(f"\n--- Step 3 (Area {area_label}): Processing Segment Details ---")
        segment_data_for_df = []
        if segments:
            print(f"  Processing {len(segments)} detailed segments...")
            processed_count, geometry_skipped_count = 0, 0
            for segment in segments: # Simplified processing loop
                start_latlon=getattr(segment,'start_latlng',None); end_latlon=getattr(segment,'end_latlng',None)
                start_lat=getattr(start_latlon,'lat',None) if isinstance(start_latlon,LatLon) else None
                start_lon=getattr(start_latlon,'lon',None) if isinstance(start_latlon,LatLon) else None
                end_lat=getattr(end_latlon,'lat',None) if isinstance(end_latlon,LatLon) else None
                end_lon=getattr(end_latlon,'lon',None) if isinstance(end_latlon,LatLon) else None
                geometry=None; coords=[start_lon,start_lat,end_lon,end_lat]
                if all(isinstance(c,(int,float)) for c in coords):
                    if (start_lon,start_lat)!=(end_lon,end_lat):
                        try: geometry=LineString([(start_lon,start_lat),(end_lon,end_lat)])
                        except Exception: geometry=None; geometry_skipped_count+=1
                    else: geometry=None; geometry_skipped_count+=1
                else: geometry=None; geometry_skipped_count+=1
                details={'segment_id':getattr(segment,'id',None),'name':getattr(segment,'name',None),
                           'activity_type':getattr(segment,'activity_type',None),'distance_m':float(getattr(segment,'distance',0.0)),
                           'average_grade_percent':getattr(segment,'average_grade',None),'maximum_grade_percent':getattr(segment,'maximum_grade',None),
                           'elevation_high_m':getattr(segment,'elevation_high',None),'elevation_low_m':getattr(segment,'elevation_low',None),
                           'total_elevation_gain_m':getattr(segment,'total_elevation_gain',None),'climb_category':getattr(segment,'climb_category',None),
                           'athlete_count':getattr(segment,'athlete_count',None),'effort_count':getattr(segment,'effort_count',None),
                           'star_count':getattr(segment,'star_count',None),'created_at':getattr(segment,'created_at',None),
                           'updated_at':getattr(segment,'updated_at',None),'map_polyline':getattr(getattr(segment,'map',None),'polyline',None),
                           'start_lat':start_lat,'start_lon':start_lon,'end_lat':end_lat,'end_lon':end_lon,'geometry':geometry}
                segment_data_for_df.append(details); processed_count+=1
            print(f"  Processed {processed_count} segments.");
            if geometry_skipped_count: print(f"  Skipped geometry for {geometry_skipped_count} segments.")
            segments_df = pd.DataFrame(segment_data_for_df); print(f"  Created Pandas DataFrame ({len(segments_df)} rows).")
            try: # Create GDF
                 segments_with_geom = segments_df[segments_df['geometry'].notna()].copy()
                 if not segments_with_geom.empty:
                     segments_gdf = gpd.GeoDataFrame(segments_with_geom, geometry='geometry', crs=SOURCE_CRS_GPS)
                     if len(segments_gdf[~segments_gdf.geometry.is_valid]): print(f"  Warning: Found invalid geometries.")
                     print(f"  Created GeoDataFrame ({len(segments_gdf)} valid geometries, CRS: {segments_gdf.crs}).")
                 else: print("  No valid geometries found."); segments_gdf = None
            except Exception as e: print(f"  ERROR Creating GDF: {e}"); traceback.print_exc(); segments_gdf = None
            if segments_gdf is not None and not segments_gdf.empty: # Post-process GDF
                if 'effort_count' in segments_gdf.columns:
                     nans = segments_gdf['effort_count'].isna().sum()
                     if nans: print(f"  Filling {nans} missing 'effort_count' values with 0.")
                     segments_gdf['effort_count'] = segments_gdf['effort_count'].fillna(0).astype(int)
                else: print("  WARNING: 'effort_count' missing. Adding zeros."); segments_gdf['effort_count'] = 0
                print(f"  Reprojecting {len(segments_gdf)} geometries to {TARGET_CRS}...")
                try: segments_gdf_proj = segments_gdf.to_crs(TARGET_CRS); print(f"  Reprojection complete (CRS: {segments_gdf_proj.crs}).")
                except Exception as e: print(f"  ERROR reprojection: {e}"); traceback.print_exc(); segments_gdf_proj = None
            # Save CSV (always save pandas DF if it exists)
            if segments_df is not None and not segments_df.empty:
                print(f"  Saving details to: {output_segments_csv_path}")
                try:
                    cols = [c for c in segments_df.columns if c != 'geometry']
                    df_save = segments_df[cols].reset_index().rename(columns={'index': 'original_index'})
                    df_save.to_csv(output_segments_csv_path, index=False, encoding='utf-8')
                    print("  Successfully saved CSV.")
                except Exception as e: print(f"  ERROR saving CSV: {e}"); traceback.print_exc()
            else: print("  No segment data to save to CSV.")
        else: print("  No segments fetched to process."); segments_df = None; segments_gdf = None; segments_gdf_proj = None

        # --- Step 4: Visualize Segments ---
        print(f"\n--- Step 4 (Area {area_label}): Generating Visualizations ---")
        if segments_gdf_proj is not None and not segments_gdf_proj.empty and target_bounds_proj:
            print(f"  Visualizing {len(segments_gdf_proj)} projected segments...")
            title_base = f"Strava Segments (Area {area_label}: {place_name})"
            plot_segments_styled(segments_gdf=segments_gdf_proj, target_bounds=target_bounds_proj, output_png_path=vis_segments_only_path,
                                 title=f"{title_base} - Color by Effort / Labeled", value_col='effort_count')
            plot_segments_styled_with_basemap(segments_gdf=segments_gdf_proj, target_bounds=target_bounds_proj, target_crs_str=TARGET_CRS,
                                              output_png_path=vis_segments_with_basemap_path, title=f"{title_base} - Color by Effort / Labeled", value_col='effort_count')
        elif segments_gdf is not None and not segments_gdf.empty and segments_gdf_proj is None and target_bounds_proj: print("  Skipping visualization: Reprojection failed.")
        else:
            reason = "no valid projected segments"
            if not target_bounds_proj: reason = "target bounds missing/failed"
            elif segments_gdf is None or segments_gdf.empty: reason = "no segments/GDF failed"
            print(f"  Skipping visualization ({reason}).")

    except Exception as e:
        print(f"\n--- Unexpected Error during processing Area {area_label} ---")
        print(f"Error Type: {type(e).__name__}"); print(f"Error Details: {e}")
        traceback.print_exc(); print("--------------------")
        return None, None, None # Return None for all results on error
    finally:
        area_end_time = time.time()
        print(f"===== Finished Processing Area {area_label} in {area_end_time - area_start_time:.2f} seconds =====")

    # Return the original pandas DataFrame, the projected GeoDataFrame, and center coords
    return segments_df, segments_gdf_proj, center_coords_wgs84

# =============================================================================
# --- Comparison Summary Function ---
# =============================================================================

def generate_comparison_summary(area_data, area_configs):
    """
    Generates a comparison DataFrame summarizing mean numeric values across areas.

    Args:
        area_data (dict): Dictionary where keys are area labels ('A', 'B', ...)
                          and values are the pandas DataFrames (segments_df) for each area.
        area_configs (list): The original list of area configuration dictionaries.

    Returns:
        pandas.DataFrame: A DataFrame summarizing the comparison, or None if no data.
    """
    print("\n--- Generating Comparison Summary ---")
    comparison_rows = []
    numeric_cols_for_mean = [ # Define numeric columns for comparison means
        'distance_m', 'average_grade_percent', 'maximum_grade_percent',
        'elevation_high_m', 'elevation_low_m', 'total_elevation_gain_m',
        'athlete_count', 'effort_count', 'star_count']
    config_map = {cfg['label']: cfg for cfg in area_configs}

    for area_label, df in area_data.items():
        if df is None or df.empty: print(f"  Skipping Area {area_label} (no data)."); continue
        print(f"  Calculating means for Area {area_label}...")
        area_config = config_map.get(area_label)
        if not area_config: place_name, radius_km = f"Area {area_label}", "N/A"
        else: place_name, radius_km = area_config.get('place_name', f"Area {area_label}"), area_config.get('radius_km', "N/A")
        summary_row = {'Area Label': area_label, 'Place Name': place_name, 'Radius (km)': radius_km, 'Segment Count': len(df)}
        for col in numeric_cols_for_mean:
            col_mean_name = f"Mean {col.replace('_',' ').title()}"
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]): summary_row[col_mean_name] = df[col].mean(skipna=True)
            else: summary_row[col_mean_name] = np.nan
        comparison_rows.append(summary_row)

    if not comparison_rows: print("  No data for comparison summary."); return None
    comparison_df = pd.DataFrame(comparison_rows).set_index('Area Label')
    print("  Comparison Summary DataFrame created:")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000): print(comparison_df)
    return comparison_df

# =============================================================================
# --- NEW: Overview Map Function ---
# =============================================================================

def plot_overview_map(area_locations, output_path):
    """
    Generates and saves an overview map showing the center points of all processed areas.

    Args:
        area_locations (dict): Dictionary where keys are area labels and values are
                               tuples of (latitude, longitude, place_name) in WGS84.
        output_path (str): Full path to save the overview map PNG.
    """
    print("\n--- Generating Overview Map ---")
    if not area_locations:
        print("  Skipping overview map: No valid area locations found.")
        return

    try:
        # Create lists of coordinates and labels
        lats = [loc[0] for loc in area_locations.values()]
        lons = [loc[1] for loc in area_locations.values()]
        labels = [loc[2] for loc in area_locations.values()] # Place names
        area_labels = list(area_locations.keys()) # 'A', 'B', ...

        # Create a GeoDataFrame from the WGS84 coordinates
        points_gdf = gpd.GeoDataFrame(
            {'label': area_labels, 'place_name': labels},
            geometry=gpd.points_from_xy(lons, lats),
            crs=SOURCE_CRS_GPS # Set initial CRS to WGS84
        )

        # Reproject to the target CRS for plotting
        print(f"  Reprojecting {len(points_gdf)} area centers to {TARGET_CRS}...")
        points_gdf_proj = points_gdf.to_crs(TARGET_CRS)
        print("  Reprojection complete.")

        # Calculate map bounds based on projected points
        minx, miny, maxx, maxy = points_gdf_proj.total_bounds
        # Add padding to the bounds (e.g., 10% of the range)
        x_range = maxx - minx
        y_range = maxy - miny
        padding_x = x_range * 0.1
        padding_y = y_range * 0.1
        # Handle cases where range is zero (e.g., single point)
        if padding_x == 0: padding_x = 5000 # Add 5km padding if only one point horizontally
        if padding_y == 0: padding_y = 5000 # Add 5km padding if only one point vertically

        map_bounds = (minx - padding_x, miny - padding_y, maxx + padding_x, maxy + padding_y)

        # --- Create the plot ---
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_xlim(map_bounds[0], map_bounds[2])
        ax.set_ylim(map_bounds[1], map_bounds[3])
        ax.set_aspect('equal', adjustable='box')
        ax.set_axis_off()

        # Plot the points
        points_gdf_proj.plot(
            ax=ax,
            color=OVERVIEW_MARKER_COLOR,
            markersize=OVERVIEW_MARKER_SIZE,
            zorder=10
        )

        # Add labels
        texts = []
        for x, y, label in zip(points_gdf_proj.geometry.x, points_gdf_proj.geometry.y, points_gdf_proj['place_name']):
            texts.append(ax.text(x, y, label, fontsize=OVERVIEW_LABEL_FONTSIZE, color=OVERVIEW_LABEL_COLOR, zorder=11))

        # Use adjustText if available to prevent label overlap
        if ADJUST_TEXT_AVAILABLE:
            print("  Adjusting label positions to reduce overlap...")
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))
        else:
             print("  (adjustText not available, labels might overlap)")


        # Add basemap
        print("  Adding basemap...")
        try:
            cx.add_basemap(ax, crs=points_gdf_proj.crs, source=CTX_PROVIDER, zoom='auto', alpha=CTX_ALPHA_BASEMAP, zorder=1)
        except Exception as ctx_e:
            print(f"    ERROR adding contextily basemap: {type(ctx_e).__name__} - {ctx_e}")

        ax.set_title("Overview Map of Processed Area Centers", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight') # Use slightly lower DPI if needed
        plt.close(fig)
        print(f"  Overview map saved successfully to: {output_path}")

    except Exception as e:
        print(f"  ERROR generating overview map: {type(e).__name__} - {e}")
        traceback.print_exc()
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)


# =============================================================================
# --- Main Execution ---
# =============================================================================

# --- Authenticate with Strava ONCE ---
print("\n--- Authenticating with Strava ---")
master_strava_client = get_strava_client(OUTPUT_BASE_DIR)

area_data_frames = {} # Dictionary to store the main pandas DataFrame for each area label
area_locations = {}   # Dictionary to store {label: (lat, lon, place_name)}

if master_strava_client:
    print("Strava authentication successful.")
    # --- Process Each Area Defined in Configuration ---
    print("\n--- Processing Areas ---")
    for config in AREAS_TO_PROCESS:
        area_label = config['label']
        place_name = config['place_name']
        # Call process_area, store the returned pandas DataFrame and center coordinates
        segments_df_result, _, center_coords = process_area(master_strava_client, config) # Get center_coords

        # Store the df result
        area_data_frames[area_label] = segments_df_result

        # Store location if successfully geocoded
        if center_coords:
            area_locations[area_label] = (center_coords[0], center_coords[1], place_name)

        # --- Create Global Variable for Interactive Use ---
        global_var_name = f'df_{area_label}'
        globals()[global_var_name] = segments_df_result
        if segments_df_result is not None:
            print(f"  DataFrame for Area {area_label} assigned to global variable: {global_var_name} (Rows: {len(segments_df_result)})")
        else:
            print(f"  Processing failed for Area {area_label}, global variable '{global_var_name}' set to None.")
        # --- End Global Variable Assignment ---


    # --- Generate and Save Comparison Summary ---
    if area_data_frames:
        comparison_df = generate_comparison_summary(area_data_frames, AREAS_TO_PROCESS)
        if comparison_df is not None and not comparison_df.empty:
            comparison_csv_path = os.path.join(OUTPUT_BASE_DIR, COMPARISON_CSV_FILENAME)
            print(f"\n--- Saving Comparison Summary ---")
            try:
                comparison_df.to_csv(comparison_csv_path, index=True, encoding='utf-8')
                print(f"Comparison summary saved successfully to: {comparison_csv_path}")
                globals()['comparison_df'] = comparison_df # Make comparison DF global too
                print(f"Comparison DataFrame assigned to global variable: comparison_df")
            except Exception as e: print(f"ERROR saving comparison summary CSV: {type(e).__name__} - {e}"); traceback.print_exc()
        else: print("\nComparison summary DataFrame empty or not generated. Skipping save.")

    # --- Generate and Save Overview Map ---
    if area_locations:
        overview_map_path = os.path.join(OUTPUT_BASE_DIR, OVERVIEW_MAP_FILENAME)
        plot_overview_map(area_locations, overview_map_path)
    else:
        print("\nSkipping overview map generation: No area locations were successfully processed.")

else:
    print("\nFATAL ERROR: Strava authentication failed. Cannot proceed.")
    sys.exit(1)


# --- Final Completion ---
overall_end_time = time.time()
print(f"\nTotal execution time for all areas: {overall_end_time - overall_start_time:.2f} seconds")
print("\n--- Strava Segment Plotter (Comparison) Workflow Finished ---")
print(f"Outputs should be in subdirectories within: {OUTPUT_BASE_DIR}")
print(f"Comparison summary saved to: {os.path.join(OUTPUT_BASE_DIR, COMPARISON_CSV_FILENAME)}")
print(f"Overview map saved to: {os.path.join(OUTPUT_BASE_DIR, OVERVIEW_MAP_FILENAME)}")
print("\nIndividual area DataFrames (e.g., df_A, df_B) should be available in your interactive session's variable explorer.")

