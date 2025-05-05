#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script to fetch details for a specific Strava Segment
and print all its available attributes and their values.

Version 1.1: Uses vars() for attribute inspection instead of to_dict().

Helps in debugging and understanding the data returned by the Strava API
via the stravalib library.

Requires: stravalib
Install: pip install stravalib
"""

import sys
from stravalib.client import Client
import warnings

# =============================================================================
# --- Configuration ---
# =============================================================================

# --- Strava API Credentials ---
# IMPORTANT: Replace placeholders with your actual credentials.
# Use the same credentials as your main enrichment script.
STRAVA_ACCESS_TOKEN = "434b3c810b4485511d8d215cb2d8fbb56709911c" # Replace with your Access Token (string)

# --- Segment to Inspect ---
# Find a segment ID on the Strava website. It's the number in the URL
# when viewing a segment, e.g., https://www.strava.com/segments/XXXXXXX
# Example ID (replace with one relevant to your area):
TARGET_SEGMENT_ID = 4004846 # Replace with a valid Segment ID (integer)

# =============================================================================
# --- Script Logic ---
# =============================================================================

if __name__ == "__main__":
    print(f"--- Strava Segment Inspector ---")

    # --- Validate Credentials ---
    if "YOUR_ACCESS_TOKEN" in STRAVA_ACCESS_TOKEN:
        print("\nERROR: Strava Access Token placeholder detected.")
        print("Please replace 'YOUR_ACCESS_TOKEN' in the script configuration.")
        sys.exit(1)

    # --- Validate Segment ID ---
    if not isinstance(TARGET_SEGMENT_ID, int):
        print("\nERROR: TARGET_SEGMENT_ID must be an integer.")
        sys.exit(1)

    # --- Initialize Strava Client ---
    print("Initializing Strava client...")
    client = Client()
    client.access_token = STRAVA_ACCESS_TOKEN

    # --- Fetch Segment Details ---
    print(f"\nFetching details for Segment ID: {TARGET_SEGMENT_ID}")
    try:
        # Use get_segment to fetch detailed information
        segment = client.get_segment(segment_id=TARGET_SEGMENT_ID)
        print(f"Successfully fetched details for segment: '{getattr(segment, 'name', 'N/A')}'")

        # --- Inspect Attributes ---
        print("\nAvailable attributes and values (using vars()):")
        try:
            # --- Modification: Use vars() instead of to_dict() ---
            attributes = vars(segment)
            # Sort by attribute name for readability
            for attr_name in sorted(attributes.keys()):
                # Avoid printing potentially very large internal/raw data structures
                attr_value = attributes[attr_name]
                if isinstance(attr_value, (dict, list)) and len(str(attr_value)) > 200:
                     print(f"- {attr_name}: [Object/Data Structure too large to display fully]")
                else:
                     print(f"- {attr_name}: {attr_value}")
            # --- End Modification ---
        except TypeError:
            print("  Could not use vars() on the segment object. Trying dir() instead...")
            # Fallback using dir() if vars() fails (less informative)
            for attr_name in sorted(dir(segment)):
                if not attr_name.startswith('_'): # Skip private/dunder methods
                    try:
                        attr_value = getattr(segment, attr_name)
                        # Avoid calling methods during inspection
                        if not callable(attr_value):
                             if isinstance(attr_value, (dict, list)) and len(str(attr_value)) > 200:
                                 print(f"- {attr_name}: [Object/Data Structure too large to display fully]")
                             else:
                                 print(f"- {attr_name}: {attr_value}")
                    except Exception as e:
                        print(f"- {attr_name}: [Error retrieving value: {e}]")


        # Explicitly check for the attributes we are interested in
        print("\nChecking specific attributes:")
        print(f"- id: {getattr(segment, 'id', 'Not Found')}") # ID is usually present
        print(f"- name: {getattr(segment, 'name', 'Not Found')}")
        print(f"- athlete_count: {getattr(segment, 'athlete_count', 'Not Found')}")
        print(f"- effort_count: {getattr(segment, 'effort_count', 'Not Found')}")
        print(f"- start_latlng: {getattr(segment, 'start_latlng', 'Not Found')}")
        print(f"- end_latlng: {getattr(segment, 'end_latlng', 'Not Found')}")
        print(f"- distance: {getattr(segment, 'distance', 'Not Found')}")
        print(f"- average_grade: {getattr(segment, 'average_grade', 'Not Found')}")
        print(f"- elevation_high: {getattr(segment, 'elevation_high', 'Not Found')}")
        print(f"- elevation_low: {getattr(segment, 'elevation_low', 'Not Found')}")
        print(f"- activity_type: {getattr(segment, 'activity_type', 'Not Found')}")
        print(f"- created_at: {getattr(segment, 'created_at', 'Not Found')}")
        print(f"- updated_at: {getattr(segment, 'updated_at', 'Not Found')}")
        print(f"- total_elevation_gain: {getattr(segment, 'total_elevation_gain', 'Not Found')}")
        print(f"- map: {getattr(segment, 'map', 'Not Found')}") # Map object often contains polyline


    except Exception as e:
        print(f"\nError fetching or inspecting segment {TARGET_SEGMENT_ID}: {e}")
        # Provide specific feedback for common errors
        if "Not Found" in str(e):
            print("  -> This segment ID might not exist or is private.")
        elif "AuthorizationError" in str(e):
             print("  -> Authorization Error. Check your access token and its permissions.")
        elif "RateLimitExceeded" in str(e):
             print("  -> Strava API Rate Limit Exceeded. Please wait and try again later.")


    print("\n--- Inspection Finished ---")

