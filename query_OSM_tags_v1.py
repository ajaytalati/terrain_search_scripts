#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script to query the Overpass API for unique values of a specific
OSM tag within a defined geographical bounding box.

This script helps in understanding the different values used for a tag
(like 'surface', 'highway', 'tracktype', etc.) in a particular area.
"""

import requests
import json
from collections import Counter

# =============================================================================
# --- Configuration ---
# =============================================================================

# --- Tag to Query ---
# Specify the OSM tag key you want to find unique values for.
# Examples: "surface", "highway", "tracktype", "smoothness", "access"
#TAG_KEY = "surface"
#TAG_KEY = "highway"
TAG_KEY = "tracktype"


# --- Region Bounding Box ---
# Define the geographical area to query using coordinates (min_lat, min_lon, max_lat, max_lon).
# Format: "south_latitude,west_longitude,north_latitude,east_longitude"
# Example: Buxton region (approx 0.5 degree box)
REGION_BBOX = "53.01,-2.16,53.51,-1.66"
# Example: UK region (approx) - Use with caution, may time out!
# REGION_BBOX = "49.8,-8.2,60.9,1.8"
# Example: Remove bounding box for wider query (HIGHLY LIKELY TO TIMEOUT on public servers)
# REGION_BBOX = None # Set to None to query without a bounding box filter

# --- Overpass API Settings ---
# Public Overpass API endpoint. Consider running your own instance for heavy use.
# Alternatives: "https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"
OVERPASS_URL = "https://lz4.overpass-api.de/api/interpreter"
# Timeout for the API request in seconds. Increase for larger areas/complex queries.
API_TIMEOUT = 60 if REGION_BBOX else 180 # Shorter timeout for smaller bbox

# --- Query Element Types ---
# Specify which OSM element types to include in the query (node, way, relation).
# Querying all is most comprehensive but slower. 'way' is often sufficient for path/road tags.
ELEMENT_TYPES = ["way"] # Options: ["node", "way", "relation"] or a subset

# =============================================================================
# --- Script Logic (Usually no need to modify below) ---
# =============================================================================

def build_overpass_query(tag_key, element_types, bbox=None):
    """Builds the Overpass QL query string dynamically."""
    query_parts = []
    bbox_filter = f"({bbox})" if bbox else "" # Add bbox filter if provided

    # Create query lines for each element type
    for elem_type in element_types:
        query_parts.append(f'  {elem_type}["{tag_key}"]{bbox_filter};')

    # Combine element queries into the final query structure
    full_query = f"""
[out:json][timeout:{API_TIMEOUT}];
(
  // Query specified element types within the bounding box (if any)
  // that have the '{tag_key}' tag defined.
{chr(10).join(query_parts)}
);
// Output only the tags of the found elements to minimize data transfer
out tags;
"""
    return full_query

# --- Main Execution ---
if __name__ == "__main__":

    # Build the query based on configuration
    query = build_overpass_query(TAG_KEY, ELEMENT_TYPES, REGION_BBOX)

    # Inform the user what is being queried
    region_desc = f"region ({REGION_BBOX})" if REGION_BBOX else "globally (may time out!)"
    print(f"Querying Overpass API ({OVERPASS_URL}) for unique '{TAG_KEY}' values")
    print(f"Element types: {', '.join(ELEMENT_TYPES)}")
    print(f"Querying {region_desc}...")
    print(f"Timeout set to {API_TIMEOUT} seconds.")
    # print("\n--- Overpass Query ---") # Uncomment to see the generated query
    # print(query)
    # print("----------------------\n")


    try:
        # Make the request to the Overpass API
        print("Sending request...")
        response = requests.get(
            OVERPASS_URL,
            params={'data': query},
            headers={'User-Agent': f'PythonTagQueryScript/1.3 ({TAG_KEY})'} # Include tag in user agent
        )
        # Check for HTTP errors (like 4xx or 5xx)
        response.raise_for_status()
        print("Query successful. Processing results...")

        # Parse the JSON response
        data = response.json()

        # --- Debug: Print number of elements received ---
        elements_received = data.get('elements', [])
        print(f"DEBUG: Received {len(elements_received)} elements from Overpass API.")
        # --- End Debug ---

        # Use a set to store unique tag values found
        unique_values = set()
        # Use a Counter to count occurrences (optional, but interesting)
        value_counts = Counter()

        # Iterate through the elements in the response data
        processed_count = 0
        for element in elements_received:
            # Check if the element has tags and if the specific tag key exists
            if 'tags' in element and TAG_KEY in element['tags']:
                tag_value = element['tags'][TAG_KEY]
                unique_values.add(tag_value)
                value_counts[tag_value] += 1
                processed_count += 1

        print(f"DEBUG: Processed {processed_count} elements containing the '{TAG_KEY}' tag.")

        # Print the results
        if unique_values:
            print(f"\nFound {len(unique_values)} unique '{TAG_KEY}' values in the queried region:")
            # Sort the values alphabetically for readability
            for value in sorted(list(unique_values)):
                print(f"- {value} (Count: {value_counts[value]})")
        else:
            # Provide more specific feedback based on whether elements were received
            if elements_received:
                 print(f"\nReceived data from API, but no elements had the '{TAG_KEY}' tag in the queried region.")
            else:
                 print(f"\nNo elements with the '{TAG_KEY}' tag found in the queried region OR no data received from API.")
                 print("Possible reasons: Tag not used in area, query too broad (timed out?), temporary API issue.")

    # --- Error Handling ---
    except requests.exceptions.Timeout:
        print(f"\nError: The request to Overpass API timed out ({API_TIMEOUT} seconds).")
        print("Consider reducing the bounding box size, simplifying the query, or increasing the API_TIMEOUT.")
    except requests.exceptions.RequestException as e:
        print(f"\nError during Overpass API request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            # Try to get more specific Overpass error messages if available
            try:
                error_data = e.response.json()
                if 'remark' in error_data:
                    print(f"Overpass API remark: {error_data['remark']}")
            except json.JSONDecodeError:
                 print(f"Response text (first 500 chars): {e.response.text[:500]}...")
            except Exception: # Catch any other error during error reporting
                 print(f"Response text (first 500 chars): {e.response.text[:500]}...")

    except json.JSONDecodeError as e:
        print(f"\nError decoding JSON response from Overpass API: {e}")
        print(f"Response text (first 500 chars): {response.text[:500]}...")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

