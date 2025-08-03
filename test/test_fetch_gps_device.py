#!/usr/bin/env python3
"""
Test Fetch GPS Device Mode
=========================
Tests device mode and stores full map and embeddings.
"""

import requests
import json
import numpy as np
from pathlib import Path


def test_fetch_gps_device():
    """Test device mode and store outputs."""
    server_url = "http://localhost:5000"
    
    print("Testing Device Mode with Storage")
    print("=" * 40)
    
    # Initialize map in device mode
    payload = {
        "lat": 50.4162,
        "lng": 30.8906,
        "meters": 1000,
        "mode": "device"
    }
    
    response = requests.post(f"{server_url}/init_map", json=payload, timeout=60)
    
    if response.status_code != 200:
        print(f"HTTP Error: {response.status_code}")
        return False
    
    result = response.json()
    if not result.get("success"):
        print(f"Error: {result.get('error')}")
        return False
    
    session_id = result["session_id"]
    map_data = result.get("map_data", {})
    
    print(f"Session: {session_id[:8]}")
    print(f"Patches: {len(map_data.get('patches', []))}")
    
    # Create directories
    maps_dir = Path("data/maps")
    embeddings_dir = Path("data/embeddings")
    maps_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full map
    if "full_map" in map_data:
        map_array = np.array(map_data["full_map"], dtype=np.uint8)
        from PIL import Image
        map_image = Image.fromarray(map_array)
        map_path = maps_dir / f"map_{session_id[:8]}.jpg"
        map_image.save(map_path, "JPEG", quality=95)
        print(f"Saved map: {map_path}")
    
    # Save embeddings
    if "patches" in map_data:
        embeddings_data = {
            "session_id": session_id,
            "patches": map_data["patches"],
            "map_bounds": map_data.get("map_bounds", {}),
            "patch_count": len(map_data["patches"])
        }
        embeddings_path = embeddings_dir / f"embeddings_{session_id[:8]}.json"
        with open(embeddings_path, 'w') as f:
            json.dump(embeddings_data, f, indent=2, default=str)
        print(f"Saved embeddings: {embeddings_path}")
    
    print("Device mode test: SUCCESS")
    return True


if __name__ == "__main__":
    test_fetch_gps_device()