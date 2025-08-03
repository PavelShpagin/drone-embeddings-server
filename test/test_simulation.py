#!/usr/bin/env python3
"""
Test Simulation
===============
Tests complete simulation with path visualization.
"""

import requests
import json
import numpy as np
import time
from pathlib import Path


def test_simulation(server_url):
    """Test complete simulation with device mode and path visualization."""
    print("Testing Complete Simulation")
    print("=" * 40)
    print(f"Server URL: {server_url}")
    
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
    
    print(f"Created session: {session_id[:8]}")
    print(f"Patches: {len(map_data.get('patches', []))}")
    
    # Create directories
    maps_dir = Path("data/maps")
    embeddings_dir = Path("data/embeddings")
    paths_dir = Path("data/paths")
    maps_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    paths_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full map
    if "full_map" in map_data:
        map_array = np.array(map_data["full_map"], dtype=np.uint8)
        from PIL import Image
        map_image = Image.fromarray(map_array)
        map_path = maps_dir / f"simulation_map_{session_id[:8]}.jpg"
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
        embeddings_path = embeddings_dir / f"simulation_embeddings_{session_id[:8]}.json"
        with open(embeddings_path, 'w') as f:
            json.dump(embeddings_data, f, indent=2, default=str)
        print(f"Saved embeddings: {embeddings_path}")
    
    # Process stream frames
    stream_path = Path("../device/data/stream")
    if not stream_path.exists():
        print(f"Stream path not found: {stream_path}")
        return False
    
    frames = [1000, 1500, 2000, 2500, 3000]
    image_files = sorted(stream_path.glob("*.jpg"))
    
    if len(image_files) < max(frames):
        print(f"Not enough images. Found {len(image_files)}, need {max(frames)}")
        return False
    
    print("Processing frames...")
    for i, frame_num in enumerate(frames):
        image_file = image_files[frame_num]
        print(f"Frame {frame_num}: {image_file.name}")
        
        try:
            with open(image_file, 'rb') as f:
                files = {'image': f}
                data = {'session_id': session_id}
                
                response = requests.post(f"{server_url}/fetch_gps", 
                                       files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    gps = result.get("gps", {})
                    print(f"  GPS: {gps.get('lat', 0):.6f}, {gps.get('lng', 0):.6f}")
                else:
                    print(f"  Error: {result.get('error')}")
            else:
                print(f"  HTTP Error: {response.status_code}")
            
            time.sleep(0.2)
            
        except Exception as e:
            print(f"  Error processing frame {frame_num}: {e}")
    
    # Generate final path visualization
    print("Generating path visualization...")
    viz_response = requests.post(f"{server_url}/visualize_path", 
                               json={"session_id": session_id}, timeout=30)
    
    if viz_response.status_code == 200:
        # Save to client_paths directory
        client_paths_dir = Path("data/client_paths")
        client_paths_dir.mkdir(parents=True, exist_ok=True)
        
        viz_path = client_paths_dir / f"simulation_path_{session_id[:8]}.jpg"
        with open(viz_path, 'wb') as f:
            f.write(viz_response.content)
        print(f"Path visualization saved: {viz_path}")
        print(f"Image size: {len(viz_response.content)} bytes")
    else:
        print(f"Visualization failed: {viz_response.status_code}")
        return False
    
    print("Simulation test: SUCCESS")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test complete simulation with path visualization")
    parser.add_argument("--remote", action="store_true", 
                       help="Test against remote AWS server instead of localhost")
    parser.add_argument("--port", type=int, default=5000,
                       help="Port number (default: 5000)")
    
    args = parser.parse_args()
    
    if args.remote:
        aws_dns = "ec2-16-171-238-14.eu-north-1.compute.amazonaws.com"
        server_url = f"http://{aws_dns}:{args.port}"
    else:
        server_url = f"http://localhost:{args.port}"
    
    test_simulation(server_url)


if __name__ == "__main__":
    main()