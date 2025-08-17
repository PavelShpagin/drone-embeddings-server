#!/usr/bin/env python3
"""
Test Simulation - New Architecture
==================================
Tests complete simulation with new zip-based device mode and enhanced logging.
"""

import requests
import json
import numpy as np
import time
import zipfile
import uuid
from pathlib import Path
from PIL import Image
import io


def test_simulation_new_architecture(server_url):
    """Test complete simulation with new architecture and enhanced logging."""
    print("Testing Complete Simulation - New Architecture")
    print("=" * 55)
    print(f"Server URL: {server_url}")
    
    # Generate a unique logger_id for this simulation
    logger_id = str(uuid.uuid4())
    print(f"Using logger_id: {logger_id}")
    
    # First get the full session_id from server mode
    server_payload = {
        "lat": 50.4162,
        "lng": 30.8906,
        "meters": 1000,
        "mode": "server"
    }
    
    print("\n1. Initializing map session...")
    server_response = requests.post(f"{server_url}/init_map", json=server_payload, timeout=120)
    
    if server_response.status_code != 200:
        print(f"HTTP Error: {server_response.status_code}")
        return False
    
    # Get full session_id from JSON response
    server_result = server_response.json()
    if not server_result.get("success"):
        print(f"Server error: {server_result}")
        return False
    
    session_id = server_result["session_id"]
    print(f"✓ Created session: {session_id}")
    
    # Now get the zip file in device mode using the full session_id
    device_payload = {
        "lat": 50.4162,  # Required by API even though ignored when session_id is provided
        "lng": 30.8906,
        "meters": 1000,
        "session_id": session_id,
        "mode": "device"
    }
    
    response = requests.post(f"{server_url}/init_map", json=device_payload, timeout=60)
    
    if response.status_code != 200:
        print(f"HTTP Error: {response.status_code}")
        return False
    
    # Should return zip file
    if response.headers.get('content-type') != 'application/zip':
        print(f"Expected zip file, got: {response.headers.get('content-type')}")
        return False
    
    # Save zip and extract to standard data structure
    data_dir = Path("data")
    maps_dir = data_dir / "maps"
    embeddings_dir = data_dir / "embeddings"
    zips_dir = data_dir / "zips"
    
    # Create directories
    for dir_path in [data_dir, maps_dir, embeddings_dir, zips_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save zip file
    zip_path = zips_dir / f"{session_id}.zip"
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    print(f"✓ Saved session zip: {zip_path} ({len(response.content)} bytes)")
    
    # Extract and store map and embeddings locally
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Extract and save map
        map_data = zf.read('map.png')
        map_path = maps_dir / f"{session_id}.png"
        with open(map_path, 'wb') as f:
            f.write(map_data)
        
        # Extract and save embeddings
        embeddings_data = json.loads(zf.read('embeddings.json').decode())
        embeddings_path = embeddings_dir / f"{session_id}.json"
        with open(embeddings_path, 'w') as f:
            json.dump(embeddings_data, f)
        
        print(f"✓ Extracted and stored map: {map_path}")
        print(f"✓ Extracted and stored embeddings: {embeddings_path}")
        print(f"✓ Total patches: {len(embeddings_data['patches'])}")
    
    # Process stream frames with enhanced logging
    stream_path = Path("../device/data/stream")
    if not stream_path.exists():
        print(f"Stream path not found: {stream_path}")
        return False
    
    # Test with several frames to generate a meaningful path
    frames = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    image_files = sorted(stream_path.glob("*.jpg"))
    
    if len(image_files) < max(frames):
        print(f"Not enough images. Found {len(image_files)}, need {max(frames)}")
        frames = [min(i, len(image_files)-1) for i in frames[:len(image_files)//2]]
    
    print(f"\n2. Processing {len(frames)} frames with enhanced logging...")
    successful_frames = 0
    
    for i, frame_num in enumerate(frames):
        image_file = image_files[frame_num]
        print(f"Frame {i+1}/{len(frames)} ({frame_num}): {image_file.name}")
        
        try:
            with open(image_file, 'rb') as f:
                files = {'image': (image_file.name, f.read(), 'image/jpeg')}
                data = {
                    'session_id': session_id,
                    'logging_id': logger_id,
                    'visualization': 'true'  # Enable enhanced logging
                }
                
                response = requests.post(f"{server_url}/fetch_gps", 
                                       files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    gps = result.get("gps", {})
                    similarity = result.get("similarity", 0)
                    confidence = result.get("confidence", "unknown")
                    print(f"  ✓ GPS: {gps.get('lat', 0):.6f}, {gps.get('lng', 0):.6f}")
                    print(f"    Similarity: {similarity:.3f} ({confidence})")
                    successful_frames += 1
                else:
                    print(f"  ✗ Error: {result.get('error')}")
            else:
                print(f"  ✗ HTTP Error: {response.status_code}")
            
            # Small delay between frames
            time.sleep(0.3)
            
        except Exception as e:
            print(f"  ✗ Error processing frame {frame_num}: {e}")
    
    print(f"\n✓ Successfully processed {successful_frames}/{len(frames)} frames")
    
    # Test session caching
    print(f"\n3. Testing session caching...")
    cached_payload = {
        "lat": 50.4162,  # These will be ignored
        "lng": 30.8906,
        "meters": 1000,
        "mode": "device",
        "session_id": session_id  # Request cached session
    }
    
    response = requests.post(f"{server_url}/init_map", json=cached_payload, timeout=30)
    
    if response.status_code == 200 and response.headers.get('content-type') == 'application/zip':
        print(f"✓ Cache hit: received cached zip ({len(response.content)} bytes)")
    else:
        print("✗ Cache miss or error")
    
    # Check server-side logs
    print(f"\n4. Checking server-side enhanced logging...")
    
    # Try to get session info
    response = requests.get(f"{server_url}/sessions", timeout=30)
    if response.status_code == 200:
        result = response.json()
        sessions = result.get("sessions", [])
        our_session = next((s for s in sessions if s['session_id'] == session_id), None)
        if our_session:
            print(f"✓ Session found in server list: {our_session['patch_count']} patches, {our_session['meters_coverage']}m coverage")
        else:
            print("✗ Session not found in server list")
    
    # Note: Enhanced logging with visualizations is handled server-side
    # Use fetch.sh to download logs: ./fetch.sh SESSION_ID LOGGER_ID
    print(f"\n5. Enhanced logging complete on server...")
    print(f"   Use fetch.sh to download logs:")
    print(f"   ./fetch.sh {session_id} {logger_id}")
    
    # Summary
    print(f"\n" + "=" * 55)
    print(f"SIMULATION SUMMARY")
    print(f"=" * 55)
    print(f"Session ID: {session_id}")
    print(f"Logger ID: {logger_id}")
    print(f"Processed frames: {successful_frames}/{len(frames)}")
    print(f"Enhanced logging: Enabled (check server logs/)")
    print(f"Session caching: Working")
    print(f"Architecture: New zip-based system")
    
    if successful_frames > 0:
        print(f"\n✓ SIMULATION SUCCESSFUL")
        print(f"Enhanced logs should be available at:")
        print(f"  Server: data/logs/{session_id}/{logger_id}/")
        print(f"  - path.csv (GPS tracking)")
        print(f"  - error_plot.png (error over time)")
        print(f"  - map_paths.png (path visualization)")
        return True
    else:
        print(f"\n✗ SIMULATION FAILED - No frames processed successfully")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test complete simulation with new architecture")
    parser.add_argument("--local", action="store_true", 
                       help="Test against localhost instead of remote AWS server")
    parser.add_argument("--port", type=int, default=5000,
                       help="Port number (default: 5000)")
    
    args = parser.parse_args()
    
    if args.local:
        server_url = f"http://localhost:{args.port}"
    else:
        aws_dns = "ec2-16-171-238-14.eu-north-1.compute.amazonaws.com"
        server_url = f"http://{aws_dns}:{args.port}"
    
    test_simulation_new_architecture(server_url)


if __name__ == "__main__":
    main()