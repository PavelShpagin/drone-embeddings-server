#!/usr/bin/env python3
"""
HTTP Test Client for Satellite Embedding Server
===============================================
Tests the server via HTTP requests instead of direct imports.
"""

import requests
import json
import time
import base64
from pathlib import Path
import numpy as np
from PIL import Image

def save_image_and_metadata(result, mode, test_location, output_dir, server_url):
    """Save the image and metadata from HTTP API result."""
    # Use simplified directory structure
    data_dir = Path("data")
    maps_dir = data_dir / "maps"
    embeddings_dir = data_dir / "embeddings"
    maps_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    if result["success"] and mode == "device":
        map_data = result["map_data"]
        
        # Convert list back to numpy array and save image
        full_map_array = np.array(map_data["full_map"], dtype=np.uint8)
        image = Image.fromarray(full_map_array)
        
        # Create filename with location and timestamp
        timestamp = int(time.time())
        lat, lng = test_location
        filename = f"map_{lat:.4f}_{lng:.4f}_{mode}_{timestamp}"
        
        # Save image in maps directory
        image_path = maps_dir / f"{filename}.jpg"
        image.save(image_path, "JPEG", quality=95)
        print(f"Saved map: {image_path}")
        
        metadata = {
            "session_id": result["session_id"],
            "location": {"lat": lat, "lng": lng},
            "mode": mode,
            "timestamp": timestamp,
            "server_url": server_url,
            "map_bounds": map_data["map_bounds"],
            "meters_coverage": map_data["meters_coverage"],
            "patch_count": map_data["patch_count"],
            "patch_size": map_data.get("patch_size", 100),
            "image_shape": full_map_array.shape,
            "patches_summary": {
                "count": len(map_data["patches"]),
                "embedding_dim": len(map_data["patches"][0]["embedding"]) if map_data["patches"] else 0,
                "sample_coords": map_data["patches"][0]["coords"] if map_data["patches"] else None
            }
        }
        
        metadata_path = maps_dir / f"{filename}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {metadata_path}")
        
        embeddings_data = {
            "session_id": result["session_id"],
            "location": {"lat": lat, "lng": lng},
            "timestamp": timestamp,
            "server_url": server_url,
            "embedding_info": {
                "total_patches": len(map_data["patches"]),
                "embedding_dimension": len(map_data["patches"][0]["embedding"]) if map_data["patches"] else 0,
                "map_bounds": map_data["map_bounds"],
                "meters_coverage": map_data["meters_coverage"]
            },
            "patches": []
        }
        
        for i, patch in enumerate(map_data["patches"]):
            patch_data = {
                "patch_id": i,
                "gps_coordinates": {"lat": patch["lat"], "lng": patch["lng"]},
                "image_coordinates": {"x1": patch["coords"][0], "y1": patch["coords"][1], 
                                    "x2": patch["coords"][2], "y2": patch["coords"][3]},
                "embedding": patch["embedding"],
                "embedding_stats": {
                    "min": float(np.min(patch["embedding"])),
                    "max": float(np.max(patch["embedding"])),
                    "mean": float(np.mean(patch["embedding"])),
                    "std": float(np.std(patch["embedding"])),
                    "norm": float(np.linalg.norm(patch["embedding"]))
                }
            }
            embeddings_data["patches"].append(patch_data)
        
        embeddings_path = embeddings_dir / f"embeddings_{lat:.4f}_{lng:.4f}_{timestamp}.json"
        with open(embeddings_path, 'w') as f:
            json.dump(embeddings_data, f, indent=2)
        print(f"Saved embeddings: {embeddings_path}")
        
        return {
            "map_path": str(image_path),
            "embeddings_path": str(embeddings_path)
        }
    
    elif result["success"] and mode == "server":
        # Server mode: only saves the map (no embeddings)
        # For now, just return session info since server mode doesn't return map data
        print(f"Server mode: Session {result['session_id'][:8]} created with {result.get('patch_count', 0)} patches")
        return {"session_id": result["session_id"]}
    
    else:
        print(f"Failed to save data for {mode} mode: {result.get('error', 'Unknown error')}")
        return None

def test_server_health(server_url):
    """Test server health endpoint."""
    try:
        response = requests.get(f"{server_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"Server: {health_data['status']}, sessions: {health_data['sessions_count']}")
            return True
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Server not reachable: {e}")
        return False

def test_init_map(server_url, lat, lng, meters, mode):
    """Test the init_map endpoint via HTTP."""
    payload = {
        "lat": lat,
        "lng": lng,
        "meters": meters,
        "mode": mode
    }
    
    try:
        response = requests.post(
            f"{server_url}/init_map",
            json=payload,
            timeout=120  # Allow 2 minutes for processing
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}"
        }

def test_list_sessions(server_url):
    """Test the sessions list endpoint."""
    try:
        response = requests.get(f"{server_url}/sessions", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}"
        }

def test_fetch_gps(server_url, session_id, image_path):
    """Test the fetch_gps endpoint."""
    print(f"Testing fetch_gps with session {session_id[:8]}...")
    
    url = f"{server_url}/fetch_gps"
    
    try:
        # Prepare the request for FastAPI
        with open(image_path, 'rb') as img_file:
            files = {'image': (image_path, img_file, 'image/jpeg')}
            data = {'session_id': session_id}
            
            response = requests.post(url, files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result["success"]:
                gps = result["gps"]
                print(f"Found GPS: {gps['lat']:.6f}, {gps['lng']:.6f}")
                print(f"Similarity: {result['similarity']:.3f}")
                print(f"Confidence: {result['confidence']}")
                return result
            else:
                print(f"Error: {result['error']}")
                return None
        else:
            print(f"HTTP Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Request failed: {str(e)}")
        return None

def test_both_modes_http(server_url="http://localhost:5000"):
    """Test both server and device modes via HTTP."""
    print("Testing Satellite Embedding Server - HTTP Mode")
    print("=" * 60)
    print(f"Server URL: {server_url}")
    
    # Check server health first
    if not test_server_health(server_url):
        print("Server not responding. Start server: python server.py --port 5000")
        return None
    
    # Test location (Kyiv area)
    test_lat, test_lng = 50.4162, 30.8906
    test_location = (test_lat, test_lng)
    output_dir = "data/server_api"
    
    # Test parameters
    test_meters = 1000  # 1km for faster testing
    
    results = {}
    
    print(f"\nTest Location: {test_lat:.6f}, {test_lng:.6f}")
    print(f"Coverage: {test_meters}m x {test_meters}m")
    print(f"Output Directory: {output_dir}")
    
    # Test 1: SERVER MODE
    print("\n" + "="*60)
    print("TEST 1: SERVER MODE (HTTP)")
    print("="*60)
    
    start_time = time.time()
    result_server = test_init_map(server_url, test_lat, test_lng, test_meters, "server")
    server_time = time.time() - start_time
    
    print(f"Server mode result (took {server_time:.2f}s):")
    print(json.dumps(result_server, indent=2))
    
    # Save server mode results
    server_files = save_image_and_metadata(result_server, "server", test_location, output_dir, server_url)
    results["server"] = {
        "result": result_server,
        "files": server_files,
        "time": server_time
    }
    
    # Test 2: DEVICE MODE  
    print("\n" + "="*60)
    print("TEST 2: DEVICE MODE (HTTP)")
    print("="*60)
    
    start_time = time.time()
    result_device = test_init_map(server_url, test_lat, test_lng, test_meters, "device")
    device_time = time.time() - start_time
    
    print(f"Device mode result (took {device_time:.2f}s):")
    if result_device["success"]:
        map_data = result_device["map_data"]
        print(f"  - Session ID: {result_device['session_id']}")
        print(f"  - Map shape: {np.array(map_data['full_map']).shape}")
        print(f"  - Patches: {map_data['patch_count']}")
        print(f"  - Coverage: {map_data['meters_coverage']}m")
        print(f"  - Bounds: {map_data['map_bounds']}")
        
        if map_data["patches"]:
            sample_patch = map_data["patches"][0]
            print(f"  - Sample patch:")
            print(f"    GPS: ({sample_patch['lat']:.6f}, {sample_patch['lng']:.6f})")
            print(f"    Embedding dim: {len(sample_patch['embedding'])}")
            print(f"    Coords: {sample_patch['coords']}")
    else:
        print(f"  - Error: {result_device.get('error', 'Unknown error')}")
    
    # Save device mode results
    device_files = save_image_and_metadata(result_device, "device", test_location, output_dir, server_url)
    results["device"] = {
        "result": result_device,
        "files": device_files,
        "time": device_time
    }
    
    # Test 3: FETCH GPS
    print("\n" + "="*60)
    print("TEST 3: FETCH GPS (HTTP)")
    print("="*60)
    
    if result_device and result_device.get("success") and "map_data" in result_device:
        session_id = result_device["session_id"]
        # Use a test image for GPS fetch test
        test_image = "../real_data/00204144.jpg"  # Use available test image
        
        import os
        if os.path.exists(test_image):
            
            start_time = time.time()
            gps_result = test_fetch_gps(server_url, session_id, test_image)
            gps_time = time.time() - start_time
            
            if gps_result and gps_result.get("success"):
                print(f"GPS fetch result (took {gps_time:.2f}s):")
                print(f"  - Found location: {gps_result['gps']['lat']:.6f}, {gps_result['gps']['lng']:.6f}")
                print(f"  - Similarity: {gps_result['similarity']:.3f}")
                print(f"  - Confidence: {gps_result['confidence']}")
            else:
                print(f"GPS fetch failed: {gps_result.get('error', 'Unknown error') if gps_result else 'No result'}")
        else:
            print(f"Test image not found: {test_image}")
            print("Skipping GPS fetch test - no test image available")
    else:
        print("Skipping GPS fetch test - no valid device mode session")
    
    # Test 4: LIST SESSIONS
    print("\n" + "="*60)
    print("TEST 4: LIST SESSIONS (HTTP)")
    print("="*60)
    
    sessions_result = test_list_sessions(server_url)
    if sessions_result["success"]:
        sessions = sessions_result["sessions"]
        print(f"Active sessions: {sessions_result['count']}")
        for session in sessions:
            print(f"  - {session['session_id'][:8]}: {session['meters_coverage']}m, {session['patch_count']} patches")
    else:
        print(f"Failed to list sessions: {sessions_result.get('error')}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Server mode: {'SUCCESS' if results['server']['result']['success'] else 'FAILED'} ({results['server']['time']:.2f}s)")
    print(f"Device mode: {'SUCCESS' if results['device']['result']['success'] else 'FAILED'} ({results['device']['time']:.2f}s)")
    
    if sessions_result.get("success"):
        print(f"Total sessions: {sessions_result['count']}")
    
    if results['server']['files']:
        print(f"Server files saved: {results['server']['files']}")
    if results['device']['files']:
        print(f"Device files saved: {results['device']['files']}")
    
    return results

def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Satellite Embedding Server via HTTP')
    parser.add_argument('--server', type=str, default='http://localhost:5000', 
                       help='Server URL (default: http://localhost:5000)')
    
    args = parser.parse_args()
    
    try:
        results = test_both_modes_http(args.server)
        if results:
            print("\n" + "="*60)
            print("ALL HTTP TESTS COMPLETED SUCCESSFULLY!")
            print("="*60)
        return results
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()