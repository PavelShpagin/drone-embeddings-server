#!/usr/bin/env python3
"""
Test Fetch GPS Device Mode - New Architecture
============================================
Tests new zip-based device mode and file-based session caching.
"""

import requests
import json
import zipfile
import numpy as np
from pathlib import Path
import tempfile
import os
from PIL import Image
import io


def test_init_map_device_mode(server_url):
    """Test init_map in device mode (returns zip file)."""
    print("Testing init_map Device Mode (New Architecture)")
    print("=" * 50)
    print(f"Server URL: {server_url}")
    
    # First get the session_id from server mode to get full UUID
    server_payload = {
        "lat": 50.4162,
        "lng": 30.8906,
        "meters": 1000,
        "mode": "server"
    }
    
    print("\n1. Getting session_id from server mode...")
    server_response = requests.post(f"{server_url}/init_map", json=server_payload, timeout=120)
    
    if server_response.status_code != 200:
        print(f"HTTP Error: {server_response.status_code}")
        return False
    
    server_result = server_response.json()
    if not server_result.get("success"):
        print(f"Error: {server_result.get('error')}")
        return False
    
    session_id = server_result["session_id"]
    print(f"✓ Created session: {session_id}")
    
    # Test 1: New session creation in device mode
    payload = {
        "lat": 50.4162,
        "lng": 30.8906,
        "meters": 1000,
        "mode": "device"
    }
    
    print("\n2. Testing device mode with new session...")
    response = requests.post(f"{server_url}/init_map", json=payload, timeout=120)
    
    if response.status_code != 200:
        print(f"HTTP Error: {response.status_code}")
        return False
    
    # Should return zip file
    if response.headers.get('content-type') != 'application/zip':
        print(f"Expected zip file, got: {response.headers.get('content-type')}")
        return False
    
    # Verify zip file received
    print(f"✓ Received zip file ({len(response.content)} bytes)")
    # Note: We already have the full session_id from the server mode call above
    
    # Save and inspect zip
    zip_path = Path("test_device_session.zip")
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    print(f"✓ Saved zip file: {zip_path} ({len(response.content)} bytes)")
    
    # Extract and verify zip contents
    with zipfile.ZipFile(zip_path, 'r') as zf:
        files = zf.namelist()
        print(f"✓ Zip contains: {files}")
        
        if 'map.png' not in files or 'embeddings.json' not in files:
            print("✗ Zip missing required files")
            return False
        
        # Extract map
        map_data = zf.read('map.png')
        map_image = Image.open(io.BytesIO(map_data))
        print(f"✓ Map image: {map_image.size} ({map_image.mode})")
        
        # Extract embeddings
        embeddings_data = json.loads(zf.read('embeddings.json').decode())
        print(f"✓ Embeddings: {len(embeddings_data['patches'])} patches")
        print(f"✓ Coverage: {embeddings_data['meters_coverage']}m")
    
    # Test 3: Session caching
    print("\n3. Testing session caching...")
    cached_payload = {
        "lat": 50.4162,  # These will be ignored
        "lng": 30.8906,
        "meters": 1000,
        "mode": "device",
        "session_id": session_id  # Request cached session
    }
    
    response = requests.post(f"{server_url}/init_map", json=cached_payload, timeout=30)
    
    if response.status_code == 200 and response.headers.get('content-type') == 'application/zip':
        print(f"✓ Cache hit: received zip file ({len(response.content)} bytes)")
    else:
        print("✗ Cache miss or error")
        return False
    
    # Cleanup
    zip_path.unlink()
    
    return session_id


def test_fetch_gps_enhanced(server_url, session_id):
    """Test enhanced fetch_gps with logging."""
    print(f"\n4. Testing enhanced fetch_gps with session: {session_id[:8]}")
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    pil_image = Image.fromarray(dummy_image)
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format='JPEG')
    img_bytes = img_buffer.getvalue()
    
    # Test basic fetch_gps
    files = {'image': ('test.jpg', img_bytes, 'image/jpeg')}
    data = {'session_id': session_id}
    
    response = requests.post(f"{server_url}/fetch_gps", files=files, data=data, timeout=30)
    
    if response.status_code != 200:
        print(f"✗ HTTP Error: {response.status_code}")
        return False
    
    result = response.json()
    if not result.get("success"):
        print(f"✗ Error: {result.get('error')}")
        return False
    
    gps = result.get("gps")
    print(f"✓ GPS prediction: {gps['lat']:.6f}, {gps['lng']:.6f}")
    print(f"✓ Similarity: {result.get('similarity', 0):.3f}")
    
    # Test enhanced logging
    print("\n5. Testing enhanced logging and visualization...")
    data.update({
        'logging_id': 'test_logger_device',
        'visualization': 'true'
    })
    
    response = requests.post(f"{server_url}/fetch_gps", files=files, data=data, timeout=30)
    
    if response.status_code != 200:
        print(f"✗ HTTP Error: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    
    result = response.json()
    
    if result.get("success"):
        print("✓ Enhanced logging request successful")
    else:
        print(f"✗ Enhanced logging failed: {result.get('error')}")
        print(f"Full response: {result}")
        return False
    
    return True


def test_sessions_endpoint(server_url):
    """Test sessions listing endpoint."""
    print("\n6. Testing sessions endpoint...")
    
    response = requests.get(f"{server_url}/sessions", timeout=30)
    
    if response.status_code != 200:
        print(f"✗ HTTP Error: {response.status_code}")
        return False
    
    result = response.json()
    if not result.get("success"):
        print(f"✗ Error: {result.get('error')}")
        return False
    
    sessions = result.get("sessions", [])
    print(f"✓ Found {len(sessions)} sessions")
    for session in sessions:
        print(f"  - {session['session_id'][:8]}: {session['meters_coverage']}m coverage, {session['patch_count']} patches")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test new architecture device mode")
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
    
    print("New Architecture Device Mode Test Suite")
    print("=" * 50)
    
    # Test device mode init_map
    session_id = test_init_map_device_mode(server_url)
    if not session_id:
        print("\n✗ Device mode test FAILED")
        return
    
    # Test enhanced fetch_gps
    if not test_fetch_gps_enhanced(server_url, session_id):
        print("\n✗ Enhanced fetch_gps test FAILED")
        return
    
    # Test sessions endpoint
    if not test_sessions_endpoint(server_url):
        print("\n✗ Sessions endpoint test FAILED")
        return
    
    print("\n✓ All tests PASSED - New architecture working correctly!")


if __name__ == "__main__":
    main()