#!/usr/bin/env python3
"""
Simple local test for new architecture
"""

import requests
import json
import numpy as np
from PIL import Image
import io
from pathlib import Path

def test_local_server():
    """Test against local server."""
    server_url = "http://localhost:5000"
    
    print("Testing Local Server - New Architecture")
    print("=" * 45)
    
    # Test 1: Create session
    print("1. Creating new session...")
    payload = {
        "lat": 50.4162,
        "lng": 30.8906,
        "meters": 1000,
        "mode": "server"
    }
    
    response = requests.post(f"{server_url}/init_map", json=payload, timeout=60)
    
    if response.status_code != 200:
        print(f"✗ HTTP Error: {response.status_code}")
        return False
    
    result = response.json()
    if not result.get("success"):
        print(f"✗ Error: {result.get('error')}")
        return False
    
    session_id = result["session_id"]
    print(f"✓ Session created: {session_id}")
    
    # Test 2: Check local files
    print("\n2. Checking server files...")
    data_dir = Path("data")
    
    # Check sessions.pkl
    sessions_file = data_dir / "sessions.pkl"
    if sessions_file.exists():
        print(f"✓ sessions.pkl exists ({sessions_file.stat().st_size} bytes)")
    else:
        print("✗ sessions.pkl not found")
    
    # Check maps directory
    maps_dir = data_dir / "maps"
    if maps_dir.exists():
        map_files = list(maps_dir.glob("*.png"))
        print(f"✓ maps/ directory: {len(map_files)} files")
        for f in map_files:
            print(f"  - {f.name} ({f.stat().st_size} bytes)")
    else:
        print("✗ maps/ directory not found")
    
    # Check embeddings directory
    embeddings_dir = data_dir / "embeddings"
    if embeddings_dir.exists():
        emb_files = list(embeddings_dir.glob("*.json"))
        print(f"✓ embeddings/ directory: {len(emb_files)} files")
        for f in emb_files:
            print(f"  - {f.name} ({f.stat().st_size} bytes)")
    else:
        print("✗ embeddings/ directory not found")
    
    # Check zips directory
    zips_dir = data_dir / "zips"
    if zips_dir.exists():
        zip_files = list(zips_dir.glob("*.zip"))
        print(f"✓ zips/ directory: {len(zip_files)} files")
        for f in zip_files:
            print(f"  - {f.name} ({f.stat().st_size} bytes)")
    else:
        print("✗ zips/ directory not found")
    
    # Test 3: Fetch GPS
    print(f"\n3. Testing fetch_gps with session: {session_id}")
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    pil_image = Image.fromarray(dummy_image)
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format='JPEG')
    img_bytes = img_buffer.getvalue()
    
    files = {'image': ('test.jpg', img_bytes, 'image/jpeg')}
    data = {
        'session_id': session_id,
        'logging_id': 'test_local',
        'visualization': 'true'
    }
    
    response = requests.post(f"{server_url}/fetch_gps", files=files, data=data, timeout=30)
    
    if response.status_code != 200:
        print(f"✗ HTTP Error: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    
    result = response.json()
    
    if result.get("success"):
        gps = result.get("gps")
        print(f"✓ GPS prediction: {gps['lat']:.6f}, {gps['lng']:.6f}")
        print(f"✓ Similarity: {result.get('similarity', 0):.3f}")
    else:
        print(f"✗ Error: {result.get('error')}")
        return False
    
    # Test 4: Check logging files
    print("\n4. Checking enhanced logging files...")
    logs_dir = data_dir / "logs" / session_id / "test_local"
    
    if logs_dir.exists():
        print(f"✓ Log directory exists: {logs_dir}")
        
        # Check CSV
        csv_file = logs_dir / "path.csv"
        if csv_file.exists():
            print(f"✓ path.csv exists ({csv_file.stat().st_size} bytes)")
            # Show CSV content
            with open(csv_file, 'r') as f:
                content = f.read()
                print(f"  Content: {content.strip()}")
        else:
            print("✗ path.csv not found")
        
        # Check plot
        plot_file = logs_dir / "error_plot.png"
        if plot_file.exists():
            print(f"✓ error_plot.png exists ({plot_file.stat().st_size} bytes)")
        else:
            print("✗ error_plot.png not found (matplotlib might not be installed)")
        
        # Check map visualization
        map_viz_file = logs_dir / "map_paths.png"
        if map_viz_file.exists():
            print(f"✓ map_paths.png exists ({map_viz_file.stat().st_size} bytes)")
        else:
            print("✗ map_paths.png not found")
    else:
        print(f"✗ Log directory not found: {logs_dir}")
    
    print("\n✓ Local test completed!")
    return True

if __name__ == "__main__":
    test_local_server()
