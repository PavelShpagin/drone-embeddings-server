#!/usr/bin/env python3

import requests
import sys
from pathlib import Path

def test_path_fix():
    server_url = "http://localhost:5000"
    
    # Test with existing session
    session_id = "7ed00656-04ac-43a2-b450-37e9ac4ddcbf"
    
    print("Testing clean architecture...")
    
    # Call fetch_gps to create path point
    with open("../device/data/stream/00000070.jpg", "rb") as f:
        files = {"image": f}
        data = {"session_id": session_id}
        response = requests.post(f"{server_url}/fetch_gps", files=files, data=data)
        print(f"Fetch GPS: {response.json().get('success', False)}")
    
    # Check server_paths
    server_paths = list(Path("data/server_paths").glob("*.jpg"))
    print(f"Server paths created: {len(server_paths)}")
    for p in server_paths:
        print(f"  {p} ({p.stat().st_size} bytes)")
    
    # Get visualization
    viz_response = requests.post(f"{server_url}/visualize_path", 
                               json={"session_id": session_id})
    
    if viz_response.status_code == 200:
        # Save to client_paths
        client_paths_dir = Path("data/client_paths")
        client_paths_dir.mkdir(exist_ok=True)
        
        client_path = client_paths_dir / f"test_path_{session_id[:8]}.jpg"
        with open(client_path, 'wb') as f:
            f.write(viz_response.content)
        
        print(f"Client path saved: {client_path} ({len(viz_response.content)} bytes)")
        
        # Check if it's actually an image
        import subprocess
        result = subprocess.run(["file", str(client_path)], capture_output=True, text=True)
        print(f"File type: {result.stdout.strip()}")
    else:
        print(f"Visualization failed: {viz_response.status_code}")

if __name__ == "__main__":
    test_path_fix()