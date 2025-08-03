#!/usr/bin/env python3

import requests
import time
from pathlib import Path

def test_path_with_lines():
    server_url = "http://localhost:5000"
    
    print("Testing path visualization with connecting lines...")
    
    # Create new session
    print("1. Creating new session...")
    init_response = requests.post(f"{server_url}/init_map", json={
        "lat": 50.4162,
        "lng": 30.8906,
        "meters": 1000,
        "mode": "device"
    })
    
    if not init_response.json().get("success"):
        print("Failed to create session")
        return
    
    session_id = init_response.json()["session_id"]
    print(f"Session created: {session_id[:8]}")
    
    # Add multiple GPS points to create a path
    stream_images = [
        "../device/data/stream/00000070.jpg",
        "../device/data/stream/00000134.jpg", 
        "../device/data/stream/00000198.jpg",
        "../device/data/stream/00000264.jpg",
        "../device/data/stream/00000352.jpg"
    ]
    
    print("\n2. Adding GPS points sequentially...")
    for i, image_path in enumerate(stream_images, 1):
        if Path(image_path).exists():
            with open(image_path, "rb") as f:
                files = {"image": f}
                data = {"session_id": session_id}
                response = requests.post(f"{server_url}/fetch_gps", files=files, data=data)
                
                if response.json().get("success"):
                    gps = response.json()["gps"]
                    print(f"  Point {i}: {gps['lat']:.6f}, {gps['lng']:.6f}")
                else:
                    print(f"  Point {i}: Failed")
            
            time.sleep(0.5)  # Small delay between points
    
    # Get final path visualization
    print("\n3. Getting path visualization...")
    viz_response = requests.post(f"{server_url}/visualize_path", 
                               json={"session_id": session_id})
    
    if viz_response.status_code == 200:
        # Save to client_paths
        client_paths_dir = Path("data/client_paths")
        client_paths_dir.mkdir(exist_ok=True)
        
        output_path = client_paths_dir / f"connected_path_{session_id[:8]}.jpg"
        with open(output_path, 'wb') as f:
            f.write(viz_response.content)
        
        print(f"Path saved: {output_path}")
        print(f"Image size: {len(viz_response.content)} bytes")
        
        # Check server path too
        server_path = Path(f"data/server_paths/path_{session_id[:8]}.jpg")
        if server_path.exists():
            print(f"Server path: {server_path} ({server_path.stat().st_size} bytes)")
        
        # Verify it's a proper image
        import subprocess
        result = subprocess.run(["file", str(output_path)], capture_output=True, text=True)
        print(f"File type: {result.stdout.strip()}")
        
        return True
    else:
        print(f"Visualization failed: {viz_response.status_code}")
        return False

if __name__ == "__main__":
    test_path_with_lines()