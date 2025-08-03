#!/usr/bin/env python3

import requests
import time
from pathlib import Path

def test_incremental_path_building():
    server_url = "http://localhost:5000"
    
    print("Testing incremental path building with red lines...")
    
    # Create new session
    init_response = requests.post(f"{server_url}/init_map", json={
        "lat": 50.4162,
        "lng": 30.8906,
        "meters": 1000,
        "mode": "device"
    })
    
    session_id = init_response.json()["session_id"]
    print(f"Session: {session_id[:8]}")
    
    # Images to create path
    images = [
        "../device/data/stream/00000070.jpg",
        "../device/data/stream/00000134.jpg", 
        "../device/data/stream/00000198.jpg"
    ]
    
    print("\nBuilding path incrementally:")
    for i, image_path in enumerate(images, 1):
        if Path(image_path).exists():
            print(f"\nAdding point {i}...")
            
            # Add GPS point
            with open(image_path, "rb") as f:
                files = {"image": f}
                data = {"session_id": session_id}
                response = requests.post(f"{server_url}/fetch_gps", files=files, data=data)
                
                if response.json().get("success"):
                    gps = response.json()["gps"]
                    print(f"  GPS: {gps['lat']:.6f}, {gps['lng']:.6f}")
                    
                    # Save intermediate path visualization
                    viz_response = requests.post(f"{server_url}/visualize_path", 
                                               json={"session_id": session_id})
                    
                    if viz_response.status_code == 200:
                        output_path = Path(f"data/client_paths/step_{i}_{session_id[:8]}.jpg")
                        with open(output_path, 'wb') as f:
                            f.write(viz_response.content)
                        print(f"  Saved: {output_path} ({len(viz_response.content)} bytes)")
                        
                        if i == 1:
                            print("    - Single red dot on map")
                        else:
                            print(f"    - {i} red dots connected with red lines")
    
    print(f"\nPath visualization complete!")
    print("Files created:")
    for step_file in sorted(Path("data/client_paths").glob(f"step_*_{session_id[:8]}.jpg")):
        print(f"  {step_file}")

if __name__ == "__main__":
    test_incremental_path_building()