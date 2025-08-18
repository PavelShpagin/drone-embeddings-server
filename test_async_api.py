#!/usr/bin/env python3
"""
Test script for the new async progress tracking API
"""

import requests
import time
import json

def test_async_init_map(server_url="http://localhost:5000"):
    """Test the new device_async mode and progress polling."""
    
    print("🧪 Testing Async Progress Tracking API")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Health check...")
    try:
        response = requests.get(f"{server_url}/health")
        print(f"   ✅ Health: {response.json()}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return
    
    # Test 2: Start async init_map
    print("\n2. Starting async init_map...")
    try:
        response = requests.post(f"{server_url}/init_map", json={
            "lat": 50.4162,
            "lng": 30.8906,
            "meters": 1000,
            "mode": "device_async"
        })
        
        if response.status_code == 200:
            result = response.json()
            task_id = result.get("task_id")
            print(f"   ✅ Got task_id: {task_id}")
            
            # Test 3: Poll for progress
            print("\n3. Polling for progress...")
            max_polls = 60  # 1 minute max
            for i in range(max_polls):
                time.sleep(1)
                
                try:
                    progress_response = requests.get(f"{server_url}/progress/{task_id}")
                    if progress_response.status_code == 200:
                        progress_data = progress_response.json()
                        status = progress_data["status"]
                        progress = progress_data["progress"]
                        message = progress_data["message"]
                        
                        print(f"   📊 Poll {i+1}: {status} - {progress}% - {message}")
                        
                        if status == "completed":
                            print(f"   ✅ Task completed!")
                            if progress_data.get("zip_data"):
                                print(f"   📦 Got zip data (size: {len(progress_data['zip_data'])} chars)")
                            if progress_data.get("session_id"):
                                print(f"   🆔 Session ID: {progress_data['session_id']}")
                            break
                        elif status == "failed":
                            print(f"   ❌ Task failed: {progress_data.get('error', 'Unknown error')}")
                            break
                    else:
                        print(f"   ❌ Progress request failed: {progress_response.status_code}")
                        break
                        
                except Exception as e:
                    print(f"   ❌ Progress polling error: {e}")
                    break
            else:
                print("   ⏰ Polling timeout reached")
                
        else:
            print(f"   ❌ Init request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    
    print("\n🎯 Test completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test async API')
    parser.add_argument('--server', default='http://localhost:5000', help='Server URL')
    
    args = parser.parse_args()
    
    test_async_init_map(args.server)
