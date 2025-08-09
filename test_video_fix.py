#!/usr/bin/env python3
"""Test the fixed video generation."""

import sys
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# Add server src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "general"))

from general.visualize_map import _append_video_frame, finalize_session_video

def create_test_image(frame_num: int, size=(400, 300)) -> Image.Image:
    """Create a test image with frame number."""
    img = Image.new('RGB', size, color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw frame number
    draw.text((10, 10), f"Frame {frame_num}", fill='black')
    
    # Draw a moving red dot
    x = 50 + (frame_num * 10) % (size[0] - 100)
    y = 50 + (frame_num * 5) % (size[1] - 100)
    draw.ellipse([x-10, y-10, x+10, y+10], fill='red')
    
    return img

def test_video_generation():
    """Test video generation with multiple frames."""
    print("Testing fixed video generation...")
    
    # Create test session
    session_id = "test1234"
    server_paths_dir = Path("data/server_paths")
    server_paths_dir.mkdir(exist_ok=True)
    
    try:
        # Generate multiple frames
        for i in range(10):
            print(f"Adding frame {i+1}/10")
            test_img = create_test_image(i)
            _append_video_frame(test_img, session_id, server_paths_dir)
            time.sleep(0.1)  # Small delay to simulate real processing
        
        # Finalize video
        print("Finalizing video...")
        finalize_session_video(session_id)
        
        # Check result
        video_path = server_paths_dir / f"path_video_{session_id[:8]}.avi"
        if video_path.exists():
            video_size = video_path.stat().st_size
            print(f"✅ Video created successfully: {video_path}")
            print(f"✅ Video size: {video_size} bytes")
            
            # Try to verify video is readable
            try:
                import cv2
                cap = cv2.VideoCapture(str(video_path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                print(f"✅ Video properties: {frame_count} frames, {fps} FPS")
                
                if frame_count > 0:
                    print("✅ Video appears to be valid and readable!")
                else:
                    print("❌ Video has 0 frames - may be corrupted")
                    
            except Exception as e:
                print(f"⚠️ Could not verify video properties: {e}")
        else:
            print(f"❌ Video file not created: {video_path}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_video_generation()


