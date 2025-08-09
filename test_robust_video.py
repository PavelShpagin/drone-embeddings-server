#!/usr/bin/env python3
"""Test the robust video generation approach."""

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

def test_robust_video():
    """Test robust video generation."""
    print("Testing robust video generation...")
    
    # Create test session
    session_id = "robust12"
    server_paths_dir = Path("data/server_paths")
    server_paths_dir.mkdir(exist_ok=True)
    
    # Clean up any existing video
    video_path = server_paths_dir / f"path_video_{session_id[:8]}.avi"
    if video_path.exists():
        video_path.unlink()
    
    try:
        # Generate frames that should trigger video updates
        for i in range(12):  # More than 10 to trigger multiple video generations
            print(f"Adding frame {i+1}/12")
            test_img = create_test_image(i)
            _append_video_frame(test_img, session_id, server_paths_dir)
            
            # Check if video exists after every 5 frames
            if (i + 1) % 5 == 0:
                if video_path.exists():
                    size = video_path.stat().st_size
                    print(f"  Video updated: {size} bytes")
                else:
                    print("  Video not yet created")
        
        # Finalize video
        print("Finalizing video...")
        finalize_session_video(session_id)
        
        # Final check
        if video_path.exists():
            video_size = video_path.stat().st_size
            print(f"✅ Final video created: {video_path}")
            print(f"✅ Final video size: {video_size} bytes")
            
            # Try to verify video properties
            try:
                import cv2
                cap = cv2.VideoCapture(str(video_path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                print(f"✅ Video properties:")
                print(f"  - Frames: {frame_count}")
                print(f"  - FPS: {fps}")
                print(f"  - Resolution: {width}x{height}")
                
                # Try to read a frame to verify it's not corrupted
                ret, frame = cap.read()
                if ret and frame is not None:
                    print("✅ Video is readable and not corrupted!")
                else:
                    print("❌ Could not read frames - video may be corrupted")
                
                cap.release()
                
            except Exception as e:
                print(f"⚠️ Could not verify video: {e}")
                
        else:
            print(f"❌ Video not created: {video_path}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_robust_video()


