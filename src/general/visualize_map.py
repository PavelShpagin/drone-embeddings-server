"""
Map Path Visualization Module
============================
Creates and updates map visualizations with GPS tracking paths.
"""

import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from typing import List, Tuple, Optional
import io


def create_path_visualization(session_data, gps_points: List[Tuple[float, float]], 
                            output_path: Optional[str] = None) -> bytes:
    """
    Create or update a map visualization with GPS tracking path.
    
    Args:
        session_data: SessionData object with full_map and map_bounds
        gps_points: List of (lat, lng) coordinates to plot
        output_path: Optional path to save the image file
        
    Returns:
        Image bytes of the visualization
    """
    if not session_data or not session_data.full_map.size:
        raise ValueError("Invalid session data")
    
    # Convert numpy array to PIL Image
    if isinstance(session_data.full_map, np.ndarray):
        if session_data.full_map.dtype != np.uint8:
            full_map_img = Image.fromarray((session_data.full_map * 255).astype(np.uint8))
        else:
            full_map_img = Image.fromarray(session_data.full_map)
    else:
        full_map_img = session_data.full_map
    
    # Create a copy for visualization
    viz_img = full_map_img.copy().convert("RGB")
    draw = ImageDraw.Draw(viz_img)
    
    if not gps_points:
        return image_to_bytes(viz_img, output_path)
    
    # Convert GPS coordinates to pixel coordinates
    pixel_points = []
    for lat, lng in gps_points:
        pixel_x, pixel_y = gps_to_pixel_coords(
            lat, lng, viz_img.size, session_data.map_bounds
        )
        pixel_points.append((pixel_x, pixel_y))
    
    # Draw path lines between consecutive points (BOLD RED)
    if len(pixel_points) > 1:
        for i in range(len(pixel_points) - 1):
            # Draw thick red line - multiple passes for boldness
            for offset in range(-3, 4):
                for offset2 in range(-3, 4):
                    x1, y1 = pixel_points[i]
                    x2, y2 = pixel_points[i + 1]
                    draw.line([(x1 + offset, y1 + offset2), (x2 + offset, y2 + offset2)], 
                             fill=(255, 0, 0), width=3)  # BOLD red line
    
    # Draw LARGE red dots for each GPS point
    dot_radius = max(8, min(viz_img.width, viz_img.height) // 50)  # Much larger dots
    for i, (x, y) in enumerate(pixel_points):
        # Make the most recent point even larger and brighter
        if i == len(pixel_points) - 1:
            radius = dot_radius * 3  # 3x larger for latest point
            color = (255, 0, 0)  # Pure bright red
        else:
            radius = dot_radius * 2  # 2x larger for other points
            color = (200, 0, 0)  # Dark red
        
        # Draw filled circle with white border for visibility
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                    fill=color, outline=(255, 255, 255), width=3)
    
    return image_to_bytes(viz_img, output_path)


def gps_to_pixel_coords(lat: float, lng: float, image_size: Tuple[int, int], 
                       map_bounds: dict) -> Tuple[int, int]:
    """
    Convert GPS coordinates to pixel coordinates on the map.
    
    Args:
        lat: Latitude
        lng: Longitude  
        image_size: (width, height) of the image
        map_bounds: Dictionary with min/max lat/lng bounds
        
    Returns:
        (x, y) pixel coordinates
    """
    img_width, img_height = image_size
    
    # Normalize GPS coordinates to 0-1 range
    lat_range = map_bounds["max_lat"] - map_bounds["min_lat"]
    lng_range = map_bounds["max_lng"] - map_bounds["min_lng"]
    
    if lat_range == 0 or lng_range == 0:
        return img_width // 2, img_height // 2
    
    # Convert to pixel coordinates
    x = int((lng - map_bounds["min_lng"]) / lng_range * img_width)
    y = int((map_bounds["max_lat"] - lat) / lat_range * img_height)  # Flip Y axis
    
    # Clamp to image bounds
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    
    return x, y


def image_to_bytes(image: Image.Image, output_path: Optional[str] = None) -> bytes:
    """
    Convert PIL Image to bytes and optionally save to file.
    
    Args:
        image: PIL Image object
        output_path: Optional path to save the image
        
    Returns:
        Image bytes
    """
    # Save to file if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format='JPEG', quality=90)
    
    # Convert to bytes
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='JPEG', quality=90)
    return img_buffer.getvalue()


def update_path_visualization(session_data, new_lat: float, new_lng: float, g_lat: Optional[float] = None, g_lon: Optional[float] = None) -> str:
    """
    Update path visualization with a new GPS point and save to server_paths.
    
    Args:
        session_data: SessionData object
        new_lat: New latitude point
        new_lng: New longitude point
        
    Returns:
        Path to saved image file
    """
    from pathlib import Path
    
    # Load existing path image or create from full map
    if session_data.path_image_file and Path(session_data.path_image_file).exists():
        viz_img = Image.open(session_data.path_image_file)
    else:
        viz_img = Image.fromarray(session_data.full_map.astype(np.uint8))
    
    draw = ImageDraw.Draw(viz_img)
    
    # Convert new GPS to pixel coordinates
    new_x, new_y = gps_to_pixel_coords(new_lat, new_lng, viz_img.size, session_data.map_bounds)
    
    # Draw line from previous predicted point if exists
    if len(session_data.path_data) > 0:
        prev_point = session_data.path_data[-1]
        prev_x, prev_y = gps_to_pixel_coords(prev_point.lat, prev_point.lng, viz_img.size, session_data.map_bounds)
        
        # Draw thin pure red line
        draw.line([(prev_x, prev_y), (new_x, new_y)], fill=(255, 0, 0), width=2)
    
    # Draw large red dot (predicted)
    dot_size = max(5, min(viz_img.width, viz_img.height) // 100)
    draw.ellipse([new_x - dot_size, new_y - dot_size, new_x + dot_size, new_y + dot_size], 
                fill=(255, 0, 0), outline=(255, 255, 255), width=2)

    # Draw ground-truth path if provided (g_lat/g_lon)
    if g_lat is not None and g_lon is not None:
        gt_x, gt_y = gps_to_pixel_coords(g_lat, g_lon, viz_img.size, session_data.map_bounds)
        # Line from previous GT point
        if hasattr(session_data, 'gt_path_data') and len(session_data.gt_path_data) > 0:
            prev_gt = session_data.gt_path_data[-1]
            prev_gt_x, prev_gt_y = gps_to_pixel_coords(prev_gt.lat, prev_gt.lng, viz_img.size, session_data.map_bounds)
            draw.line([(prev_gt_x, prev_gt_y), (gt_x, gt_y)], fill=(0, 200, 0), width=2)
        # GT dot
        draw.ellipse([gt_x - dot_size, gt_y - dot_size, gt_x + dot_size, gt_y + dot_size], 
                     fill=(0, 200, 0), outline=(255, 255, 255), width=2)
    
    # Save to server_paths
    server_paths_dir = Path("data/server_paths")
    server_paths_dir.mkdir(parents=True, exist_ok=True)
    
    image_path = server_paths_dir / f"path_{session_data.session_id[:8]}.jpg"
    viz_img.save(image_path, 'JPEG', quality=95)
    
    # Append frame to video for real-time time-lapse (if enabled)
    _append_video_frame(viz_img, session_data.session_id, server_paths_dir)
    
    return str(image_path)


# Global frame buffers for robust video generation
_frame_buffers = {}

def _append_video_frame(viz_img: Image.Image, session_id: str, server_paths_dir: Path) -> None:
    """Store frame for later video generation - more reliable approach."""
    try:
        import numpy as np
        
        session_key = session_id[:8]
        
        # Initialize frame buffer for session
        if session_key not in _frame_buffers:
            _frame_buffers[session_key] = []
        
        # Convert PIL to numpy array for storage
        img_array = np.array(viz_img)
        _frame_buffers[session_key].append(img_array)
        
        # Keep only last 100 frames to prevent memory issues
        if len(_frame_buffers[session_key]) > 100:
            _frame_buffers[session_key] = _frame_buffers[session_key][-100:]
        
        # Generate video every 5 frames for near real-time updates
        if len(_frame_buffers[session_key]) % 5 == 0:
            _generate_video_from_buffer(session_key, server_paths_dir)
                
    except Exception as e:
        # Don't let video frame saving break the main visualization
        print(f"Warning: Failed to buffer video frame: {e}")


def _generate_video_from_buffer(session_key: str, server_paths_dir: Path) -> None:
    """Generate video from frame buffer using most compatible method."""
    try:
        import cv2
        
        if session_key not in _frame_buffers or not _frame_buffers[session_key]:
            return
        
        frames = _frame_buffers[session_key]
        video_path = server_paths_dir / f"path_video_{session_key}.avi"
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Use most compatible codec and settings
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Motion JPEG - most reliable
        writer = cv2.VideoWriter(str(video_path), fourcc, 2.0, (width, height))
        
        if not writer.isOpened():
            print(f"Warning: Could not create video writer for session {session_key}")
            return
        
        # Write all frames
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        
        # Properly close writer
        writer.release()
        
        # Verify video was created successfully
        if video_path.exists() and video_path.stat().st_size > 1000:
            # Video created successfully
            pass
        else:
            print(f"Warning: Video generation failed for session {session_key}")
                
    except Exception as e:
        print(f"Warning: Failed to generate video from buffer: {e}")





def finalize_session_video(session_id: str) -> None:
    """Finalize video generation for a session."""
    session_key = session_id[:8]
    
    # Generate final video from all buffered frames
    if session_key in _frame_buffers:
        from pathlib import Path
        server_paths_dir = Path("data/server_paths")
        _generate_video_from_buffer(session_key, server_paths_dir)
        
        frame_count = len(_frame_buffers[session_key])
        print(f"Finalized video for session {session_key} with {frame_count} frames")
        
        # Clean up buffer
        del _frame_buffers[session_key]





def _save_individual_frame(viz_img: Image.Image, session_id: str, server_paths_dir: Path) -> None:
    """Fallback: Save individual frame (for when OpenCV not available)."""
    frames_dir = server_paths_dir / f"frames_{session_id[:8]}"
    frames_dir.mkdir(exist_ok=True)
    
    frame_files = list(frames_dir.glob("frame_*.jpg"))
    frame_number = len(frame_files)
    
    frame_path = frames_dir / f"frame_{frame_number:04d}.jpg"
    viz_img.save(frame_path, 'JPEG', quality=85)
    
    # Cleanup old frames (keep last 50 frames max)
    if frame_number > 50:
        old_frames = sorted(frame_files)[:-45]
        for old_frame in old_frames:
            old_frame.unlink(missing_ok=True)


def get_session_video_path(session_id: str, server_paths_dir: Path) -> str:
    """Get path to real-time generated video file."""
    video_path = server_paths_dir / f"path_video_{session_id[:8]}.avi"
    
    # Finalize video if still being written
    finalize_session_video(session_id)
    
    if not video_path.exists():
        raise ValueError("No video found for this session - ensure GPS processing has been run")
    
    return str(video_path)


def process_path_visualization_request(session_id: str, sessions: dict) -> dict:
    """
    Process a path visualization request for a session.
    Simply returns the stored path image.
    
    Args:
        session_id: Session ID to visualize
        sessions: Sessions dictionary
        
    Returns:
        Dictionary with visualization result
    """
    try:
        if session_id not in sessions:
            return {
                "success": False,
                "error": f"Session {session_id} not found"
            }
        
        session_data = sessions[session_id]
        
        if not session_data.path_image_file or not Path(session_data.path_image_file).exists():
            return {
                "success": False,
                "error": "No path visualization available for this session"
            }
        
        # Read image file and convert to bytes
        with open(session_data.path_image_file, 'rb') as f:
            image_bytes = f.read()
        
        return {
            "success": True,
            "image_bytes": image_bytes,
            "points_count": len(session_data.path_data)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }