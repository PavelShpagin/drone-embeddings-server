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


def update_path_visualization(session_data, new_lat: float, new_lng: float) -> str:
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
    
    # Draw line from previous point if exists
    if len(session_data.path_data) > 0:
        prev_point = session_data.path_data[-1]
        prev_x, prev_y = gps_to_pixel_coords(prev_point.lat, prev_point.lng, viz_img.size, session_data.map_bounds)
        
        # Draw thin pure red line
        draw.line([(prev_x, prev_y), (new_x, new_y)], fill=(255, 0, 0), width=2)
    
    # Draw large red dot (5x5 pixels minimum)
    dot_size = max(5, min(viz_img.width, viz_img.height) // 100)
    draw.ellipse([new_x - dot_size, new_y - dot_size, new_x + dot_size, new_y + dot_size], 
                fill=(255, 0, 0), outline=(255, 255, 255), width=2)
    
    # Save to server_paths
    server_paths_dir = Path("data/server_paths")
    server_paths_dir.mkdir(parents=True, exist_ok=True)
    
    image_path = server_paths_dir / f"path_{session_data.session_id[:8]}.jpg"
    viz_img.save(image_path, 'JPEG', quality=95)
    
    return str(image_path)


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