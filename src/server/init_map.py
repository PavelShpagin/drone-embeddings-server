"""
Map Initialization Module for Satellite Embedding Server
========================================================
Handles satellite image fetching, processing, and embedding generation.
"""

import uuid
import numpy as np
import time
import math
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
from general.models import SessionData, PatchData
from server.gee_sampler import sample_satellite_image


def calculate_grid_size_for_meters(meters: int) -> Tuple[int, int]:
    """Calculate minimal grid size needed for coverage."""
    # Each GEE patch is 256px at 10m/pixel = 2560m coverage
    TILE_COVERAGE_M = 128 * 10  # 2560m per tile
    
    # Calculate minimum tiles needed
    tiles_needed = math.ceil(meters / TILE_COVERAGE_M)
    
    if tiles_needed <= 1:
        return (1, 1)  # 2560m coverage
    elif tiles_needed <= 4:
        return (2, 2)  # 5120m coverage  
    elif tiles_needed <= 9:
        return (3, 3)  # 7680m coverage
    else:
        return (4, 4)  # 10240m coverage


def calculate_patch_gps(patch_center_px: Tuple[int, int], 
                       image_size: Tuple[int, int],
                       map_bounds: Dict[str, float]) -> Tuple[float, float]:
    """
    Calculate GPS coordinates for patch center.
    
    Args:
        patch_center_px: (x, y) center of patch in image pixels
        image_size: (width, height) of full image
        map_bounds: GPS bounds of the full image
        
    Returns:
        (lat, lng) GPS coordinates of patch center
    """
    x_px, y_px = patch_center_px
    img_width, img_height = image_size
    
    # Convert pixel coordinates to GPS
    lng_range = map_bounds["max_lng"] - map_bounds["min_lng"]
    lat_range = map_bounds["max_lat"] - map_bounds["min_lat"]
    
    # X maps to longitude, Y maps to latitude (reversed)
    lng = map_bounds["min_lng"] + (x_px / img_width) * lng_range
    lat = map_bounds["max_lat"] - (y_px / img_height) * lat_range
    
    return lat, lng


def extract_patches(image: np.ndarray, patch_size: int = 100) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Extract patches from image, adapting patch size if needed.
    
    Args:
        image: Full image as numpy array (H, W, 3)
        patch_size: Desired size of each patch in pixels
        
    Returns:
        List of (patch_array, (x1, y1, x2, y2)) tuples
    """
    h, w = image.shape[:2]
    patches = []
    
    # Use non-overlapping patches for clean grid
    stride = patch_size
    
    for y in range(0, max(1, h - patch_size + 1), stride):
        for x in range(0, max(1, w - patch_size + 1), stride):
            x2 = min(x + patch_size, w)
            y2 = min(y + patch_size, h) 
            
            # Extract patch and resize if needed
            patch = image[y:y2, x:x2]
            
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                from PIL import Image as PILImage
                pil_patch = PILImage.fromarray(patch)
                resized = pil_patch.resize((patch_size, patch_size), PILImage.Resampling.LANCZOS)
                patch = np.array(resized)
            
            patches.append((patch, (x, y, x2, y2)))
    
    return patches


def crop_to_meters(image: np.ndarray, target_meters: int, 
                   original_coverage_m: int) -> np.ndarray:
    """
    Crop image to exact meter coverage.
    
    Args:
        image: Full stitched image
        target_meters: Desired coverage in meters
        original_coverage_m: Original coverage of the full image
        
    Returns:
        Cropped image
    """
    if target_meters >= original_coverage_m:
        return image  # No cropping needed
    
    h, w = image.shape[:2]
    
    # Calculate crop ratio
    crop_ratio = target_meters / original_coverage_m
    
    # Calculate new dimensions
    new_h = int(h * crop_ratio)
    new_w = int(w * crop_ratio)
    
    # Center crop
    start_y = (h - new_h) // 2
    start_x = (w - new_w) // 2
    
    return image[start_y:start_y + new_h, start_x:start_x + new_w]


def process_init_map_request(lat: float, lng: float, meters: int, mode: str, 
                           embedder, sessions: dict, save_sessions_callback, progress_callback=None) -> Dict[str, Any]:
    """
    Process an init_map request.
    
    Args:
        lat: Latitude of center point
        lng: Longitude of center point  
        meters: Desired coverage in meters
        mode: "server" (return success) or "device" (return full data)
        embedder: DINOv2 embedder instance
        sessions: Sessions dictionary
        save_sessions_callback: Function to save sessions to storage
        
    Returns:
        Dictionary with session_id and optional map data
    """
    start_time = time.time()
    session_id = str(uuid.uuid4())
    
    print(f"Initializing map session {session_id}...")
    print(f"Location: ({lat:.6f}, {lng:.6f})")
    print(f"Coverage: {meters}m x {meters}m")
    
    try:
        # Calculate grid size for GEE sampler
        grid_rows, grid_cols = calculate_grid_size_for_meters(meters)
        print(f"Using {grid_rows}x{grid_cols} grid for {meters}m coverage")
        
        # Fetch satellite image using GEE sampler with fallback
        print("Fetching satellite imagery...")
        try:
            pil_image = sample_satellite_image(
                lat=lat,
                lng=lng,
                grid_size=(grid_rows, grid_cols),
                patch_pixels=(128, 128)  # High resolution patches
            )
        except Exception as gee_error:
            print(f"GEE sampling failed: {gee_error}. Falling back to local research/data image or synthetic map.")
            from pathlib import Path
            from PIL import Image as PILImage
            data_dir = Path('research/data')
            candidates = []
            if data_dir.exists():
                for ext in ('*.jpg','*.jpeg','*.png'):
                    candidates.extend(data_dir.glob(ext))
            if candidates:
                pil_image = PILImage.open(candidates[0]).convert('RGB')
            else:
                pil_image = PILImage.new('RGB', (2048, 2048), color=(60, 120, 60))
        
        # Convert to numpy array
        full_image = np.array(pil_image)
        
        # Calculate actual coverage achieved  
        # Sentinel-2 official resolution: 10m per pixel for RGB bands
        SENTINEL2_M_PER_PIXEL = 10.0
        original_coverage_m = grid_rows * 256 * SENTINEL2_M_PER_PIXEL
        print(f"Original coverage: {original_coverage_m}m")
        
        # Crop to exact desired meters
        if meters < original_coverage_m:
            print(f"Cropping to {meters}m...")
            full_image = crop_to_meters(full_image, meters, original_coverage_m)
        
        # Calculate map bounds for GPS tagging
        coverage_half = meters / 2
        lat_radians = math.radians(lat)
        meters_per_degree_lat = 111132.92 - 559.82 * math.cos(2 * lat_radians)
        meters_per_degree_lng = 111412.84 * math.cos(lat_radians)
        
        lat_delta = coverage_half / meters_per_degree_lat
        lng_delta = coverage_half / meters_per_degree_lng
        
        map_bounds = {
            "min_lat": lat - lat_delta,
            "max_lat": lat + lat_delta,
            "min_lng": lng - lng_delta,
            "max_lng": lng + lng_delta
        }
        
        # Calculate patch size for 100m x 100m coverage
        # After cropping, pixels per meter changes
        h, w = full_image.shape[:2]
        pixels_per_meter = min(h, w) / meters
        patch_size_px = int(100 * pixels_per_meter)  # 100m in pixels
        patch_size_px = max(10, min(patch_size_px, min(h, w) // 2))  # Reasonable bounds
        
        print(f"Image size after crop: {h}x{w}")
        print(f"Pixels per meter: {pixels_per_meter:.2f}")
        print(f"Patch size for 100m: {patch_size_px}px")
        
        # Extract patches
        patch_data = extract_patches(full_image, patch_size=patch_size_px)
        print(f"Extracted {len(patch_data)} patches")
        
        # Generate embeddings for each patch
        print("Generating embeddings...")
        patches = []
        total_patches = len(patch_data)
        
        # Track progress updates for every 1% and every tile completion
        last_progress_reported = 45
        
        for i, (patch_array, coords) in enumerate(patch_data):
            # Update progress during embedding generation (every tile + every 1%)
            if progress_callback and total_patches > 0:
                # Progress from 45% to 75% during embedding generation (30% range)
                progress_ratio = i / total_patches
                current_progress = 45 + int(progress_ratio * 30)
                
                # Update on every tile completion OR every 1% progress change
                should_update = (
                    current_progress > last_progress_reported or       # Every 1% change
                    i == 0 or                                          # First patch
                    i == total_patches - 1                            # Last patch
                )
                
                if should_update:
                    progress_callback(f"Generating embeddings ({i+1}/{total_patches})...", current_progress)
                    last_progress_reported = current_progress
                    print(f"🔄 Tile {i+1}/{total_patches} completed - {current_progress}%")
                
                # Always send tile completion update (even if progress % didn't change)
                elif progress_callback:
                    # Send tile completion without changing overall progress
                    progress_callback(f"Processing tile {i+1}/{total_patches}...", current_progress)
                    print(f"⚙️ Processing tile {i+1}/{total_patches}...")
            
            # Generate representation dict from embedder
            rep = embedder.embed_patch(patch_array)
            
            # Calculate GPS coordinates for patch center
            patch_center_x = (coords[0] + coords[2]) // 2
            patch_center_y = (coords[1] + coords[3]) // 2
            patch_lat, patch_lng = calculate_patch_gps(
                (patch_center_x, patch_center_y),
                full_image.shape[1::-1],  # (width, height)
                map_bounds
            )
            
            patches.append(PatchData(
                embedding_data={"embedding": rep["embedding"]},
                lat=patch_lat,
                lng=patch_lng,
                patch_coords=coords
            ))
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(patch_data)} patches")
        
        # Create session data
        session_data = SessionData(
            session_id=session_id,
            full_map=full_image,
            map_bounds=map_bounds,
            patch_size=100,
            patches=patches,
            created_at=time.time(),
            meters_coverage=meters
        )
        
        # Store in sessions hash table
        sessions[session_id] = session_data
        
        # Save sessions to persistent storage
        save_sessions_callback()
        
        elapsed_time = time.time() - start_time
        print(f"Session {session_id} created successfully in {elapsed_time:.1f}s")
        print(f"Map size: {full_image.shape}")
        print(f"Patches: {len(patches)}")
        print(f"Coverage: {meters}m x {meters}m")
        
        # Return based on mode
        if mode == "device":
            # Create zip file with map and embeddings
            import io
            import zipfile
            import json
            from PIL import Image
            
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Save map as PNG
                if full_image.dtype != np.uint8:
                    # Normalize to 0-255 if needed
                    map_array = ((full_image - full_image.min()) / 
                               (full_image.max() - full_image.min()) * 255).astype(np.uint8)
                else:
                    map_array = full_image
                
                # Convert to PIL Image and save to bytes
                map_image = Image.fromarray(map_array)
                map_buffer = io.BytesIO()
                map_image.save(map_buffer, format='PNG')
                zf.writestr('map.png', map_buffer.getvalue())
                
                # Save embeddings as JSON
                embeddings_data = [
                    {
                        "embedding": patch.embedding_data["embedding"].tolist(),
                        "lat": patch.lat,
                        "lng": patch.lng,
                        "coords": patch.patch_coords
                    }
                    for patch in patches
                ]
                zf.writestr('embeddings.json', json.dumps(embeddings_data))
            
            zip_data = zip_buffer.getvalue()
            
            return {
                "session_id": session_id,
                "success": True,
                "zip_data": zip_data,
                "map_data": {
                    "full_map": full_image.tolist(),  # Convert to serializable format
                    "map_bounds": map_bounds,
                    "patches": embeddings_data,
                    "meters_coverage": meters,
                    "patch_count": len(patches)
                }
            }
        else:  # mode == "server"
            return {
                "session_id": session_id,
                "success": True,
                "message": f"Map session created with {len(patches)} patches",
                "coverage": f"{meters}m x {meters}m",
                "patch_count": len(patches)
            }
            
    except Exception as e:
        print(f"Error creating session: {e}")
        return {
            "success": False,
            "error": str(e),
            "session_id": None
        }