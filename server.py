#!/usr/bin/env python3
"""
Satellite Image Embedding Server
===============================
AWS-compatible server for processing satellite imagery with TinyDINO embeddings.

API:
- init_map(lat, lng, meters=2000, mode="server") -> session_id or map_data
"""

import sys
import uuid
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import time
import math
import pickle
import os
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import threading

# Add src to path for GEE sampler
sys.path.append(str(Path(__file__).parent / "src"))
from gee_sampler import sample_satellite_image

@dataclass
class PatchData:
    """Data for a single image patch with embedding and GPS coordinates."""
    embedding: np.ndarray
    lat: float
    lng: float
    patch_coords: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in image coordinates

@dataclass
class SessionData:
    """Complete session data for a map region."""
    session_id: str
    full_map: np.ndarray  # Full stitched map as numpy array
    map_bounds: Dict[str, float]  # {"min_lat", "max_lat", "min_lng", "max_lng"}
    patch_size: int  # Size of each patch in pixels
    patches: List[PatchData]  # All patches with embeddings and GPS
    created_at: float
    meters_coverage: int

class TinyDINOEmbedder:
    """DINOv2 embedding model for satellite imagery."""
    
    def __init__(self, model_name: str = "dinov2_vits14"):
        """Initialize the DINOv2 embedding model."""
        try:
            import torch
            import torchvision.transforms as transforms
            import os
            
            os.environ["XFORMERS_DISABLED"] = "1"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
            self.model = self.model.to(self.device).eval()
            
            with torch.no_grad():
                dummy_output = self.model(torch.randn(1, 3, 224, 224).to(self.device))
                self.embedding_dim = dummy_output.shape[1]
            
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.is_loaded = True
            print(f"DINOv2 loaded: {model_name}, dim={self.embedding_dim}, device={self.device}")
            
        except Exception as e:
            print(f"DINOv2 failed: {e}. Using random embeddings.")
            self.model = None
            self.embedding_dim = 384
            self.is_loaded = False
    
    def embed_patch(self, patch: np.ndarray) -> np.ndarray:
        """Generate embedding for image patch using DINOv2."""
        if not self.is_loaded or self.model is None:
            return np.random.normal(0, 1, self.embedding_dim).astype(np.float32)
        
        try:
            import torch
            from PIL import Image as PILImage
            
            if patch.dtype != np.uint8:
                patch = (patch * 255).astype(np.uint8)
            
            input_tensor = self.transform(PILImage.fromarray(patch)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(input_tensor).cpu().numpy().flatten()
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"Embedding error: {e}")
            return np.random.normal(0, 1, self.embedding_dim).astype(np.float32)

class SatelliteEmbeddingServer:
    """Main server class for satellite image processing."""
    
    def __init__(self, storage_file: str = "data/sessions.pkl"):
        """Initialize the server with persistent session storage."""
        self.storage_file = storage_file
        self.sessions: Dict[str, SessionData] = {}
        self.embedder = TinyDINOEmbedder()
        
        # Create storage directory if it doesn't exist
        os.makedirs(os.path.dirname(storage_file), exist_ok=True)
        
        # Load existing sessions
        self._load_sessions()
        print(f"Satellite Embedding Server initialized with {len(self.sessions)} existing sessions")
    
    def _load_sessions(self):
        """Load sessions from persistent storage."""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'rb') as f:
                    self.sessions = pickle.load(f)
                print(f"Loaded {len(self.sessions)} sessions")
            else:
                print("Starting fresh")
        except Exception as e:
            print(f"Error loading sessions: {e}, starting fresh")
            self.sessions = {}
    
    def _save_sessions(self):
        """Save sessions to persistent storage."""
        try:
            with open(self.storage_file, 'wb') as f:
                pickle.dump(self.sessions, f)
            print(f"Saved {len(self.sessions)} sessions")
        except Exception as e:
            print(f"Save error: {e}")
    
    def _calculate_grid_size_for_meters(self, meters: int) -> Tuple[int, int]:
        """Calculate grid size to get good patch resolution."""
        # Target: ~1-2 meters per pixel for good resolution
        # Each GEE patch is 256px at ~10m/pixel = 2560m coverage
        # For 1000m desired, use 2x2 grid (5120m) for better resolution
        if meters <= 1000:
            return (2, 2)  # 5120m coverage, better resolution
        elif meters <= 2000:
            return (3, 3)  # 7680m coverage
        else:
            return (4, 4)  # 10240m coverage
    
    def _calculate_patch_gps(self, patch_center_px: Tuple[int, int], 
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
    
    def _extract_patches(self, image: np.ndarray, patch_size: int = 100) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
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
        
        # Always extract multiple patches, even from small images
        # Calculate stride to get good coverage
        if h < 200 or w < 200:
            # For small images, use overlapping patches
            stride = patch_size // 2
        else:
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
    
    def _crop_to_meters(self, image: np.ndarray, target_meters: int, 
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
    
    def init_map(self, lat: float, lng: float, meters: int = 2000, 
                mode: str = "server") -> Dict[str, Any]:
        """
        Initialize a new map session with embeddings.
        
        Args:
            lat: Latitude of center point
            lng: Longitude of center point  
            meters: Desired coverage in meters (default 2km)
            mode: "server" (return success) or "device" (return full data)
            
        Returns:
            Dictionary with session_id and optional map data
        """
        start_time = time.time()
        session_id = str(uuid.uuid4())
        
        print(f"Initializing map session {session_id[:8]}...")
        print(f"Location: ({lat:.6f}, {lng:.6f})")
        print(f"Coverage: {meters}m x {meters}m")
        
        try:
            # Calculate grid size for GEE sampler
            grid_rows, grid_cols = self._calculate_grid_size_for_meters(meters)
            print(f"Using {grid_rows}x{grid_cols} grid for {meters}m coverage")
            
            # Fetch satellite image using GEE sampler
            print("Fetching satellite imagery...")
            image_path = sample_satellite_image(
                lat=lat,
                lng=lng,
                grid_size=(grid_rows, grid_cols),
                patch_pixels=(256, 256)  # High resolution patches
            )
            
            # Load and convert to numpy array
            pil_image = Image.open(image_path)
            full_image = np.array(pil_image)
            
            # Calculate actual coverage achieved  
            # Sentinel-2 official resolution: 10m per pixel for RGB bands
            SENTINEL2_M_PER_PIXEL = 10.0
            original_coverage_m = grid_rows * 256 * SENTINEL2_M_PER_PIXEL
            print(f"Original coverage: {original_coverage_m}m")
            
            # Crop to exact desired meters
            if meters < original_coverage_m:
                print(f"Cropping to {meters}m...")
                full_image = self._crop_to_meters(full_image, meters, original_coverage_m)
            
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
            patch_data = self._extract_patches(full_image, patch_size=patch_size_px)
            print(f"Extracted {len(patch_data)} patches")
            
            # Generate embeddings for each patch
            print("Generating embeddings...")
            patches = []
            for i, (patch_array, coords) in enumerate(patch_data):
                # Generate embedding
                embedding = self.embedder.embed_patch(patch_array)
                
                # Calculate GPS coordinates for patch center
                patch_center_x = (coords[0] + coords[2]) // 2
                patch_center_y = (coords[1] + coords[3]) // 2
                patch_lat, patch_lng = self._calculate_patch_gps(
                    (patch_center_x, patch_center_y),
                    full_image.shape[1::-1],  # (width, height)
                    map_bounds
                )
                
                patches.append(PatchData(
                    embedding=embedding,
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
            self.sessions[session_id] = session_data
            
            # Save sessions to persistent storage
            self._save_sessions()
            
            elapsed_time = time.time() - start_time
            print(f"Session {session_id[:8]} created successfully in {elapsed_time:.1f}s")
            print(f"Map size: {full_image.shape}")
            print(f"Patches: {len(patches)}")
            print(f"Coverage: {meters}m x {meters}m")
            
            # Return based on mode
            if mode == "device":
                return {
                    "session_id": session_id,
                    "success": True,
                    "map_data": {
                        "full_map": full_image.tolist(),  # Convert to serializable format
                        "map_bounds": map_bounds,
                        "patches": [
                            {
                                "embedding": patch.embedding.tolist(),
                                "lat": patch.lat,
                                "lng": patch.lng,
                                "coords": patch.patch_coords
                            }
                            for patch in patches
                        ],
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
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data by ID."""
        return self.sessions.get(session_id)
    
    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self.sessions.keys())
    
    def cleanup_session(self, session_id: str) -> bool:
        """Remove a session from memory and persistent storage."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self._save_sessions()
            print(f"Session {session_id[:8]} cleaned up")
            return True
        return False

# Global server instance
server = SatelliteEmbeddingServer()

def init_map(lat: float, lng: float, meters: int = 2000, mode: str = "server") -> Dict[str, Any]:
    """API endpoint for initializing a map session."""
    return server.init_map(lat, lng, meters, mode)

# Pydantic models for request/response
class InitMapRequest(BaseModel):
    lat: float
    lng: float
    meters: int = 2000
    mode: str = "server"

class HealthResponse(BaseModel):
    status: str
    sessions_count: int
    server: str

class SessionInfo(BaseModel):
    session_id: str
    created_at: float
    meters_coverage: int
    patch_count: int
    map_bounds: dict

class SessionsResponse(BaseModel):
    success: bool
    sessions: list[SessionInfo]
    count: int

# FastAPI HTTP Server
app = FastAPI(
    title="Satellite Embedding Server",
    description="AWS-compatible server for processing satellite imagery with TinyDINO embeddings",
    version="1.0.0"
)

@app.post("/init_map")
async def http_init_map(request: InitMapRequest):
    """HTTP endpoint for initializing a map session."""
    try:
        result = server.init_map(request.lat, request.lng, request.meters, request.mode)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "success": False,
            "error": str(e)
        })

@app.get("/sessions", response_model=SessionsResponse)
async def http_list_sessions():
    """HTTP endpoint for listing all sessions."""
    try:
        sessions = server.list_sessions()
        session_data = []
        
        for session_id in sessions:
            session = server.get_session(session_id)
            if session:
                session_data.append(SessionInfo(
                    session_id=session_id,
                    created_at=session.created_at,
                    meters_coverage=session.meters_coverage,
                    patch_count=len(session.patches),
                    map_bounds=session.map_bounds
                ))
        
        return SessionsResponse(
            success=True,
            sessions=session_data,
            count=len(sessions)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e)
        })

@app.get("/health", response_model=HealthResponse)
async def http_health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        sessions_count=len(server.list_sessions()),
        server="Satellite Embedding Server"
    )

def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the HTTP server."""
    print(f"Server starting on {host}:{port}")
    if debug:
        print(f"Docs: http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port, log_level="debug" if debug else "info", access_log=debug)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Satellite Embedding Server')
    parser.add_argument('--port', type=int, default=5000, help='Port to run server on (default: 5000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port, debug=args.debug)
