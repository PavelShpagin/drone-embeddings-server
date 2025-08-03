"""
Core Server Class for Satellite Embedding Server
================================================
Main server logic including session management and persistent storage.
"""

import os
import pickle
from typing import Dict, List, Optional
import time
from models import SessionData, PathPoint
from embedder import TinyDINOEmbedder
from init_map import process_init_map_request
from fetch_gps import process_fetch_gps_request
from image_metadata import extract_metadata
import time


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
            # Remove the incompatible pickle file (sessions are not critical)
            if os.path.exists(self.storage_file):
                backup_file = self.storage_file + '.backup'
                try:
                    os.rename(self.storage_file, backup_file)
                    print(f"Moved incompatible sessions to {backup_file}")
                except:
                    os.remove(self.storage_file)
                    print("Removed incompatible sessions file")
            self.sessions = {}
    
    def _save_sessions(self):
        """Save sessions to persistent storage."""
        try:
            with open(self.storage_file, 'wb') as f:
                pickle.dump(self.sessions, f)
            print(f"Saved {len(self.sessions)} sessions")
        except Exception as e:
            print(f"Save error: {e}")
    
    def init_map(self, lat: float, lng: float, meters: int = 2000, mode: str = "server"):
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
        return process_init_map_request(
            lat=lat, 
            lng=lng, 
            meters=meters, 
            mode=mode,
            embedder=self.embedder,
            sessions=self.sessions,
            save_sessions_callback=self._save_sessions
        )
    
    def fetch_gps(self, image_data: bytes, session_id: str):
        """
        Find GPS coordinates for an image by matching against session embeddings.
        
        Args:
            image_data: Raw image bytes
            session_id: Session ID to search in
            
        Returns:
            Dictionary with GPS coordinates and similarity info
        """
        result = process_fetch_gps_request(image_data, session_id, self.embedder, self.sessions)
        
        # Add to path visualization if successful
        if result.get("success") and "gps" in result:
            gps = result["gps"]
            self.add_path_point(session_id, gps["lat"], gps["lng"], image_data)
        
        return result
    
    def add_path_point(self, session_id: str, lat: float, lng: float, 
                      image_data: Optional[bytes] = None, metadata: Optional[dict] = None):
        """
        Add a path point to a session's tracking data.
        
        Args:
            session_id: Session ID to add point to
            lat: Latitude coordinate
            lng: Longitude coordinate
            image_data: Optional image bytes
            metadata: Optional metadata dictionary
            
        Returns:
            Dictionary with success status
        """
        try:
            if session_id not in self.sessions:
                return {
                    "success": False,
                    "error": f"Session {session_id} not found"
                }
            
            session_data = self.sessions[session_id]
            
            # Extract metadata if image provided
            pos2D = None
            height = None
            full_metadata = metadata or {}
            
            if image_data and len(image_data) > 0:
                try:
                    # Extract embedding
                    import numpy as np
                    from PIL import Image
                    import io
                    
                    image = Image.open(io.BytesIO(image_data))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image_array = np.array(image)
                    embedding = self.embedder.embed_patch(image_array)
                    
                    # Extract metadata
                    # Save image temporarily for metadata extraction
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                        tmp_file.write(image_data)
                        tmp_path = tmp_file.name
                    
                    try:
                        img_metadata = extract_metadata(tmp_path)
                    finally:
                        import os
                        os.unlink(tmp_path)  # Clean up temp file
                    
                    if img_metadata and img_metadata.telemetry:
                        pos2D = img_metadata.telemetry.position_2d
                        height = img_metadata.telemetry.height
                        full_metadata.update({
                            'fms': img_metadata.telemetry.fms,
                            'height': height,
                            'tilt': img_metadata.telemetry.tilt,
                            'quaternion': img_metadata.telemetry.quaternion,
                            'acceleration': img_metadata.telemetry.acceleration
                        })
                    
                except Exception as e:
                    print(f"Error processing image metadata: {e}")
                    embedding = None
            else:
                embedding = None
            
            # Create path point
            path_point = PathPoint(
                lat=lat,
                lng=lng,
                timestamp=time.time(),
                pos2D=pos2D,
                height=height,
                image_data=image_data,
                embedding=embedding,
                metadata=full_metadata
            )
            
            # Add to session path data
            if session_data.path_data is None:
                session_data.path_data = []
            session_data.path_data.append(path_point)
            
            # Update path visualization incrementally
            session_data.path_image_file = self._update_path_visualization_internal(session_data, lat, lng)
            
            # Save sessions
            self._save_sessions()
            
            return {
                "success": True,
                "session_id": session_id,
                "path_points": len(session_data.path_data),
                "message": f"Added path point {len(session_data.path_data)}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data by ID."""
        return self.sessions.get(session_id)
    
    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self.sessions.keys())
    
    def _update_path_visualization_internal(self, session_data, new_lat: float, new_lng: float) -> str:
        """Update path visualization internally without import issues."""
        from pathlib import Path
        from PIL import Image, ImageDraw
        import numpy as np
        
        # Load existing path image or create from full map
        if session_data.path_image_file and Path(session_data.path_image_file).exists():
            viz_img = Image.open(session_data.path_image_file)
        else:
            viz_img = Image.fromarray(session_data.full_map.astype(np.uint8))
        
        draw = ImageDraw.Draw(viz_img)
        
        # Convert new GPS to pixel coordinates
        def gps_to_pixel_coords(lat, lng, image_size, map_bounds):
            img_width, img_height = image_size
            lat_range = map_bounds["max_lat"] - map_bounds["min_lat"]
            lng_range = map_bounds["max_lng"] - map_bounds["min_lng"]
            
            if lat_range == 0 or lng_range == 0:
                return img_width // 2, img_height // 2
            
            x = int((lng - map_bounds["min_lng"]) / lng_range * img_width)
            y = int((map_bounds["max_lat"] - lat) / lat_range * img_height)
            
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            return x, y
        
        new_x, new_y = gps_to_pixel_coords(new_lat, new_lng, viz_img.size, session_data.map_bounds)
        
        # Draw line from previous point if exists (need at least 2 points)
        if len(session_data.path_data) >= 2:
            prev_point = session_data.path_data[-2]  # Get the actual previous point
            prev_x, prev_y = gps_to_pixel_coords(prev_point.lat, prev_point.lng, viz_img.size, session_data.map_bounds)
            
            # Draw simple thin red line
            draw.line([(prev_x, prev_y), (new_x, new_y)], fill=(255, 0, 0), width=2)
        
        # Draw large red dot
        dot_size = max(5, min(viz_img.width, viz_img.height) // 100)
        draw.ellipse([new_x - dot_size, new_y - dot_size, new_x + dot_size, new_y + dot_size], 
                    fill=(255, 0, 0), outline=(255, 255, 255), width=2)
        
        # Save to server_paths
        server_paths_dir = Path("data/server_paths")
        server_paths_dir.mkdir(parents=True, exist_ok=True)
        
        image_path = server_paths_dir / f"path_{session_data.session_id[:8]}.jpg"
        viz_img.save(image_path, 'JPEG', quality=95)
        
        return str(image_path)

    def cleanup_session(self, session_id: str) -> bool:
        """Remove a session from memory and persistent storage."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self._save_sessions()
            print(f"Session {session_id[:8]} cleaned up")
            return True
        return False