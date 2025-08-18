"""
Core Server Class for Satellite Embedding Server
================================================
Main server logic including session management and persistent storage.
"""

import os
import pickle
import json
import zipfile
import io
from pathlib import Path
from typing import Dict, List, Optional
import time
import numpy as np
from PIL import Image
from general.models import SessionMetadata, SessionData, PathPoint
from server.embedder import TinyDINOEmbedder
from server.init_map import process_init_map_request
from general.fetch_gps import process_fetch_gps_request
from general.image_metadata import extract_metadata
import time


class SatelliteEmbeddingServer:
    """Main server class for satellite image processing."""
    
    def __init__(self, storage_file: str = "data/sessions.pkl"):
        """Initialize the server with persistent session storage."""
        self.storage_file = storage_file
        self.sessions: Dict[str, SessionMetadata] = {}
        self.embedder = TinyDINOEmbedder()
        
        # Create storage directories
        self.data_dir = Path("data")
        self.maps_dir = self.data_dir / "maps"
        self.embeddings_dir = self.data_dir / "embeddings"
        self.zips_dir = self.data_dir / "zips"
        self.logs_dir = self.data_dir / "logs"
        
        for dir_path in [self.data_dir, self.maps_dir, self.embeddings_dir, self.zips_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
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
    
    def save_sessions(self):
        """Public method to save sessions to persistent storage."""
        self._save_sessions()
    
    def init_map(self, lat: float, lng: float, meters: int = 2000, mode: str = "server", session_id: Optional[str] = None):
        """
        Initialize a new map session or return cached session.
        
        Args:
            lat: Latitude of center point
            lng: Longitude of center point  
            meters: Desired coverage in meters (default 2km)
            mode: "server" (return success) or "device" (return full data)
            session_id: Optional session ID for caching
            
        Returns:
            Dictionary with session_id and optional map data
        """
        # Check for cached session
        if session_id and session_id in self.sessions:
            return self._return_cached_session(session_id, mode)
        
        # Create new session
        return self._create_new_session(lat, lng, meters, mode)
    
    def _return_cached_session(self, session_id: str, mode: str):
        """Return cached session data."""
        session_meta = self.sessions[session_id]
        
        if mode == "device":
            # Return zip file for device mode
            try:
                with open(session_meta.zip_path, 'rb') as f:
                    zip_data = f.read()
                return {
                    "session_id": session_id,
                    "success": True,
                    "zip_data": zip_data,
                    "cached": True
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to read cached zip: {e}",
                    "session_id": session_id
                }
        else:
            # Server mode - return success message
            return {
                "session_id": session_id,
                "success": True,
                "message": "Using cached session",
                "cached": True
            }
    
    def _create_new_session(self, lat: float, lng: float, meters: int, mode: str):
        """Create new session with full processing."""
        # Use existing process_init_map_request for processing
        temp_sessions = {}  # Temporary dict for legacy compatibility
        result = process_init_map_request(
            lat=lat, 
            lng=lng, 
            meters=meters, 
            mode="device",  # Always get full data for storage
            embedder=self.embedder,
            sessions=temp_sessions,
            save_sessions_callback=lambda: None,  # No-op for temp
            progress_callback=None  # No progress callback for direct calls
        )
        
        if not result.get("success"):
            return result
            
        # Extract session data and store files
        session_id = result["session_id"]
        session_data = temp_sessions[session_id]
        
        # Store files and create metadata
        session_meta = self._store_session_files(session_data)
        self.sessions[session_id] = session_meta
        self._save_sessions()
        
        # Return appropriate response based on mode
        if mode == "device":
            try:
                with open(session_meta.zip_path, 'rb') as f:
                    zip_data = f.read()
                return {
                    "session_id": session_id,
                    "success": True,
                    "zip_data": zip_data
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to read created zip: {e}",
                    "session_id": session_id
                }
        else:
            return {
                "session_id": session_id,
                "success": True,
                "message": f"Map session created with {len(session_data.patches)} patches",
                "coverage": f"{meters}m x {meters}m",
                "patch_count": len(session_data.patches)
            }
    
    def _store_session_files(self, session_data: SessionData) -> SessionMetadata:
        """Store session files and return metadata."""
        session_id = session_data.session_id
        
        # Store map image
        map_path = self.maps_dir / f"{session_id}.png"
        map_image = Image.fromarray(session_data.full_map.astype(np.uint8))
        map_image.save(map_path)
        
        # Store embeddings with metadata
        embeddings_path = self.embeddings_dir / f"{session_id}.json"
        embeddings_data = {
            "session_id": session_id,
            "map_bounds": session_data.map_bounds,
            "patch_size": session_data.patch_size,
            "meters_coverage": session_data.meters_coverage,
            "patches": [
                {
                    "embedding": patch.embedding_data["embedding"].tolist(),
                    "lat": patch.lat,
                    "lng": patch.lng,
                    "coords": patch.patch_coords
                }
                for patch in session_data.patches
            ]
        }
        
        with open(embeddings_path, 'w') as f:
            json.dump(embeddings_data, f)
        
        # Create zip file
        zip_path = self.zips_dir / f"{session_id}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(map_path, "map.png")
            zf.write(embeddings_path, "embeddings.json")
        
        return SessionMetadata(
            session_id=session_id,
            created_at=session_data.created_at,
            map_path=str(map_path),
            embeddings_path=str(embeddings_path),
            zip_path=str(zip_path)
        )

    def fetch_gps(self, image_data: bytes, session_id: str, logging_id: Optional[str] = None, visualization: bool = False):
        """
        Find GPS coordinates for an image by matching against session embeddings.
        
        Args:
            image_data: Raw image bytes
            session_id: Session ID to search in
            logging_id: Optional logging ID for enhanced logging
            visualization: Enable visualization updates
            
        Returns:
            Dictionary with GPS coordinates and similarity info
        """
        # Load embeddings from file for matching
        if session_id not in self.sessions:
            return {
                "success": False,
                "error": f"Session {session_id} not found"
            }
        
        session_meta = self.sessions[session_id]
        
        # Load embeddings data
        try:
            with open(session_meta.embeddings_path, 'r') as f:
                embeddings_data = json.load(f)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load embeddings: {e}"
            }
        
        # Find closest patch using embeddings
        result = self._find_closest_patch(image_data, embeddings_data)
        
        # Enhanced logging if logging_id provided
        if result.get("success") and logging_id and visualization:
            self._update_enhanced_logs(session_id, logging_id, result, image_data)
        
        return result
    
    def _find_closest_patch(self, image_data: bytes, embeddings_data: dict):
        """Find closest patch using cosine similarity."""
        try:
            # Convert image to embedding
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_array = np.array(image)
            query_embedding = self.embedder.embed_patch(image_array)["embedding"]
            
            # Find closest patch
            best_similarity = -1
            best_patch = None
            
            for patch in embeddings_data["patches"]:
                patch_embedding = np.array(patch["embedding"])
                
                # Cosine similarity
                dot_product = np.dot(query_embedding, patch_embedding)
                norm_query = np.linalg.norm(query_embedding)
                norm_patch = np.linalg.norm(patch_embedding)
                similarity = dot_product / (norm_query * norm_patch)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_patch = patch
            
            if best_patch is None:
                return {
                    "success": False,
                    "error": "No patches found"
                }
            
            return {
                "success": True,
                "session_id": embeddings_data["session_id"],
                "gps": {
                    "lat": best_patch["lat"],
                    "lng": best_patch["lng"]
                },
                "similarity": float(best_similarity),
                "confidence": "high" if best_similarity > 0.8 else "medium" if best_similarity > 0.6 else "low",
                "patch_coords": best_patch["coords"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to process image: {e}"
            }
    
    def _update_enhanced_logs(self, session_id: str, logging_id: str, result: dict, image_data: bytes):
        """Update enhanced logs with CSV, error plots, and map visualization."""
        import io
        import matplotlib.pyplot as plt
        import csv
        import math
        
        # Create logging directory
        log_dir = self.logs_dir / session_id / logging_id
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract metadata for ground truth GPS
        try:
            # Save image temporarily for metadata extraction
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_file.write(image_data)
                tmp_path = tmp_file.name
            
            try:
                img_metadata = extract_metadata(tmp_path)
                if img_metadata and img_metadata.telemetry:
                    g_lat = img_metadata.telemetry.position_2d[0] if img_metadata.telemetry.position_2d else None
                    g_lng = img_metadata.telemetry.position_2d[1] if img_metadata.telemetry.position_2d else None
                else:
                    g_lat = g_lng = None
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            g_lat = g_lng = None
        
        # Get predicted GPS
        pred_lat = result["gps"]["lat"]
        pred_lng = result["gps"]["lng"]
        
        # Calculate error in meters (haversine distance)
        error_meters = 0
        if g_lat is not None and g_lng is not None:
            # Haversine formula
            R = 6371000  # Earth radius in meters
            dlat = math.radians(pred_lat - g_lat)
            dlng = math.radians(pred_lng - g_lng)
            a = (math.sin(dlat/2)**2 + 
                 math.cos(math.radians(g_lat)) * math.cos(math.radians(pred_lat)) * 
                 math.sin(dlng/2)**2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            error_meters = R * c
        
        # Update CSV log
        csv_path = log_dir / "path.csv"
        frame_num = self._get_next_frame_number(csv_path)
        
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if frame_num == 1:  # Write header for first frame
                writer.writerow(['frame_num', 'g_lat', 'g_lng', 'pred_lat', 'pred_lng', 'error_meters', 'similarity'])
            writer.writerow([frame_num, g_lat, g_lng, pred_lat, pred_lng, error_meters, result["similarity"]])
        
        # Update error plot
        self._update_error_plot(csv_path, log_dir)
        
        # Update map visualization
        self._update_map_visualization(session_id, log_dir, g_lat, g_lng, pred_lat, pred_lng)
        
    def _get_next_frame_number(self, csv_path: Path) -> int:
        """Get next frame number from CSV."""
        if not csv_path.exists():
            return 1
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
                if len(rows) <= 1:  # Only header or empty
                    return 1
                return len(rows)  # Header + data rows
        except:
            return 1
    
    def _update_error_plot(self, csv_path: Path, log_dir: Path):
        """Update error vs time plot with 50-frame average."""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
        except ImportError as e:
            print(f"Warning: Could not import plotting libraries: {e}")
            print("Enhanced plotting disabled. Install matplotlib and pandas for full functionality.")
            return
        
        try:
            # Read CSV data
            df = pd.read_csv(csv_path)
            if len(df) == 0:
                return
            
            # Ensure numeric columns
            df['frame_num'] = pd.to_numeric(df['frame_num'], errors='coerce')
            df['error_meters'] = pd.to_numeric(df['error_meters'], errors='coerce')
            
            # Remove any rows with NaN values in critical columns
            df = df.dropna(subset=['frame_num', 'error_meters'])
            
            if len(df) == 0:
                print("No valid data for plotting")
                return
            
            # Calculate 50-frame rolling average
            window_size = min(50, len(df))
            if window_size >= 1:
                df['error_avg_50'] = df['error_meters'].rolling(window=window_size, center=True, min_periods=1).mean()
            else:
                df['error_avg_50'] = df['error_meters']
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            # Plot individual errors
            if len(df) > 1:
                plt.plot(df['frame_num'], df['error_meters'], alpha=0.3, label='Error (m)', color='red', marker='o', markersize=2)
                plt.plot(df['frame_num'], df['error_avg_50'], label=f'{window_size}-frame average', color='blue', linewidth=2)
            else:
                plt.scatter(df['frame_num'], df['error_meters'], label='Error (m)', color='red', s=50)
            
            plt.xlabel('Frame Number')
            plt.ylabel('Error (meters)')
            plt.title('GPS Prediction Error Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add current average text
            if len(df) > 0:
                if len(df) > 1 and not df['error_avg_50'].isna().all():
                    current_avg = df['error_avg_50'].iloc[-1]
                    if pd.isna(current_avg):
                        current_avg = df['error_meters'].mean()
                else:
                    current_avg = df['error_meters'].iloc[-1]
                
                plt.text(0.02, 0.98, f'Current avg: {current_avg:.1f}m', 
                        transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        verticalalignment='top')
            
            plt.tight_layout()
            plt.savefig(log_dir / "error_plot.png", dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error updating plot: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_map_visualization(self, session_id: str, log_dir: Path, g_lat: Optional[float], g_lng: Optional[float], 
                                 pred_lat: float, pred_lng: float):
        """Update map visualization with green (ground truth) and red (predicted) dots."""
        try:
            session_meta = self.sessions[session_id]
            
            # Load map image
            map_image = Image.open(session_meta.map_path)
            
            # Load embeddings for map bounds
            with open(session_meta.embeddings_path, 'r') as f:
                embeddings_data = json.load(f)
            map_bounds = embeddings_data["map_bounds"]
            
            # Convert GPS to pixel coordinates
            def gps_to_pixel(lat, lng):
                img_width, img_height = map_image.size
                lat_range = map_bounds["max_lat"] - map_bounds["min_lat"]
                lng_range = map_bounds["max_lng"] - map_bounds["min_lng"]
                
                if lat_range == 0 or lng_range == 0:
                    return img_width // 2, img_height // 2
                
                x = int((lng - map_bounds["min_lng"]) / lng_range * img_width)
                y = int((map_bounds["max_lat"] - lat) / lat_range * img_height)
                
                x = max(0, min(x, img_width - 1))
                y = max(0, min(y, img_height - 1))
                return x, y
            
            # Load existing visualization or create new
            viz_path = log_dir / "map_paths.png"
            if viz_path.exists():
                viz_image = Image.open(viz_path)
            else:
                viz_image = map_image.copy()
            
            from PIL import ImageDraw
            draw = ImageDraw.Draw(viz_image)
            
            # Draw predicted location (red)
            pred_x, pred_y = gps_to_pixel(pred_lat, pred_lng)
            dot_size = max(3, min(viz_image.width, viz_image.height) // 200)
            draw.ellipse([pred_x - dot_size, pred_y - dot_size, pred_x + dot_size, pred_y + dot_size], 
                        fill=(255, 0, 0), outline=(255, 255, 255), width=1)
            
            # Draw ground truth location (green) if available
            if g_lat is not None and g_lng is not None:
                gt_x, gt_y = gps_to_pixel(g_lat, g_lng)
                draw.ellipse([gt_x - dot_size, gt_y - dot_size, gt_x + dot_size, gt_y + dot_size], 
                            fill=(0, 255, 0), outline=(255, 255, 255), width=1)
                
                # Draw line between ground truth and predicted
                draw.line([(gt_x, gt_y), (pred_x, pred_y)], fill=(255, 255, 0), width=1)
            
            viz_image.save(viz_path)
            
        except Exception as e:
            print(f"Error updating map visualization: {e}")

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
    
    def get_session(self, session_id: str) -> Optional[SessionMetadata]:
        """Get session metadata by ID."""
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
        
        image_path = server_paths_dir / f"path_{session_data.session_id}.jpg"
        viz_img.save(image_path, 'JPEG', quality=95)
        
        return str(image_path)

    def cleanup_session(self, session_id: str) -> bool:
        """Remove a session from memory and persistent storage."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self._save_sessions()
            print(f"Session {session_id} cleaned up")
            return True
        return False