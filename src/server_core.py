"""
Core Server Class for Satellite Embedding Server
================================================
Main server logic including session management and persistent storage.
"""

import os
import pickle
from typing import Dict, List, Optional
from models import SessionData
from embedder import TinyDINOEmbedder
from init_map import process_init_map_request
from fetch_gps import process_fetch_gps_request


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
        return process_fetch_gps_request(image_data, session_id, self.embedder, self.sessions)
    
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