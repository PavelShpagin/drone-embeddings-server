"""
Data Models for Satellite Embedding Server
==========================================
Contains all data structures and Pydantic models used by the server.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pydantic import BaseModel


# Core Data Structures
@dataclass
class PatchData:
    """Data for a single image patch with representation data and GPS coordinates.

    embedding_data holds at least key "embedding" (np.ndarray). It can be extended
    with additional keys by different methods (e.g. patch image, metadata, etc.).
    """
    embedding_data: Dict[str, Any]
    lat: float
    lng: float
    patch_coords: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in image coordinates


@dataclass
class PathPoint:
    """Single point in a GPS tracking path."""
    lat: float
    lng: float
    timestamp: float
    pos2D: Optional[Tuple[float, float]] = None
    height: Optional[float] = None
    image_data: Optional[bytes] = None
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

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
    path_data: List[PathPoint] = field(default_factory=list)  # GPS tracking path data
    path_image_file: Optional[str] = None  # Incremental path visualization image


# Pydantic Models for API
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
    sessions: List[SessionInfo]
    count: int


class FetchGpsRequest(BaseModel):
    session_id: str


class FetchGpsResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    gps: Optional[dict] = None
    similarity: Optional[float] = None
    confidence: Optional[str] = None
    patch_coords: Optional[list] = None
    error: Optional[str] = None

class VisualizePathRequest(BaseModel):
    session_id: str

class VisualizePathResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    path_points: Optional[int] = None
    message: Optional[str] = None
    error: Optional[str] = None

class GenerateVideoRequest(BaseModel):
    session_id: str
    fps: Optional[float] = 2.0

class GenerateVideoResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    video_path: Optional[str] = None
    frame_count: Optional[int] = None