"""
GPS Fetching Module for Satellite Embedding Server
Handles finding GPS coordinates by matching image embeddings.
"""
import numpy as np
from typing import Optional, Dict, Any, List
from PIL import Image
import io
from models import SessionData, PatchData


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def find_closest_patch(query_embedding: np.ndarray, session_data) -> Optional[Dict[str, Any]]:
    """
    Find the closest patch in session data to the query embedding.
    
    Args:
        query_embedding: Embedding vector from input image
        session_data: SessionData object containing patches
        
    Returns:
        Dictionary with closest patch data including lat, lng, similarity score
    """
    if not session_data or not session_data.patches:
        return None
    
    best_similarity = -1.0
    best_patch = None
    
    for patch in session_data.patches:
        similarity = cosine_similarity(query_embedding, patch.embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_patch = patch
    
    if best_patch is None:
        return None
    
    return {
        "lat": best_patch.lat,
        "lng": best_patch.lng, 
        "similarity": float(best_similarity),
        "patch_coords": best_patch.patch_coords,
        "confidence": "high" if best_similarity > 0.8 else "medium" if best_similarity > 0.6 else "low"
    }


def process_fetch_gps_request(image_data: bytes, session_id: str, embedder, sessions: dict) -> Dict[str, Any]:
    """
    Process a fetch_gps request.
    
    Args:
        image_data: Raw image bytes
        session_id: Session ID to search in
        embedder: DINOv2 embedder instance
        sessions: Sessions dictionary
        
    Returns:
        Response dictionary with GPS coordinates or error
    """
    try:
        # Check if session exists
        if session_id not in sessions:
            return {
                "success": False,
                "error": f"Session {session_id} not found"
            }
        
        session_data = sessions[session_id]
        
        # Convert image bytes to numpy array
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image)
        
        # Generate embedding for input image
        query_embedding = embedder.embed_patch(image_array)
        
        # Find closest patch
        result = find_closest_patch(query_embedding, session_data)
        
        if result is None:
            return {
                "success": False,
                "error": "No patches found in session"
            }
        
        return {
            "success": True,
            "session_id": session_id,
            "gps": {
                "lat": result["lat"],
                "lng": result["lng"]
            },
            "similarity": result["similarity"],
            "confidence": result["confidence"],
            "patch_coords": result["patch_coords"]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }