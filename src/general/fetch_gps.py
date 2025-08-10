"""
High-level GPS processing pipeline: decode -> embed -> match -> visualize.
"""
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
import io

from models import SessionData
from general.process import find_closest_patch
from general.image_metadata import extract_metadata


def process_fetch_gps_request(
    image_data: bytes,
    session_id: str,
    embedder,
    sessions: dict,
    visualize: bool = True,
    image_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Process a fetch_gps request: embed, match, update path, return lat/lng."""
    try:
        if session_id not in sessions:
            return {"success": False, "error": f"Session {session_id} not found"}

        session_data: SessionData = sessions[session_id]

        # Convert image bytes to numpy array
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image)

        # Extract metadata from the image path if available
        metadata_dict: Dict[str, Any] = {}
        try:
            meta_path = image_path or getattr(image, 'filename', None)
            if meta_path and isinstance(meta_path, str):
                md = extract_metadata(meta_path)
                telemetry = md.telemetry
                if telemetry:
                    metadata_dict = {
                        "pos2D": telemetry.position_2d,
                        "height": telemetry.height,
                        "g_lat": None,
                        "g_lon": None,
                    }
        except Exception:
            pass

        # Generate embedding
        rep = embedder.embed_patch(image_array)
        query_embedding = rep.get("embedding", rep)

        # Find closest patch with metadata provided (for future flexibility)
        result = find_closest_patch(query_embedding, session_data, metadata=metadata_dict)
        if result is None:
            return {"success": False, "error": "No patches found in session"}

        if visualize:
            import time
            from general.visualize_map import update_path_visualization
            from general.models import PathPoint

            image_path2 = update_path_visualization(session_data, result["lat"], result["lng"])
            session_data.path_image_file = image_path2

            new_point = PathPoint(lat=result["lat"], lng=result["lng"], timestamp=time.time())
            session_data.path_data.append(new_point)

        return {
            "success": True,
            "session_id": session_id,
            "gps": {"lat": result["lat"], "lng": result["lng"]},
            "similarity": result["similarity"],
            "confidence": result["confidence"],
            "patch_coords": result["patch_coords"]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
