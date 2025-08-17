"""
High-level GPS processing pipeline: decode -> embed -> match -> visualize.
"""
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
import io
import json

from models import SessionData
from general.process import find_closest_patch
from general.image_metadata import extract_metadata
from general.models import PathPoint
from pathlib import Path
import math


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def _update_statistics(session_data, pred_lat: float, pred_lng: float, gt_lat: float, gt_lng: float) -> None:
    try:
        stats_dir = Path('data/statistics')
        stats_dir.mkdir(parents=True, exist_ok=True)
        sid = session_data.session_id
        csv_path = stats_dir / f'stats_{sid}.csv'
        txt_path = stats_dir / f'stats_{sid}.txt'
        curve_png = stats_dir / f'error_curve_{sid}.png'
        hist_png = stats_dir / f'error_hist_{sid}.png'

        error_m = _haversine_m(pred_lat, pred_lng, gt_lat, gt_lng)
        frame_idx = len(session_data.path_data)

        header = 'frame,pred_lat,pred_lng,gt_lat,gt_lng,error_m\n'
        if not csv_path.exists():
            csv_path.write_text(header)
        with open(csv_path, 'a') as f:
            f.write(f'{frame_idx},{pred_lat},{pred_lng},{gt_lat},{gt_lng},{error_m}\n')

        # Recompute plots
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import csv
            errors = []
            frames = []
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    frames.append(int(row['frame']))
                    errors.append(float(row['error_m']))
            if errors:
                arr = np.array(errors)
                plt.figure(figsize=(8,4))
                plt.plot(frames, arr, label='Error (m)')
                plt.xlabel('Frame')
                plt.ylabel('Meters')
                plt.title('Prediction Error Over Time')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(curve_png)
                plt.close()

                plt.figure(figsize=(6,4))
                plt.hist(arr, bins=30, color='#1f77b4', alpha=0.9)
                plt.xlabel('Meters')
                plt.ylabel('Count')
                plt.title('Error Distribution')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(hist_png)
                plt.close()

                summary = f"count={len(arr)}, mean={arr.mean():.2f} m, median={np.median(arr):.2f} m, max={arr.max():.2f} m\n"
                with open(txt_path, 'w') as f:
                    f.write(summary)
        except Exception as _:
            pass
    except Exception as _:
        pass


def find_closest_patch_from_data(query_embedding, patches, metadata=None):
    """Find closest patch from loaded embeddings data."""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    if not patches:
        return None
    
    # Extract embeddings from patches
    patch_embeddings = []
    for patch in patches:
        if "embedding" in patch:
            patch_embeddings.append(patch["embedding"])
    
    if not patch_embeddings:
        return None
    
    # Convert to numpy array
    patch_embeddings = np.array(patch_embeddings)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, patch_embeddings)[0]
    
    # Find best match
    best_idx = np.argmax(similarities)
    best_patch = patches[best_idx]
    
    return {
        "lat": best_patch["lat"],
        "lng": best_patch["lng"], 
        "similarity": float(similarities[best_idx]),
        "patch_index": int(best_idx)
    }


def process_fetch_gps_request(
    image_data: bytes,
    session_id: str,
    embedder,
    logging_id: Optional[str] = None,
    visualization: bool = False,
    image_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Process a fetch_gps request: embed, match, update path, return lat/lng."""
    try:
        # Load embeddings from local file (device version)
        embeddings_path = Path(f"data/embeddings/{session_id}.json")
        if not embeddings_path.exists():
            return {"success": False, "error": f"Embeddings not found for session {session_id}"}
        
        with open(embeddings_path, 'r') as f:
            embeddings_data = json.load(f)
        
        # Extract patches data
        patches = embeddings_data.get("patches", [])
        if not patches:
            return {"success": False, "error": f"No patches found for session {session_id}"}

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
                        "g_lat": getattr(telemetry, 'g_lat', None),
                        "g_lon": getattr(telemetry, 'g_lon', None),
                    }
        except Exception:
            pass

        # Generate embedding (support dict or ndarray from different embedders)
        rep = embedder.embed_patch(image_array)
        if isinstance(rep, dict):
            query_embedding = rep.get("embedding")
        else:
            query_embedding = rep
        if query_embedding is None:
            return {"success": False, "error": "Embedder returned no embedding"}

        # Find closest patch using loaded embeddings data
        result = find_closest_patch_from_data(query_embedding, patches, metadata=metadata_dict)
        if result is None:
            return {"success": False, "error": "No patches found in session"}

        # Enhanced logging if requested (device version - simplified)
        if visualization and logging_id:
            try:
                # Create logs directory structure
                logs_dir = Path(f"data/logs/{session_id}/{logging_id}")
                logs_dir.mkdir(parents=True, exist_ok=True)
                
                # Log GPS data to CSV
                g_lat = metadata_dict.get("g_lat")
                g_lng = metadata_dict.get("g_lng")
                
                import csv
                import time
                csv_path = logs_dir / "path.csv"
                
                # Write header if file doesn't exist
                write_header = not csv_path.exists()
                with open(csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if write_header:
                        writer.writerow(['frame_num', 'g_lat', 'g_lng', 'pred_lat', 'pred_lng', 'error_meters', 'similarity'])
                    
                    # Calculate frame number and error
                    frame_num = len(list(csv.reader(open(csv_path, 'r')))) if csv_path.exists() else 1
                    error_meters = 0
                    if g_lat is not None and g_lng is not None:
                        # Simple distance calculation
                        import math
                        R = 6371000  # Earth radius in meters
                        dlat = math.radians(result["lat"] - g_lat)
                        dlng = math.radians(result["lng"] - g_lng)
                        a = (math.sin(dlat/2)**2 + 
                             math.cos(math.radians(g_lat)) * math.cos(math.radians(result["lat"])) * 
                             math.sin(dlng/2)**2)
                        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                        error_meters = R * c
                    
                    writer.writerow([frame_num, g_lat, g_lng, result["lat"], result["lng"], error_meters, result["similarity"]])
                
            except Exception as e:
                print(f"Warning: Enhanced logging failed: {e}")

        return {
            "success": True,
            "session_id": session_id,
            "gps": {"lat": result["lat"], "lng": result["lng"]},
            "similarity": result["similarity"],
            "patch_index": result["patch_index"]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
