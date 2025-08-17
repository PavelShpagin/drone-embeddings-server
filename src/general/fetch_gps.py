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

        # Find closest patch with metadata provided (for future flexibility)
        result = find_closest_patch(query_embedding, session_data, metadata=metadata_dict)
        if result is None:
            return {"success": False, "error": "No patches found in session"}

        if visualize:
            import time
            from general.visualize_map import update_path_visualization

            g_lat = metadata_dict.get("g_lat")
            g_lon = metadata_dict.get("g_lon")
            image_path2 = update_path_visualization(session_data, result["lat"], result["lng"], g_lat, g_lon)
            session_data.path_image_file = image_path2

            new_point = PathPoint(lat=result["lat"], lng=result["lng"], timestamp=time.time())
            session_data.path_data.append(new_point)

            # Append GT point if available
            if g_lat is not None and g_lon is not None and hasattr(session_data, 'gt_path_data'):
                gt_point = PathPoint(lat=g_lat, lng=g_lon, timestamp=time.time())
                session_data.gt_path_data.append(gt_point)
                _update_statistics(session_data, result["lat"], result["lng"], g_lat, g_lon)

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
