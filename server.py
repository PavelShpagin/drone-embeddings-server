#!/usr/bin/env python3
"""
Satellite Image Embedding Server
===============================
AWS-compatible server for processing satellite imagery with DINOv2 embeddings.

API:
- init_map(lat, lng, meters=2000, mode="server") -> session_id or map_data
- fetch_gps(image, session_id) -> GPS coordinates
"""

import sys
from pathlib import Path
import time
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
import uvicorn

# Add src to path for imports
server_src_path = str(Path(__file__).parent / "src")
sys.path.insert(0, server_src_path)
# Add general to path so models import works
sys.path.insert(0, str(Path(__file__).parent / "src" / "general"))

from src.general.models import (
    InitMapRequest, HealthResponse, SessionInfo, SessionsResponse,
    FetchGpsRequest, FetchGpsResponse, VisualizePathRequest, VisualizePathResponse,
    GenerateVideoRequest, GenerateVideoResponse
)
from src.server.server_core import SatelliteEmbeddingServer


# Global server instance
server = SatelliteEmbeddingServer()


def init_map(lat: float, lng: float, meters: int = 2000, mode: str = "server", session_id: Optional[str] = None) -> Dict[str, Any]:
    """Wrapper function for backward compatibility."""
    return server.init_map(lat, lng, meters, mode, session_id)


# FastAPI HTTP Server
app = FastAPI(
    title="Satellite Embedding Server",
    description="Process satellite imagery with DINOv2 embeddings for GPS localization",
    version="1.0.0"
)


@app.post("/init_map")
async def http_init_map(request: InitMapRequest):
    """HTTP endpoint for initializing map sessions."""
    try:
        result = server.init_map(
            lat=request.lat,
            lng=request.lng,
            meters=request.meters,
            mode=request.mode,
            session_id=request.session_id
        )
        # Handle zip_data response for device mode
        if request.mode == "device" and "zip_data" in result:
            from fastapi.responses import Response
            return Response(
                content=result["zip_data"], 
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename=session_{result['session_id']}.zip"}
            )
        # Optionally return compressed payload for device mode
        elif request.mode == "device" and request.compressed:
            from fastapi.responses import Response
            import gzip
            import json
            payload = json.dumps(result).encode('utf-8')
            compressed = gzip.compress(payload, compresslevel=5)
            return Response(content=compressed, media_type="application/json", headers={
                "Content-Encoding": "gzip"
            })
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
                # Load embeddings data to get patch count and bounds
                try:
                    import json
                    with open(session.embeddings_path, 'r') as f:
                        embeddings_data = json.load(f)
                    
                    session_data.append(SessionInfo(
                        session_id=session_id,
                        created_at=session.created_at,
                        meters_coverage=embeddings_data["meters_coverage"],
                        patch_count=len(embeddings_data["patches"]),
                        map_bounds=embeddings_data["map_bounds"]
                    ))
                except Exception as e:
                    print(f"Error loading session {session_id}: {e}")
                    continue
        
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


@app.post("/fetch_gps", response_model=FetchGpsResponse)
async def http_fetch_gps(session_id: str = Form(...), image: UploadFile = File(...), 
                        logging_id: Optional[str] = Form(None), visualization: bool = Form(False)):
    """HTTP endpoint for finding GPS coordinates from an image."""
    try:
        # Read image data
        image_data = await image.read()
        
        # Process request
        result = server.fetch_gps(image_data, session_id, logging_id, visualization)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "success": False,
            "error": str(e)
        })

@app.post("/visualize_path")
async def http_visualize_path(request: VisualizePathRequest):
    """HTTP endpoint for generating path visualization."""
    try:
        from src.general.visualize_map import process_path_visualization_request
        result = process_path_visualization_request(request.session_id, server.sessions)
        
        if result["success"]:
            from fastapi.responses import StreamingResponse
            import io
            return StreamingResponse(
                io.BytesIO(result["image_bytes"]),
                media_type="image/jpeg",
                headers={"Content-Disposition": f"attachment; filename=path_{request.session_id}.jpg"}
            )
        else:
            return VisualizePathResponse(**result)
                
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "success": False,
            "error": str(e)
        })


@app.post("/get_video")
async def http_get_video(request: GenerateVideoRequest):
    """HTTP endpoint for downloading real-time generated path video."""
    try:
        from src.general.visualize_map import get_session_video_path
        from pathlib import Path
        
        server_paths_dir = Path("data/server_paths")
        
        # Get real-time generated video
        video_path = get_session_video_path(request.session_id, server_paths_dir)
        
        # Return video file as download
        from fastapi.responses import FileResponse
        return FileResponse(
            video_path,
            media_type="video/avi",
            filename=f"path_video_{request.session_id}.avi"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "success": False,
            "error": str(e)
        })


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
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port, debug=args.debug)