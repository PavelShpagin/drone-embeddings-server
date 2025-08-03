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
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
import uvicorn

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from models import (
    InitMapRequest, HealthResponse, SessionInfo, SessionsResponse, 
    FetchGpsRequest, FetchGpsResponse
)
from server_core import SatelliteEmbeddingServer


# Global server instance
server = SatelliteEmbeddingServer()


def init_map(lat: float, lng: float, meters: int = 2000, mode: str = "server") -> Dict[str, Any]:
    """Wrapper function for backward compatibility."""
    return server.init_map(lat, lng, meters, mode)


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
            mode=request.mode
        )
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


@app.post("/fetch_gps", response_model=FetchGpsResponse)
async def http_fetch_gps(session_id: str = Form(...), image: UploadFile = File(...)):
    """HTTP endpoint for finding GPS coordinates from an image."""
    try:
        # Read image data
        image_data = await image.read()
        
        # Process request
        result = server.fetch_gps(image_data, session_id)
        return result
        
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