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
import uuid
import asyncio
import json
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
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
from src.server.fetch_logs import package_session_logs, get_available_sessions, get_session_logs_summary
from src.server.server_core import SatelliteEmbeddingServer
from pydantic import BaseModel
from typing import List, Dict, Any

# Log-related models
class FetchLogsRequest(BaseModel):
    session_id: str
    logger_id: Optional[str] = None

class LogsSessionInfo(BaseModel):
    session_id: str
    loggers: List[Dict[str, Any]]
    logger_count: int

class AvailableLogsResponse(BaseModel):
    success: bool
    sessions: List[LogsSessionInfo]
    total_sessions: int

class LogsSummaryResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    loggers: Optional[Dict[str, Any]] = None
    total_size: Optional[int] = None
    total_files: Optional[int] = None
    message: Optional[str] = None

# Progress tracking models
class ProgressTask:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.status = "running"  # "running", "completed", "failed"
        self.progress = 0  # 0-100
        self.message = "Starting..."
        self.zip_data = None
        self.session_id = None
        self.error = None
        self.created_at = time.time()
        
class ProgressResponse(BaseModel):
    status: str
    progress: int
    message: str
    zip_data: Optional[str] = None  # Base64 encoded zip data when completed
    session_id: Optional[str] = None
    error: Optional[str] = None

class AsyncInitResponse(BaseModel):
    task_id: str
    status: str = "started"


# Global server instance
server = SatelliteEmbeddingServer()

# Global background tasks storage
background_tasks: Dict[str, ProgressTask] = {}

# WebSocket connection management
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        self.active_connections[task_id] = websocket
        print(f"✓ WebSocket connected for task {task_id}")
    
    def disconnect(self, task_id: str):
        if task_id in self.active_connections:
            del self.active_connections[task_id]
            print(f"✗ WebSocket disconnected for task {task_id}")
    
    async def send_progress(self, task_id: str, progress_data: dict):
        if task_id in self.active_connections:
            try:
                await self.active_connections[task_id].send_text(json.dumps(progress_data))
                print(f"📡 Sent progress to {task_id}: {progress_data['progress']}% - {progress_data['message']}")
            except Exception as e:
                print(f"❌ Error sending progress to {task_id}: {e}")
                self.disconnect(task_id)

manager = ConnectionManager()

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
async def http_init_map(request: InitMapRequest, background_tasks_runner: BackgroundTasks):
    """HTTP endpoint for initializing map sessions."""
    try:
        # Handle device_async mode for progress tracking
        if request.mode == "device_async":
            task_id = str(uuid.uuid4())
            task = ProgressTask(task_id)
            background_tasks[task_id] = task
            
            # Start background processing
            background_tasks_runner.add_task(
                _process_init_map_async,
                task_id=task_id,
                lat=request.lat,
                lng=request.lng,
                meters=request.meters,
                session_id=request.session_id
            )
            
            return AsyncInitResponse(task_id=task_id)
        
        # Handle regular modes (server, device)
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


@app.get("/progress/{task_id}", response_model=ProgressResponse)
async def get_progress(task_id: str):
    """Get progress of an async init_map task."""
    task = background_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail={
            "success": False,
            "error": "Task not found"
        })
    
    return ProgressResponse(
        status=task.status,
        progress=task.progress,
        message=task.message,
        zip_data=task.zip_data,
        session_id=task.session_id,
        error=task.error
    )


@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time progress updates."""
    try:
        await manager.connect(websocket, task_id)
        
        # Send initial connection confirmation
        await manager.send_progress(task_id, {
            "status": "connected",
            "progress": 0,
            "message": "WebSocket connected - waiting for task to start...",
            "task_id": task_id
        })
        
        # Keep connection alive and listen for client messages
        while True:
            try:
                # Wait for any client messages (mostly just to detect disconnection)
                await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                # No message received, just continue
                continue
            except WebSocketDisconnect:
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(task_id)


async def _process_init_map_async(task_id: str, lat: float, lng: float, meters: int, session_id: Optional[str] = None):
    """Background task for processing init_map requests with progress updates."""
    task = background_tasks[task_id]
    
    def update_progress(progress: int, message: str):
        """Update task progress and send WebSocket notification."""
        task.progress = progress
        task.message = message
        print(f"Task {task_id}: {progress}% - {message}")
        
        # Send WebSocket update
        progress_data = {
            "status": task.status,
            "progress": progress,
            "message": message,
            "task_id": task_id
        }
        
        # Use asyncio to send the WebSocket update
        try:
            asyncio.create_task(manager.send_progress(task_id, progress_data))
        except Exception as e:
            print(f"Failed to send WebSocket update: {e}")
    
    try:
        update_progress(5, "Initializing satellite data request...")
        
        # Check for cached session first
        if session_id and session_id in server.sessions:
            update_progress(30, "Found cached session, loading data...")
            result = server._return_cached_session(session_id, "device")
            
            if result.get("success"):
                update_progress(80, "Processing cached data...")
                
                # Convert zip_data to base64 for JSON response
                if "zip_data" in result:
                    import base64
                    task.zip_data = base64.b64encode(result["zip_data"]).decode('utf-8')
                    task.session_id = result["session_id"]
                
                update_progress(100, "Cached data ready!")
                task.status = "completed"
                
                # Send final WebSocket notification with completion data
                await manager.send_progress(task_id, {
                    "status": "completed",
                    "progress": 100,
                    "message": "Cached data ready!",
                    "task_id": task_id,
                    "zip_data": task.zip_data,
                    "session_id": task.session_id
                })
                
                # Clean up after some time
                await asyncio.sleep(10)
                if task_id in background_tasks:
                    del background_tasks[task_id]
                return
        
        update_progress(10, "Connecting to satellite imagery service...")
        
        # Create new session with progress updates
        update_progress(15, "Downloading satellite imagery...")
        await asyncio.sleep(1)  # Simulate network delay
        
        update_progress(30, "Processing satellite tiles...")
        await asyncio.sleep(1)
        
        update_progress(45, "Generating embeddings...")
        
        # Call the actual processing with progress callback
        from server.init_map import process_init_map_request
        from general.models import SessionMetadata
        import json
        temp_sessions = {}
        
        def progress_wrapper(message, progress):
            """Wrapper to make progress callback async-compatible."""
            print(f"PROGRESS WRAPPER CALLED: {progress}% - {message}")
            update_progress(progress, message)
            # Force immediate update to task object
            task.progress = progress
            task.message = message
        
        result = process_init_map_request(
            lat=lat,
            lng=lng, 
            meters=meters,
            mode="device",
            embedder=server.embedder,
            sessions=temp_sessions,
            save_sessions_callback=lambda: None,
            progress_callback=progress_wrapper
        )
        
        # Store session data in server
        if result.get("success"):
            session_id = result["session_id"]
            session_data = temp_sessions[session_id]
            
            # Store files in server storage
            import os
            os.makedirs("data/maps", exist_ok=True)
            os.makedirs("data/embeddings", exist_ok=True)
            
            # Save map image (convert numpy array to PNG bytes)
            from PIL import Image
            import numpy as np
            map_path = f"data/maps/{session_id}.png"
            
            # Convert numpy array to PIL Image and save
            if session_data.full_map.dtype != np.uint8:
                # Normalize to 0-255 if needed
                map_array = ((session_data.full_map - session_data.full_map.min()) / 
                           (session_data.full_map.max() - session_data.full_map.min()) * 255).astype(np.uint8)
            else:
                map_array = session_data.full_map
                
            map_image = Image.fromarray(map_array)
            map_image.save(map_path)
            
            # Save embeddings (extract embeddings from patches)
            embeddings_path = f"data/embeddings/{session_id}.json"
            embeddings_data = []
            for patch in session_data.patches:
                embeddings_data.append({
                    "embedding": patch.embedding_data["embedding"].tolist() if hasattr(patch.embedding_data["embedding"], 'tolist') else patch.embedding_data["embedding"],
                    "lat": patch.lat,
                    "lng": patch.lng,
                    "coords": patch.patch_coords
                })
            
            with open(embeddings_path, "w") as f:
                json.dump(embeddings_data, f)
            
            # Store session metadata
            server.sessions[session_id] = SessionMetadata(
                session_id=session_id,
                created_at=session_data.created_at,
                map_path=map_path,
                embeddings_path=embeddings_path,
                zip_path=""  # Will be set when zip is created
            )
            server.save_sessions()
        
        if result.get("success"):
            update_progress(80, "Packaging data...")
            
            # Convert zip_data to base64 for JSON response
            if "zip_data" in result:
                import base64
                task.zip_data = base64.b64encode(result["zip_data"]).decode('utf-8')
                task.session_id = result["session_id"]
            
            update_progress(100, "Map data ready!")
            task.status = "completed"
            
            # Send final WebSocket notification with completion data
            await manager.send_progress(task_id, {
                "status": "completed",
                "progress": 100,
                "message": "Map data ready!",
                "task_id": task_id,
                "zip_data": task.zip_data,
                "session_id": task.session_id
            })
        else:
            task.status = "failed"
            task.error = result.get("error", "Unknown error")
            task.message = f"Failed: {task.error}"
    
    except Exception as e:
        task.status = "failed"
        task.error = str(e)
        task.message = f"Error: {str(e)}"
        print(f"Error in background task {task_id}: {e}")
    
    # Clean up after some time
    await asyncio.sleep(10)
    if task_id in background_tasks:
        del background_tasks[task_id]


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


@app.post("/fetch_logs")
async def http_fetch_logs(request: FetchLogsRequest):
    """Fetch logs for a session as a zip file."""
    try:
        zip_data = package_session_logs(request.session_id, request.logger_id)
        
        if zip_data is None:
            raise HTTPException(status_code=404, detail={
                "success": False,
                "error": f"No logs found for session {request.session_id}" + 
                        (f" and logger {request.logger_id}" if request.logger_id else "")
            })
        
        # Generate filename
        if request.logger_id:
            filename = f"logs_{request.session_id}_{request.logger_id}.zip"
        else:
            filename = f"logs_{request.session_id}_all.zip"
        
        from fastapi.responses import Response
        return Response(
            content=zip_data,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e)
        })


@app.get("/available_logs", response_model=AvailableLogsResponse)
async def http_available_logs():
    """Get information about available session logs."""
    try:
        logs_info = get_available_sessions()
        
        return AvailableLogsResponse(
            success=True,
            sessions=[LogsSessionInfo(**session) for session in logs_info["sessions"]],
            total_sessions=logs_info["total_sessions"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "sessions": [],
            "total_sessions": 0
        })


@app.get("/logs_summary/{session_id}", response_model=LogsSummaryResponse)
async def http_logs_summary(session_id: str):
    """Get detailed summary of logs for a specific session."""
    try:
        summary = get_session_logs_summary(session_id)
        
        if summary is None:
            return LogsSummaryResponse(
                success=False,
                message=f"No logs found for session {session_id}"
            )
        
        return LogsSummaryResponse(
            success=True,
            **summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail={
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