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
from typing import Dict, Any, Optional, Set

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
        self.progress = 0.0  # 0-100, allow fractional progress for smooth UI updates
        self.message = "Starting..."
        self.zip_data = None
        self.session_id = None
        self.error = None
        self.created_at = time.time()
        self.last_polled_at = self.created_at
        
class ProgressResponse(BaseModel):
    status: str
    progress: float
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

# Custom exception for cooperative cancellation in worker threads
class OperationCancelled(Exception):
    pass

# WebSocket connection management
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.task_connections: Dict[str, str] = {}  # task_id -> connection_id mapping
        self.cancelled_connections: Set[str] = set()
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        print(f"✓ WebSocket connected: {connection_id}")
    
    def disconnect(self, connection_id: str):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        # Cancel all tasks associated with this connection
        cancelled_tasks = []
        for task_id, conn_id in list(self.task_connections.items()):
            if conn_id == connection_id:
                # Cancel the background task
                if task_id in background_tasks:
                    task = background_tasks[task_id]
                    task.status = "cancelled"
                    task.message = "Connection lost - task cancelled"
                    cancelled_tasks.append(task_id)
                    print(f"🚫 Cancelled task {task_id} due to connection loss")
                # Remove from task mapping
                del self.task_connections[task_id]
        
        if cancelled_tasks:
            print(f"✗ WebSocket disconnected: {connection_id} (cancelled {len(cancelled_tasks)} tasks)")
        else:
            print(f"✗ WebSocket disconnected: {connection_id}")
    
    def register_task(self, task_id: str, connection_id: str):
        """Register a task with a WebSocket connection."""
        self.task_connections[task_id] = connection_id
        print(f"📋 Task {task_id} registered with connection {connection_id}")
        # If this connection was previously cancelled, immediately cancel this task too
        if connection_id in self.cancelled_connections and task_id in background_tasks:
            t = background_tasks[task_id]
            t.status = "cancelled"
            t.message = "Task cancelled by client"
    
    async def send_progress(self, task_id: str, progress_data: dict):
        """Send progress update to the WebSocket connection for this task."""
        connection_id = self.task_connections.get(task_id)
        if connection_id and connection_id in self.active_connections:
            try:
                websocket = self.active_connections[connection_id]
                await websocket.send_text(json.dumps(progress_data))
                print(f"📡 Sent to {connection_id}: {progress_data['progress']}% - {progress_data['message']}")
                return True
            except Exception as e:
                print(f"❌ Error sending progress to {connection_id}: {e}")
                self.disconnect(connection_id)
                return False
        return False

manager = ConnectionManager()

def init_map(lat: float, lng: float, meters: int = 2000, mode: str = "server", session_id: Optional[str] = None, fetch_only: bool = False) -> Dict[str, Any]:
    """Wrapper function for backward compatibility."""
    return server.init_map(lat, lng, meters, mode, session_id, fetch_only)


# FastAPI HTTP Server
app = FastAPI(
    title="Satellite Embedding Server",
    description="Process satellite imagery with DINOv2 embeddings for GPS localization",
    version="1.0.0"
)


@app.websocket("/ws/{connection_id}")
async def websocket_endpoint(websocket: WebSocket, connection_id: str):
    """WebSocket endpoint for real-time progress updates."""
    try:
        await manager.connect(websocket, connection_id)
        
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection_confirmed",
            "connection_id": connection_id,
            "message": "WebSocket connected successfully"
        }))
        
        # Keep connection alive and listen for client messages
        while True:
            try:
                # Wait for client messages (cancellation, etc.)
                message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                try:
                    data = json.loads(message)
                    message_type = data.get("type")
                    
                    if message_type == "cancel_task":
                        # Handle task cancellation from device
                        task_id = data.get("task_id")
                        if task_id in background_tasks:
                            task = background_tasks[task_id]
                            task.status = "cancelled"
                            task.message = "Task cancelled by client"
                            
                            # Send cancellation confirmation
                            await websocket.send_text(json.dumps({
                                "type": "task_cancelled",
                                "task_id": task_id,
                                "message": "Task cancelled successfully"
                            }))
                            
                            print(f"🚫 Task {task_id} cancelled via WebSocket")
                    elif message_type == "cancel_all":
                        # Cancel all tasks registered to this connection
                        self.cancelled_connections.add(connection_id)
                        cancelled_any = False
                        for tid, conn_id in list(manager.task_connections.items()):
                            if conn_id == connection_id and tid in background_tasks:
                                t = background_tasks[tid]
                                t.status = "cancelled"
                                t.message = "Task cancelled by client"
                                cancelled_any = True
                        await websocket.send_text(json.dumps({
                            "type": "connection_cancelled",
                            "connection_id": connection_id,
                            "cancelled": cancelled_any
                        }))
                    
                    elif message_type == "register_task":
                        # Register task with this connection
                        task_id = data.get("task_id")
                        if task_id:
                            manager.register_task(task_id, connection_id)
                            await websocket.send_text(json.dumps({
                                "type": "task_registered",
                                "task_id": task_id,
                                "message": "Task registered for real-time updates"
                            }))
                        
                except json.JSONDecodeError:
                    print(f"Invalid JSON received: {message}")
                    
            except asyncio.TimeoutError:
                # No message received, continue
                continue
            except WebSocketDisconnect:
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(connection_id)


@app.post("/init_map")
async def http_init_map(
    lat: float = Form(...),
    lng: float = Form(...),
    meters: int = Form(2000),
    mode: str = Form("server"),
    session_id: Optional[str] = Form(None),
    connection_id: Optional[str] = Form(None),
    fetch_only: bool = Form(False),
    background_tasks_runner: BackgroundTasks = BackgroundTasks()
):
    """HTTP endpoint for initializing map sessions."""
    try:
        # Convert empty session_id to None
        if session_id == "":
            session_id = None
            
        # Handle device_async mode for progress tracking
        if mode == "device_async":
            task_id = str(uuid.uuid4())
            task = ProgressTask(task_id)
            background_tasks[task_id] = task
            
            # Register task with WebSocket connection if provided
            if connection_id:
                manager.register_task(task_id, connection_id)
            
            # Start background processing
            background_tasks_runner.add_task(
                _process_init_map_async,
                task_id=task_id,
                lat=lat,
                lng=lng,
                meters=meters,
                session_id=session_id,
                fetch_only=fetch_only
            )
            
            return AsyncInitResponse(task_id=task_id)
        
        # Handle regular modes (server, device)
        result = server.init_map(
            lat=lat,
            lng=lng,
            meters=meters,
            mode=mode,
            session_id=session_id,
            fetch_only=fetch_only
        )
        
        # Handle zip_data response for device mode
        if mode == "device" and "zip_data" in result:
            from fastapi.responses import Response
            return Response(
                content=result["zip_data"], 
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename=session_{result['session_id']}.zip"}
            )
        # Optionally return compressed payload for device mode
        # Remove invalid request branch (no request object here)
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
    # Mark as polled now to delay cleanup
    task.last_polled_at = time.time()
    
    return ProgressResponse(
        status=task.status,
        progress=task.progress,
        message=task.message,
        zip_data=task.zip_data,
        session_id=task.session_id,
        error=task.error
    )


async def _process_init_map_async(task_id: str, lat: float, lng: float, meters: int, session_id: Optional[str] = None, fetch_only: bool = False):
    """Background task for processing init_map requests with progress updates."""
    task = background_tasks[task_id]
    
    loop = asyncio.get_running_loop()

    def update_progress(progress: float, message: str):
        """Update task progress and send WebSocket notification."""
        task.progress = float(progress)
        task.message = message
        print(f"Task {task_id}: {progress}% - {message}")
        
        # Prepare basic progress data
        progress_payload = {
            "type": "progress_update",
            "task_id": task_id,
            "status": task.status,
            "progress": float(progress),
            "message": message
        }
        
        # Include zip_data and session_id if task is completed and has data
        if task.status == "completed" and hasattr(task, 'zip_data') and task.zip_data:
            progress_payload["zip_data"] = task.zip_data
            progress_payload["session_id"] = getattr(task, 'session_id', task_id)
            print(f"📦 Including zip_data in completion message for {task_id}")
        
        # Ensure WS send happens on the event loop thread even when called from worker threads
        def _send():
            asyncio.create_task(manager.send_progress(task_id, progress_payload))
        try:
            loop.call_soon_threadsafe(_send)
        except RuntimeError:
            # Fallback in case loop not available
            try:
                asyncio.run_coroutine_threadsafe(manager.send_progress(task_id, progress_payload), loop)
            except Exception as e:
                print(f"Failed to send WebSocket update: {e}")
    
    try:
        update_progress(5, "Initializing satellite data request...")
        
        # Use the server's init_map logic to handle session validation
        initial_result = server.init_map(
            lat=lat, lng=lng, meters=meters, mode="device", 
            session_id=session_id, fetch_only=fetch_only
        )
        
        # If server returned an error (session not found), propagate it
        if not initial_result.get("success"):
            update_progress(100, initial_result.get("error", "Session not found"))
            task.status = "error"
            task.error = initial_result.get("error", "Session not found")
            await asyncio.sleep(1)  # Brief delay for UI
            if task_id in background_tasks:
                del background_tasks[task_id]
            return
        
        # If we got a cached session successfully
        if initial_result.get("cached"):
            update_progress(30, "Found cached session, loading data...")
            await asyncio.sleep(1)  # Allow time for device to start polling
            
            update_progress(80, "Processing cached data...")
            await asyncio.sleep(1)
            
            # Convert zip_data to base64 for JSON response
            if "zip_data" in initial_result:
                import base64
                task.zip_data = base64.b64encode(initial_result["zip_data"]).decode('utf-8')
                task.session_id = initial_result["session_id"]
            
            task.status = "completed"
            update_progress(100, "Cached data ready!")
            
            # Keep task available for polling longer
            await asyncio.sleep(30)  # Increased delay to ensure device can fetch
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
            # Cooperative cancellation: abort as soon as server marks task cancelled
            if task.status == "cancelled":
                print(f"Cancellation detected for task {task_id}; aborting work")
                raise OperationCancelled()
            print(f"PROGRESS WRAPPER CALLED: {progress}% - {message}")
            update_progress(progress, message)
            task.progress = progress
            task.message = message
        
        # Run heavy synchronous processing in a worker thread so event loop can deliver WS updates
        def _run_processing():
            try:
                return process_init_map_request(
                    lat=lat,
                    lng=lng, 
                    meters=meters,
                    mode="device",
                    embedder=server.embedder,
                    sessions=temp_sessions,
                    save_sessions_callback=lambda: None,
                    progress_callback=progress_wrapper
                )
            except OperationCancelled:
                return {"success": False, "cancelled": True}
        result = await loop.run_in_executor(None, _run_processing)
        
        # If cancelled during processing, stop early
        if (isinstance(result, dict) and result.get("cancelled")) or task.status == "cancelled":
            task.status = "cancelled"
            update_progress(task.progress or 0, "Cancelled")
            return

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
            
            task.status = "completed"
            update_progress(100, "Map data ready!")
        else:
            task.status = "failed"
            task.error = result.get("error", "Unknown error")
            task.message = f"Failed: {task.error}"
    
    except Exception as e:
        task.status = "failed"
        task.error = str(e)
        task.message = f"Error: {str(e)}"
        print(f"Error in background task {task_id}: {e}")
    
    # Retain task for a bit longer to avoid 404s while device polls immediately after completion
    # Wait up to 60 seconds since last poll before deleting
    try:
        for _ in range(60):
            await asyncio.sleep(1)
            t = background_tasks.get(task_id)
            if not t:
                break
            # If polled within the last 5 seconds, extend retention
            if time.time() - (t.last_polled_at or t.created_at) < 5:
                continue
            # If not polled recently and task completed/failed, allow deletion
            if t.status in ("completed", "failed", "cancelled"):
                break
        if task_id in background_tasks:
            del background_tasks[task_id]
    except Exception:
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