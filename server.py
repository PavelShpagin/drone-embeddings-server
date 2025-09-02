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
from fastapi.middleware.cors import CORSMiddleware
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

def clean_message_for_production(message: str) -> str:
    """Clean up technical messages for production with smooth progress."""
    # Remove technical junk like arrows, brackets, etc.
    clean_msg = message.replace("==>", "").replace("->", "").replace(">>", "")
    clean_msg = clean_msg.replace("[", "").replace("]", "").strip()
    
    # Map technical terms to clean user-friendly messages (no counters)
    if "connection" in clean_msg.lower() or "connecting" in clean_msg.lower():
        return "Connecting to satellite services..."
    elif "downloading" in clean_msg.lower() and ("tiles" in clean_msg.lower() or "imagery" in clean_msg.lower()):
        return "Downloading satellite imagery..."
    elif "stitching" in clean_msg.lower():
        return "Processing satellite data..."
    elif "generating embeddings" in clean_msg.lower() or "embedding" in clean_msg.lower() or "processing tile" in clean_msg.lower():
        return "Generating AI embeddings..."
    elif "processing" in clean_msg.lower() and "tile" in clean_msg.lower():
        return "Generating AI embeddings..."
    elif "packaging" in clean_msg.lower() or "compressing" in clean_msg.lower():
        return "Packaging mission data..."
    elif "session" in clean_msg.lower() and "created" in clean_msg.lower():
        return "Finalizing mission data..."
    elif "finaliz" in clean_msg.lower() or "complet" in clean_msg.lower():
        return "Mission complete!"
    else:
        # For other messages, clean them up but keep the essence
        return clean_msg if clean_msg else "Processing..."

def init_map(lat: float, lng: float, meters: int = 2000, mode: str = "server", session_id: Optional[str] = None, fetch_only: bool = False) -> Dict[str, Any]:
    """Wrapper function for backward compatibility."""
    return server.init_map(lat, lng, meters, mode, session_id, fetch_only)


# FastAPI HTTP Server
app = FastAPI(
    title="Satellite Embedding Server",
    description="Process satellite imagery with DINOv2 embeddings for GPS localization",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests from device server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8888", "http://127.0.0.1:8888"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.websocket("/ws/hello")
async def hello_websocket(websocket: WebSocket):
    """Hello World WebSocket endpoint for testing."""
    await websocket.accept()
    print(f"✅ Hello World WebSocket connected")
    
    counter = 0
    try:
        while True:
            counter += 1
            await websocket.send_json({
                "type": "hello",
                "message": f"Hello World {counter}",
                "counter": counter
            })
            await asyncio.sleep(1)  # Send every second
    except WebSocketDisconnect:
        print(f"❌ Hello World WebSocket disconnected")

@app.websocket("/ws/mission")
async def mission_websocket(websocket: WebSocket):
    """Mission WebSocket endpoint for real-time progress updates."""
    await websocket.accept()
    print(f"✅ Mission WebSocket connected")
    
    try:
        # Wait for mission parameters
        data = await websocket.receive_json()
        print(f"📨 Received mission request: {data}")
        
        if data.get('type') == 'init_map':
            lat = data.get('lat')
            lng = data.get('lng')
            km = data.get('km')
            
            if not all([lat, lng, km]):
                await websocket.send_json({
                    "type": "error",
                    "message": "Missing required parameters: lat, lng, km"
                })
                return
            
            # Start the mission and stream progress
            await stream_mission_progress(websocket, lat, lng, km)
        else:
            await websocket.send_json({
                "type": "error",
                "message": f"Unknown mission type: {data.get('type')}"
            })
            
    except WebSocketDisconnect:
        print(f"❌ Mission WebSocket disconnected")
    except Exception as e:
        print(f"❌ Mission WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Mission failed: {str(e)}"
            })
        except:
            pass

@app.websocket("/ws/fetch")
async def fetch_websocket(websocket: WebSocket):
    """WebSocket endpoint for fetching cached embeddings/data by session ID."""
    await websocket.accept()
    print(f"✅ Fetch WebSocket connected")
    
    try:
        # Wait for fetch request
        data = await websocket.receive_json()
        print(f"📨 Received fetch request: {data}")
        
        if data.get("type") == "fetch_cached":
            session_id = data.get("session_id")
            
            if not session_id or len(session_id) != 8:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid 8-character session ID required"
                })
                return
            
            # Stream fetch progress
            await stream_fetch_progress(websocket, session_id)
        else:
            await websocket.send_json({
                "type": "error",
                "message": f"Unknown fetch type: {data.get('type')}"
            })
            
    except WebSocketDisconnect:
        print(f"❌ Fetch WebSocket disconnected")
    except Exception as e:
        print(f"❌ Fetch WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Fetch failed: {str(e)}"
            })
        except:
            pass

async def stream_fetch_progress(websocket: WebSocket, session_id: str):
    """Stream progress updates for fetching cached data."""
    print(f"🔍 Fetching cached data for session: {session_id}")
    
    try:
        # Send initial progress
        await websocket.send_json({
            "type": "progress",
            "message": "Searching for cached session...",
            "progress": 10.0,
            "session_id": session_id,
            "status": "running"
        })
        
        # Check if session exists in server.sessions
        if session_id not in server.sessions:
            await websocket.send_json({
                "type": "error",
                "message": f"Session {session_id} not found in cache",
                "progress": 0.0,
                "session_id": session_id,
                "status": "error"
            })
            return
        
        session_metadata = server.sessions[session_id]
        
        # Progress: Found session
        await websocket.send_json({
            "type": "progress",
            "message": "Found cached session! Loading data...",
            "progress": 30.0,
            "session_id": session_id,
            "status": "running"
        })
        
        # Read the cached files and create zip
        await asyncio.sleep(0.5)  # Simulate loading time
        
        await websocket.send_json({
            "type": "progress",
            "message": "Packaging cached data...",
            "progress": 60.0,
            "session_id": session_id,
            "status": "running"
        })
        
        # Create zip with cached data
        import zipfile
        import io
        import base64
        import os
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add map file if exists
            if hasattr(session_metadata, 'map_path') and os.path.exists(session_metadata.map_path):
                zip_file.write(session_metadata.map_path, 'map.png')
            
            # Add embeddings file if exists
            if hasattr(session_metadata, 'embeddings_path') and os.path.exists(session_metadata.embeddings_path):
                zip_file.write(session_metadata.embeddings_path, 'embeddings.json')
        
        # Save zip file temporarily for download
        temp_zip_path = f"data/temp/{session_id}_cached.zip"
        os.makedirs("data/temp", exist_ok=True)
        with open(temp_zip_path, 'wb') as f:
            f.write(zip_buffer.getvalue())
        
        zip_size = len(zip_buffer.getvalue())
        
        # Send completion with download URL instead of large data
        await websocket.send_json({
            "type": "progress",
            "message": "Cached data ready!",
            "progress": 100.0,
            "session_id": session_id,
            "status": "complete",
            "download_url": f"/download_cached/{session_id}",
            "zip_size": zip_size,
            # Include stored lat/lng/km so device can update state
            "lat": getattr(session_metadata, 'lat', None),
            "lng": getattr(session_metadata, 'lng', None),
            "km": getattr(session_metadata, 'km', None)
        })
        
        print(f"✅ Cached data sent for session: {session_id}")
        
    except Exception as e:
        print(f"❌ Fetch progress error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Failed to fetch cached data: {str(e)}",
            "progress": 0.0,
            "session_id": session_id,
            "status": "error"
        })

@app.websocket("/ws/logs")
async def logs_websocket(websocket: WebSocket):
    """WebSocket endpoint for uploading logs from device to server."""
    await websocket.accept()
    print(f"✅ Logs WebSocket connected")
    
    try:
        # Wait for logs upload request
        data = await websocket.receive_json()
        print(f"📨 Received logs request: {data.get('type')}")
        
        if data.get("type") == "upload_logs":
            logs_data = data.get("logs_data")
            
            if not logs_data:
                await websocket.send_json({
                    "type": "error",
                    "message": "No logs data provided"
                })
                return
            
            # Stream logs upload progress
            await stream_logs_upload(websocket, logs_data)
        else:
            await websocket.send_json({
                "type": "error",
                "message": f"Unknown logs type: {data.get('type')}"
            })
            
    except WebSocketDisconnect:
        print(f"❌ Logs WebSocket disconnected")
    except Exception as e:
        print(f"❌ Logs WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Logs upload failed: {str(e)}"
            })
        except:
            pass

async def stream_logs_upload(websocket: WebSocket, logs_data: str):
    """Stream progress updates for uploading logs."""
    print(f"📤 Uploading logs data...")
    
    try:
        # Send initial progress
        await websocket.send_json({
            "type": "progress",
            "message": "Receiving logs data...",
            "progress": 20.0,
            "status": "running"
        })
        
        # Decode and save logs
        import base64
        import zipfile
        import io
        import os
        from datetime import datetime
        
        await asyncio.sleep(0.3)  # Simulate processing time
        
        await websocket.send_json({
            "type": "progress",
            "message": "Processing logs archive...",
            "progress": 50.0,
            "status": "running"
        })
        
        # Decode the base64 logs data
        logs_zip_data = base64.b64decode(logs_data)
        
        # Create logs directory if it doesn't exist
        logs_dir = "data/device_logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logs_filename = f"device_logs_{timestamp}.zip"
        logs_path = os.path.join(logs_dir, logs_filename)
        
        await websocket.send_json({
            "type": "progress",
            "message": "Saving logs to server...",
            "progress": 80.0,
            "status": "running"
        })
        
        # Write logs file
        with open(logs_path, 'wb') as f:
            f.write(logs_zip_data)
        
        file_size = len(logs_zip_data)
        print(f"✅ Logs saved: {logs_path} ({file_size} bytes)")
        
        await websocket.send_json({
            "type": "progress",
            "message": f"Logs uploaded successfully! ({file_size} bytes)",
            "progress": 100.0,
            "status": "complete",
            "logs_filename": logs_filename
        })
        
    except Exception as e:
        print(f"❌ Logs upload error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Failed to upload logs: {str(e)}",
            "progress": 0.0,
            "status": "error"
        })

async def stream_mission_progress(websocket: WebSocket, lat: float, lng: float, km: float):
    """Stream real-time progress updates for a mission using actual data processing."""
    import uuid
    import time
    
    task_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Create a background task for real data processing
        task = ProgressTask(task_id)
        background_tasks[task_id] = task
        
        # Custom progress callback that sends updates via WebSocket
        async def websocket_progress_callback(progress: float, message: str):
            await websocket.send_json({
                "type": "progress",
                "message": message,
                "progress": round(progress, 1),  # Round to 1 decimal for clean display
                "session_id": getattr(task, 'session_id', task_id[:8]),
                "status": "complete" if progress >= 100 else "running"
            })
        
        # Override the task's progress update to use our WebSocket callback
        original_update = task.update_progress if hasattr(task, 'update_progress') else None
        
        def update_progress_wrapper(progress: float, message: str):
            task.progress = progress
            task.message = message
            # Send via WebSocket
            asyncio.create_task(websocket_progress_callback(progress, message))
        
        # Start real data processing
        meters = int(km * 1000)  # Convert km to meters
        
        # Send initial progress
        await websocket_progress_callback(0, "Initializing satellite data request...")
        
        # Run the actual data processing pipeline
        loop = asyncio.get_running_loop()
        
        # Smooth progress manager to prevent jumps and flickering
        class SmoothProgressManager:
            def __init__(self):
                self.last_progress = 0
                self.last_message = ""
                self.progress_history = []
                
            async def send_progress(self, task_id: str, progress_data: dict):
                progress = float(progress_data.get('progress', 0))
                message = progress_data.get('message', 'Processing...')
                
                # CRITICAL: Prevent backwards progress jumps
                if progress < self.last_progress - 0.1 and progress < 95:
                    # Skip backwards jumps unless it's a completion reset
                    return
                
                # Skip 0% "Processing..." messages that interrupt real progress
                if progress == 0 and message.strip() == 'Processing...' and self.last_progress > 5:
                    return
                
                # Clean the message for production
                clean_msg = clean_message_for_production(message)
                
                # Only send if progress increased or message meaningfully changed
                # Finer granularity for tile processing
                is_tile_msg = "tile" in message.lower() or "completed" in message.lower()
                min_progress_step = 0.01 if is_tile_msg else 0.1
                progress_increased = progress > self.last_progress + min_progress_step
                message_changed = clean_msg != self.last_message and clean_msg != "Processing..."
                
                if progress_increased or message_changed:
                    # Smooth out progress to prevent jumps
                    smooth_progress = max(self.last_progress, progress)
                    
                    await websocket_progress_callback(smooth_progress, clean_msg)
                    self.last_progress = smooth_progress
                    self.last_message = clean_msg
                    
                    # Track progress history for debugging
                    self.progress_history.append((smooth_progress, clean_msg))
                    if len(self.progress_history) > 10:
                        self.progress_history.pop(0)
        
        # Set up direct WebSocket progress callback
        async def websocket_progress_callback(progress: float, message: str):
            """Send progress updates directly to WebSocket."""
            try:
                progress_data = {
                    "type": "progress",
                    "progress": float(progress),
                    "message": clean_message_for_production(message),
                    "status": "running"  # Never mark as complete here, let the completion handler do it
                }
                await websocket.send_json(progress_data)
                print(f"📡 WebSocket progress: {progress}% - {message}")
            except Exception as e:
                print(f"❌ WebSocket progress error: {e}")
        
        # Store the callback globally for the task
        globals()[f'websocket_callback_{task_id}'] = websocket_progress_callback
        
        try:
            await _process_init_map_async(
                task_id=task_id, 
                lat=lat, 
                lng=lng, 
                meters=meters, 
                session_id=None,  # Always None to force new session creation
                fetch_only=False
            )
            
            # Wait a moment for the last progress update to be processed
            await asyncio.sleep(0.5)
            
            # Wait for task completion with proper session_id handling
            max_wait = 20  # Wait up to 20 seconds for completion
            for i in range(max_wait):
                if task.status == "completed":
                    break
                # Also check if we have zip_data (processing might be done even if status isn't updated)
                if hasattr(task, 'zip_data') and task.zip_data:
                    print(f"🔍 Found zip_data, waiting for session_id...")
                    # Wait longer for session_id to be set properly
                    for j in range(10):  # Wait up to 10 more seconds for session_id
                        if hasattr(task, 'session_id') and task.session_id:
                            print(f"✅ Found session_id: {task.session_id}")
                            break
                        await asyncio.sleep(1.0)
                    break
                await asyncio.sleep(1.0)
            
            print(f"🔍 Task status: {task.status}, has zip_data: {hasattr(task, 'zip_data') and bool(getattr(task, 'zip_data', None))}, session_id: {getattr(task, 'session_id', 'NOT SET')}")

            # Always try to emit a completion frame - with fallback session selection
            try:
                session_id_to_use = getattr(task, 'session_id', None)
                zip_b64 = getattr(task, 'zip_data', None)

                # If we have zip_data but no session_id, try to find the most recent session
                if zip_b64 and not session_id_to_use:
                    print("🔍 Have zip_data but no session_id, finding most recent session...")
                    if hasattr(task, 'request_info'):
                        # Find session matching our request parameters
                        req_info = task.request_info
                        for sid, session in server.sessions.items():
                            if (abs(session.lat - req_info['lat']) < 0.001 and 
                                abs(session.lng - req_info['lng']) < 0.001 and
                                abs(session.km - req_info['km']) < 0.1):
                                session_id_to_use = sid
                                print(f"✅ Found matching session: {session_id_to_use}")
                                break
                    
                    # Fallback: use the most recent session
                    if not session_id_to_use and server.sessions:
                        session_id_to_use = max(server.sessions.keys(), key=lambda k: server.sessions[k].created_at)
                        print(f"🔄 Using most recent session as fallback: {session_id_to_use}")

                if zip_b64 and session_id_to_use:
                    # Force task status to completed if we have valid data
                    task.status = "completed"
                    task.session_id = session_id_to_use
                    
                    # Check zip data size to avoid WebSocket frame limit
                    zip_size = len(zip_b64) if zip_b64 else 0
                    print(f"📦 Zip data size: {zip_size} bytes")
                    
                    if zip_size > 800000:  # 800KB threshold to stay under 1MB limit
                        # Save zip data to temp file and provide download URL
                        temp_zip_path = f"data/temp/{session_id_to_use}_mission.zip"
                        import os
                        os.makedirs(os.path.dirname(temp_zip_path), exist_ok=True)
                        
                        # Decode base64 and save to file
                        import base64
                        zip_bytes = base64.b64decode(zip_b64)
                        with open(temp_zip_path, 'wb') as f:
                            f.write(zip_bytes)
                        
                        print(f"💾 Large zip saved to temp file: {temp_zip_path}")
                        
                        completion_data = {
                            "type": "progress",
                            "message": "Mission complete!",
                            "progress": 100.0,
                            "session_id": session_id_to_use,
                            "status": "complete",
                            "download_url": f"/download_mission/{session_id_to_use}",
                            "zip_size": zip_size
                        }
                    else:
                        # Small zip, send directly
                        completion_data = {
                            "type": "progress",
                            "message": "Mission complete!",
                            "progress": 100.0,
                            "session_id": session_id_to_use,
                            "status": "complete",
                            "zip_data": zip_b64
                        }
                    
                    try:
                        await websocket.send_json(completion_data)
                        print("✅ Completion message sent successfully (packaged)")
                        await asyncio.sleep(2.0)
                    except Exception as e:
                        print(f"❌ Failed to send completion data: {e}")
                else:
                    print(f"⚠️ Unable to emit completion yet (zip_data: {bool(zip_b64)}, session_id: {session_id_to_use})")
            except Exception as e:
                print(f"❌ Error during completion packaging: {e}")
            
            return
        except Exception as e:
            print(f"❌ Async processing failed: {e}")
        finally:
            # Clean up the callback
            callback_key = f'websocket_callback_{task_id}'
            if callback_key in globals():
                del globals()[callback_key]
        
        def run_real_processing():
            """Fallback sync processing."""
            try:
                result = server.init_map(
                    lat=lat, lng=lng, meters=meters, mode="device", 
                    session_id=None, fetch_only=False
                )
                
                if result.get("success"):
                    # Store the result data in the task
                    if "zip_data" in result:
                        import base64
                        task.zip_data = base64.b64encode(result["zip_data"]).decode('utf-8')
                        task.session_id = result.get("session_id", task_id[:8])
                    
                    task.status = "completed"
                    # Send completion via WebSocket with data
                    completion_data = {
                        "type": "progress",
                        "message": "Mission complete! (100%)",
                        "progress": 100.0,
                        "session_id": task.session_id,
                        "status": "complete",
                        "zip_data": task.zip_data
                    }
                    print(f"✅ Mission completed: session={task.session_id}, data_size={len(task.zip_data) if task.zip_data else 0} chars")
                    
                    # Send completion message synchronously to ensure it's delivered
                    future = asyncio.run_coroutine_threadsafe(
                        websocket.send_json(completion_data),
                        loop
                    )
                    # Wait for the message to be sent
                    try:
                        future.result(timeout=5.0)
                        print(f"✅ Completion data transmitted successfully")
                    except Exception as e:
                        print(f"❌ Failed to send completion data: {e}")
                else:
                    task.status = "error"
                    task.error = result.get("error", "Processing failed")
                    asyncio.run_coroutine_threadsafe(
                        websocket.send_json({
                            "type": "error",
                            "message": result.get("error", "Processing failed")
                        }),
                        loop
                    )
                    
            except Exception as e:
                task.status = "error"
                task.error = str(e)
                asyncio.run_coroutine_threadsafe(
                    websocket.send_json({
                        "type": "error",
                        "message": f"Processing failed: {str(e)}"
                    }),
                    loop
                )
        
        # Run processing in thread pool to avoid blocking
        await loop.run_in_executor(None, run_real_processing)
        
        # Wait for completion or error
        while task.status not in ["completed", "error"]:
            await asyncio.sleep(0.5)
        
        if task.status == "completed":
            print(f"✅ Mission {task_id} completed successfully with real data")
        else:
            print(f"❌ Mission {task_id} failed: {task.error}")
        
        # Cleanup
        if task_id in background_tasks:
            del background_tasks[task_id]
        
    except WebSocketDisconnect:
        print(f"❌ Mission WebSocket disconnected during progress streaming")
        if task_id in background_tasks:
            del background_tasks[task_id]
    except Exception as e:
        print(f"❌ Error streaming mission progress: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Mission failed: {str(e)}"
            })
        except:
            pass
        if task_id in background_tasks:
            del background_tasks[task_id]

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
        
        # Keep connection alive and listen for client messages - NO TIMEOUT for long operations
        while True:
            try:
                # Wait for client messages (cancellation, init_map requests, etc.) - NO TIMEOUT
                message = await websocket.receive_text()  # Remove timeout completely for long operations
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
                    
                    elif message_type == "init_map":
                        # Handle init_map request via WebSocket
                        print(f"📡 Received init_map request via WebSocket from {connection_id}")
                        
                        # Extract parameters from WebSocket message
                        lat = float(data.get("lat"))
                        lng = float(data.get("lng"))
                        meters = int(data.get("meters"))
                        # Use None when no session_id is provided so a new session is created
                        session_id = data.get("session_id") or None
                        fetch_only = data.get("fetch_only", False)
                        
                        # Create task
                        task_id = str(uuid.uuid4())
                        task = ProgressTask(task_id)
                        background_tasks[task_id] = task
                        
                        # Register task with this WebSocket connection
                        manager.register_task(task_id, connection_id)
                        
                        # Send task started confirmation
                        await websocket.send_text(json.dumps({
                            "type": "task_started",
                            "task_id": task_id,
                            "message": "Map initialization started"
                        }))
                        
                        # Start background processing on the event loop
                        asyncio.create_task(
                            _process_init_map_async(
                                task_id=task_id,
                                lat=lat,
                                lng=lng,
                                meters=meters,
                                session_id=session_id,
                                fetch_only=fetch_only
                            )
                        )
                        
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


# HTTP init_map endpoint removed - using pure WebSocket communication only


# HTTP progress polling removed - using pure WebSocket communication only


async def _process_init_map_async(task_id: str, lat: float, lng: float, meters: int, session_id: Optional[str] = None, fetch_only: bool = False):
    """Background task for processing init_map requests with progress updates."""
    task = background_tasks[task_id]
    
    loop = asyncio.get_running_loop()

    # Smooth progress tracker to prevent backwards jumps
    smooth_progress_tracker = {"last_progress": 0, "last_message": ""}
    # Track request parameters for completion fallback selection
    task_request_info = {"lat": lat, "lng": lng, "km": float(meters) / 1000.0}
    setattr(background_tasks[task_id], "request_info", task_request_info)
    
    def update_progress(progress: float, message: str):
        """Update task progress with smooth progress (no backwards jumps)."""
        # Normalize parameter order
        try:
            normalized_progress = float(progress)
            normalized_message = str(message)
        except (TypeError, ValueError):
            try:
                normalized_progress = float(message)
                normalized_message = str(progress)
            except (TypeError, ValueError):
                normalized_progress = 0.0
                normalized_message = str(message)

        # CRITICAL: Implement smooth progress to prevent backwards jumps
        # Skip 0% "Processing..." messages that interrupt real progress
        if normalized_progress == 0.0 and normalized_message.strip() == "Processing..." and smooth_progress_tracker["last_progress"] > 5:
            return  # Skip this update
        
        # Prevent backwards jumps (unless it's completion reset)
        if normalized_progress < smooth_progress_tracker["last_progress"] - 0.5 and normalized_progress < 95:
            return  # Skip backwards jumps
        
        # Clean the message for production
        clean_msg = clean_message_for_production(normalized_message)
        
        # Determine granularity based on phase
        lower_msg = normalized_message.lower() if isinstance(normalized_message, str) else ""
        is_tile = ("tile" in lower_msg) or ("processing tile" in lower_msg) or ("completed" in lower_msg and "/" in lower_msg)
        is_embedding = ("embedding" in lower_msg)
        # Very small steps for tiles to show continuous progress, modest for embeddings, larger for milestones
        min_step = 0.001 if is_tile else (0.01 if is_embedding else 0.1)

        # Only send if progress increased meaningfully or message changed
        progress_increased = normalized_progress > smooth_progress_tracker["last_progress"] + min_step
        message_changed = clean_msg != smooth_progress_tracker["last_message"] and clean_msg != "Processing..."
        
        if progress_increased or message_changed:
            # Use the higher of current or last progress for smoothness
            smooth_progress = max(smooth_progress_tracker["last_progress"], normalized_progress)
            
            task.progress = smooth_progress
            task.message = clean_msg
            print(f"Task {task_id}: {smooth_progress:.1f}% - {clean_msg}")
            
            # Update tracker
            smooth_progress_tracker["last_progress"] = smooth_progress
            smooth_progress_tracker["last_message"] = clean_msg
            
            # Send progress via WebSocket callback if available
            callback = globals().get(f'websocket_callback_{task_id}')
            if callback:
                def _send():
                    asyncio.create_task(callback(smooth_progress, clean_msg))
                try:
                    loop.call_soon_threadsafe(_send)
                except RuntimeError:
                    try:
                        asyncio.run_coroutine_threadsafe(callback(smooth_progress, clean_msg), loop)
                    except Exception as e:
                        print(f"Failed to send WebSocket update: {e}")
    
    try:
        update_progress(5, "Initializing satellite data request...")
        
        # Use the server's init_map logic to handle session validation
        # Adapter so downstream expects (message, progress) but our update_progress is (progress, message)
        initial_result = server.init_map(
            lat=lat, lng=lng, meters=meters, mode="device", 
            session_id=session_id, fetch_only=fetch_only,
            progress_callback=(lambda m, p: update_progress(p, m))
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
        from src.server.init_map import process_init_map_request
        from src.general.models import SessionMetadata
        import json
        temp_sessions = {}
        
        def progress_wrapper(message, progress):
            """Wrapper to make progress callback async-compatible."""
            # Cooperative cancellation: abort as soon as server marks task cancelled
            if task.status == "cancelled":
                print(f"Cancellation detected for task {task_id}; aborting work")
                raise OperationCancelled()
            print(f"PROGRESS WRAPPER CALLED: {progress}% - {message}")
            # Adapter: ensure update_progress(progress, message)
            update_progress(float(progress), message)
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
            print(f"🔍 Processing successful result: session_id={session_id}, has_zip_data={'zip_data' in result}")
            
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
            
            # Store session metadata with original parameters
            # Extract lat/lng from map_bounds center and convert meters to km
            bounds = session_data.map_bounds
            center_lat = (bounds["min_lat"] + bounds["max_lat"]) / 2
            center_lng = (bounds["min_lng"] + bounds["max_lng"]) / 2
            coverage_km = session_data.meters_coverage / 1000.0
            
            server.sessions[session_id] = SessionMetadata(
                session_id=session_id,
                created_at=session_data.created_at,
                map_path=map_path,
                embeddings_path=embeddings_path,
                zip_path="",  # Will be set when zip is created
                lat=center_lat,
                lng=center_lng,
                km=coverage_km
            )
            server.save_sessions()
            
            # CRITICAL: Always set session_id from result BEFORE creating zip
            task.session_id = result["session_id"]
            print(f"✅ Set task.session_id = {task.session_id}")
            
            # Create zip file with map and embeddings
            update_progress(80, "Packaging mission data...")
            import zipfile
            import io
            import base64
            
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.write(map_path, 'map.png')
                zip_file.write(embeddings_path, 'embeddings.json')
            
            task.zip_data = base64.b64encode(zip_buffer.getvalue()).decode('utf-8')
            print(f"📦 Created zip_data for session {task.session_id}, size: {len(task.zip_data)} chars")
        
        if result.get("success"):
            task.status = "completed"
            update_progress(100, "Mission complete!")
            print(f"✅ Task {task_id} completed successfully with session_id: {getattr(task, 'session_id', 'NOT SET')}")
        else:
            task.status = "failed"
            task.error = result.get("error", "Unknown error")
            task.message = f"Failed: {task.error}"
            print(f"❌ Task {task_id} failed: {task.error}")
    
    except Exception as e:
        import traceback
        task.status = "failed"
        task.error = str(e)
        task.message = f"Error: {str(e)}"
        print(f"Error in background task {task_id}: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
    
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
                    
                    # Handle different embeddings file formats
                    if isinstance(embeddings_data, dict) and "patches" in embeddings_data:
                        # New structured format
                        session_data.append(SessionInfo(
                            session_id=session_id,
                            created_at=session.created_at,
                            meters_coverage=embeddings_data["meters_coverage"],
                            patch_count=len(embeddings_data["patches"]),
                            map_bounds=embeddings_data["map_bounds"]
                        ))
                    elif isinstance(embeddings_data, list):
                        # Old format - raw embedding arrays, use defaults
                        session_data.append(SessionInfo(
                            session_id=session_id,
                            created_at=session.created_at,
                            meters_coverage=1000,  # Default value
                            patch_count=len(embeddings_data),
                            map_bounds={"min_lat": 0, "max_lat": 0, "min_lng": 0, "max_lng": 0}  # Default
                        ))
                    else:
                        print(f"Unknown embeddings format for session {session_id}")
                        continue
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


@app.get("/download_cached/{session_id}")
async def download_cached(session_id: str):
    """Download cached session data as zip file."""
    import os
    try:
        temp_zip_path = f"data/temp/{session_id}_cached.zip"
        
        if not os.path.exists(temp_zip_path):
            raise HTTPException(status_code=404, detail="Cached data not found")
        
        from fastapi.responses import FileResponse
        return FileResponse(
            temp_zip_path,
            media_type="application/zip",
            filename=f"cached_data_{session_id}.zip"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download_mission/{session_id}")
async def download_mission(session_id: str):
    """Download mission session data as zip file."""
    import os
    try:
        temp_zip_path = f"data/temp/{session_id}_mission.zip"
        
        if not os.path.exists(temp_zip_path):
            raise HTTPException(status_code=404, detail="Mission data not found")
        
        from fastapi.responses import FileResponse
        return FileResponse(
            temp_zip_path,
            media_type="application/zip",
            filename=f"mission_data_{session_id}.zip"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cancel_task")
async def cancel_task(task_id: str = Form(...), connection_id: Optional[str] = Form(None)):
    """FORCEFULLY cancel a background task with immediate termination."""
    try:
        if task_id in background_tasks:
            task = background_tasks[task_id]
            
            # STRICT CANCELLATION: Mark as cancelled immediately
            task.status = "cancelled"
            task.message = "Task forcefully cancelled by client request"
            print(f"🚫 FORCEFUL CANCELLATION: Task {task_id} marked for immediate termination")
            
            # If connection_id provided, also simulate disconnect for that connection
            if connection_id:
                manager.disconnect(connection_id)
                print(f"🔌 Simulated disconnect for connection {connection_id}")
            
            # Force immediate cleanup - don't wait for graceful completion
            task.progress = 0
            task.message = "Cancelled"
            print(f"⚡ Task {task_id} status updated for immediate abort")
            
            return {"success": True, "message": f"Task {task_id} forcefully cancelled"}
        else:
            return {"success": False, "message": f"Task {task_id} not found"}
            
    except Exception as e:
        return {"success": False, "message": f"Error cancelling task: {e}"}


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