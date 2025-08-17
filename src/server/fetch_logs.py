#!/usr/bin/env python3
"""
Log Fetching Functionality for Drone Embeddings Server
=====================================================
Provides functionality to package and download logs from the server.
"""

import zipfile
import tempfile
import shutil
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def package_session_logs(session_id: str, logger_id: Optional[str] = None) -> Optional[bytes]:
    """
    Package logs for a session (or specific logger) into a zip file.
    
    Args:
        session_id: Session ID to fetch logs for
        logger_id: Optional logger ID to fetch specific logs
        
    Returns:
        Zip file bytes or None if no logs found
    """
    logs_base_dir = Path("data/logs")
    session_logs_dir = logs_base_dir / session_id
    
    if not session_logs_dir.exists():
        print(f"No logs found for session: {session_id}")
        return None
    
    # Create temporary zip file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
        temp_zip_path = temp_file.name
    
    try:
        with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add session metadata
            metadata = {
                "session_id": session_id,
                "logger_id": logger_id,
                "export_time": datetime.now().isoformat(),
                "logs_structure": {}
            }
            
            if logger_id:
                # Package specific logger logs
                logger_dir = session_logs_dir / logger_id
                if logger_dir.exists():
                    _add_logger_to_zip(zf, logger_dir, logger_id, metadata)
                else:
                    print(f"No logs found for logger: {logger_id}")
                    return None
            else:
                # Package all loggers for this session
                for logger_dir in session_logs_dir.iterdir():
                    if logger_dir.is_dir():
                        _add_logger_to_zip(zf, logger_dir, logger_dir.name, metadata)
            
            # Add metadata file
            zf.writestr("logs_metadata.json", json.dumps(metadata, indent=2))
        
        # Read zip file and return bytes
        with open(temp_zip_path, 'rb') as f:
            zip_data = f.read()
        
        return zip_data
    
    finally:
        # Clean up temporary file
        try:
            Path(temp_zip_path).unlink()
        except:
            pass


def _add_logger_to_zip(zf: zipfile.ZipFile, logger_dir: Path, logger_id: str, metadata: Dict[str, Any]):
    """Add all files from a logger directory to the zip file."""
    files_added = []
    
    for file_path in logger_dir.rglob('*'):
        if file_path.is_file():
            # Create archive path: logger_id/filename
            archive_path = f"{logger_id}/{file_path.name}"
            zf.write(file_path, archive_path)
            files_added.append({
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "archive_path": archive_path
            })
    
    metadata["logs_structure"][logger_id] = {
        "files": files_added,
        "total_files": len(files_added)
    }


def get_available_sessions() -> Dict[str, Any]:
    """
    Get information about available session logs.
    
    Returns:
        Dictionary with session information
    """
    logs_base_dir = Path("data/logs")
    
    if not logs_base_dir.exists():
        return {"sessions": [], "total_sessions": 0}
    
    sessions_info = []
    
    for session_dir in logs_base_dir.iterdir():
        if session_dir.is_dir():
            session_id = session_dir.name
            loggers = []
            
            for logger_dir in session_dir.iterdir():
                if logger_dir.is_dir():
                    logger_id = logger_dir.name
                    files = list(logger_dir.rglob('*'))
                    files = [f for f in files if f.is_file()]
                    
                    loggers.append({
                        "logger_id": logger_id,
                        "file_count": len(files),
                        "files": [f.name for f in files]
                    })
            
            sessions_info.append({
                "session_id": session_id,
                "loggers": loggers,
                "logger_count": len(loggers)
            })
    
    return {
        "sessions": sessions_info,
        "total_sessions": len(sessions_info)
    }


def get_session_logs_summary(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed summary of logs for a specific session.
    
    Args:
        session_id: Session ID to get summary for
        
    Returns:
        Summary dictionary or None if session not found
    """
    logs_base_dir = Path("data/logs")
    session_logs_dir = logs_base_dir / session_id
    
    if not session_logs_dir.exists():
        return None
    
    summary = {
        "session_id": session_id,
        "loggers": {},
        "total_size": 0,
        "total_files": 0
    }
    
    for logger_dir in session_logs_dir.iterdir():
        if logger_dir.is_dir():
            logger_id = logger_dir.name
            logger_files = []
            logger_size = 0
            
            for file_path in logger_dir.rglob('*'):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    logger_files.append({
                        "filename": file_path.name,
                        "size": file_size,
                        "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    })
                    logger_size += file_size
            
            summary["loggers"][logger_id] = {
                "files": logger_files,
                "file_count": len(logger_files),
                "total_size": logger_size
            }
            summary["total_size"] += logger_size
            summary["total_files"] += len(logger_files)
    
    return summary
