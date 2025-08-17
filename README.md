# Drone Embeddings Server - New Architecture

Real-time GPS localization using DINOv2 embeddings and satellite imagery with enhanced file-based session management.

## Quick Start

### Server Setup

```bash
cd server
python server.py --port 5000 --debug
```

### Run Tests

```bash
cd server
# Test new zip-based device mode
python test/test_fetch_gps_device.py

# Test complete simulation with enhanced logging
python test/test_simulation.py

# Test against local server
python test/test_simulation.py --local
```

### Remote Testing (AWS)

```bash
# Test against remote AWS server
python test/test_simulation.py --remote
python test/test_fetch_gps_device.py --remote
```

## New Architecture Features

### 🔄 **Session Caching**

- `init_map` accepts optional `session_id` for instant cached retrieval
- Sessions persist across server restarts
- Lightweight metadata storage in `sessions.pkl`

### 📦 **Zip-based Distribution**

- Device mode returns compressed zip files containing maps + embeddings
- Reduced network transfer and atomic downloads
- Clean separation of concerns

### 📊 **Enhanced Logging**

- Per-session, per-logger directory structure: `logs/{session_id}/{logger_id}/`
- Real-time CSV tracking: ground truth vs predicted GPS
- Error plots with 50-frame rolling averages
- Map visualizations with ground truth (green) and predicted (red) paths

### 🗃️ **File-based Storage**

- Maps: `data/maps/{session_id}.png`
- Embeddings: `data/embeddings/{session_id}.json` (includes metadata)
- Zips: `data/zips/{session_id}.zip`
- Logs: `data/logs/{session_id}/{logger_id}/`

## API Endpoints

### Core Endpoints

- `POST /init_map` - Initialize map or return cached session (supports `session_id`)
- `POST /fetch_gps` - Enhanced GPS matching with optional logging (`logging_id`, `visualization`)
- `GET /sessions` - List all sessions with metadata
- `GET /health` - Server health check

### Legacy Endpoints (still supported)

- `POST /visualize_path` - Return current path visualization
- `POST /get_video` - Download path time-lapse video

## New Request Parameters

### init_map

```json
{
  "lat": 50.4162,
  "lng": 30.8906,
  "meters": 1000,
  "mode": "device", // "server" or "device"
  "session_id": "uuid" // Optional: return cached session
}
```

### fetch_gps

```json
{
  "session_id": "uuid",
  "logging_id": "logger123", // Optional: enhanced logging
  "visualization": true // Optional: enable visualizations
}
```

## Data Flow

### New Session Creation

1. **Init Map**: Creates satellite patches, embeddings, and stores as files
2. **File Storage**: Map → PNG, Embeddings → JSON, Package → ZIP
3. **Session Registry**: Lightweight metadata in `sessions.pkl`
4. **Response**: ZIP file (device) or success message (server)

### Cached Session Retrieval

1. **Cache Check**: Lookup session_id in `sessions.pkl`
2. **File Access**: Direct file serving from existing ZIP
3. **Response**: Cached ZIP file or success message

### Enhanced GPS Processing

1. **Image Processing**: Convert image → DINOv2 embedding
2. **Patch Matching**: Cosine similarity against cached embeddings
3. **Enhanced Logging**: CSV tracking, error plots, map visualization
4. **Response**: GPS + similarity + confidence metrics

## File Structure

```
data/
├── sessions.pkl           # Lightweight session metadata
├── maps/                  # Satellite images
│   └── {session_id}.png
├── embeddings/            # Patch embeddings + metadata
│   └── {session_id}.json
├── zips/                  # Packaged downloads
│   └── {session_id}.zip
└── logs/                  # Enhanced logging
    └── {session_id}/
        └── {logger_id}/
            ├── path.csv
            ├── error_plot.png
            └── map_paths.png
```

## Installation

```bash
# Core dependencies
pip install fastapi uvicorn torch torchvision numpy pillow earthengine-api

# Enhanced logging dependencies
pip install matplotlib pandas

# Optional: for development
pip install opencv-python
```

## Performance Improvements

- **Cold start**: ~20-30s (GEE fetch + embedding generation)
- **Cache hit**: ~200ms (direct file serving)
- **GPS matching**: ~100-300ms (depending on patch count)
- **Storage**: 90% reduction in `sessions.pkl` size

## Testing

All test scripts updated for new architecture:

```bash
# Comprehensive device mode testing
python test/test_fetch_gps_device.py

# Full simulation with enhanced logging
python test/test_simulation.py

# Custom test script
python test_new_server.py
```

## Migration Notes

- Legacy endpoints remain functional
- Existing sessions auto-migrate to file-based storage
- Enhanced logging is opt-in via `logging_id` parameter
- Backward compatibility maintained for all client integrations
