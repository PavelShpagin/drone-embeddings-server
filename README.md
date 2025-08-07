# Drone Embeddings Server

Real-time GPS localization server using DINOv2 embeddings and satellite imagery.

## Overview

The server provides GPS localization by matching drone camera images against pre-computed satellite image embeddings. The system supports both standalone server mode and device integration.

## Architecture Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client API    │───▶│  Server Session  │───▶│  GEE Sampler    │
│                 │    │  Hash Table      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   DINOv2 Model   │───▶│  Patch Storage  │
                       │  (384-dim embed) │    │  + GPS Tags     │
                       └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Persistent Store │
                       │ (data/sessions.pkl) │
                       └──────────────────┘
```

## Architecture

### Code Organization

```
server/
├── server.py               # Main FastAPI application
├── src/
│   ├── general/           # Shared modules (used by both server and device)
│   │   ├── models.py      # Data structures and Pydantic models
│   │   ├── fetch_gps.py   # GPS coordinate matching logic
│   │   ├── visualize_map.py # Path visualization
│   │   └── image_metadata.py # Image processing utilities
│   └── server/            # Server-specific modules
│       ├── server_core.py # Main server logic and session management
│       ├── init_map.py    # Map initialization and patch extraction
│       ├── gee_sampler.py # Google Earth Engine interface
│       └── embedder.py    # DINOv2 embedding model
├── data/                  # Data storage
│   ├── maps/             # Full satellite maps (session_id.pkl)
│   ├── embeddings/       # Patch embeddings (session_id.json)
│   ├── server_paths/     # Server-generated path visualizations
│   ├── client_paths/     # Client-downloaded path images
│   └── sessions.pkl      # Session metadata and path tracking
└── test/                 # Testing scripts
    ├── test_fetch_gps_device.py
    └── test_simulation.py
```

### Device Integration

```
device/
├── localizer.py          # TCP adapter for device-server communication
├── reader.cpp            # C++ image processing client
├── src/                  # Shared code (copied from server/src)
│   ├── general/          # Same as server/src/general
│   ├── server/           # Copy of server modules for embedder access
│   └── device/           # Device-specific modules
│       └── init_map_wrapper.py # HTTP client for server init_map
└── data/                 # Local device storage (same structure as server)
```

## API Endpoints

### `/init_map` (POST)

Initialize a new mapping session with satellite imagery.

**Request:**

```json
{
  "lat": 50.4162,
  "lng": 30.8906,
  "meters": 1000,
  "mode": "server" // or "device" for full data return
}
```

**Response:**

```json
{
  "success": true,
  "session_id": "74e654e3-79f3-46f0-aefc-0eaae66ea08e",
  "created_at": "2025-08-07T04:13:27.123456",
  "meters_coverage": 1000,
  "patches": 100,
  "map_bounds": {...}
}
```

### `/fetch_gps` (POST)

Find GPS coordinates from drone image.

**Request:** Multipart form with `session_id` and `image` file.

**Response:**

```json
{
  "success": true,
  "session_id": "...",
  "gps": { "lat": 50.414851, "lng": 30.884261 },
  "similarity": 0.8234,
  "confidence": "high",
  "patch_coords": [10, 20, 20, 30]
}
```

### `/visualize_path` (POST)

Get path visualization with GPS tracking.

**Request:**

```json
{
  "session_id": "74e654e3-79f3-46f0-aefc-0eaae66ea08e"
}
```

**Response:** JPEG image bytes with red dots and connecting lines.

### `/sessions` (GET)

List all active sessions.

### `/health` (GET)

Server health check.

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Google Earth Engine account

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install PyTorch CPU version (if no GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Set up Google Earth Engine credentials
source secrets/secret.sh  # Contains GEE service account key
```

### Running the Server

```bash
# From project root
python server/server.py --port 5000 --debug

# Or with specific host
python server/server.py --host 0.0.0.0 --port 5000
```

## Device Integration

The device submodule enables real-time GPS localization on edge devices:

### Architecture

1. **Device calls server** for `init_map` via HTTP to get session and map data
2. **Local processing** of `fetch_gps` using cached embeddings
3. **TCP communication** between C++ reader and Python localizer

### Device Setup

```bash
# Start device localizer (Python TCP server)
cd device
python localizer.py

# Start reader (C++ image processor)
./reader
```

### Device Workflow

1. `reader.cpp` sends `init_map` request to `localizer.py`
2. `localizer.py` calls server HTTP API, caches results locally
3. `reader.cpp` processes stream images, sends to `localizer.py` via TCP
4. `localizer.py` processes `fetch_gps` locally using cached embeddings
5. GPS coordinates logged to `data/reader.txt`

## Testing

### Server Testing

```bash
# Test device mode initialization and stream processing
cd server
python test/test_fetch_gps_device.py [--remote] [--port 5000]

# Test full simulation with path visualization
python test/test_simulation.py [--remote] [--port 5000]
```

### Remote Testing

Set environment variable for AWS testing:

```bash
export AWS_SERVER_DNS=your-server-dns.com
python test/test_simulation.py --remote
```

## Data Flow

### Server Mode

1. Client calls `/init_map` → Server generates map and embeddings
2. Client calls `/fetch_gps` → Server finds GPS and updates path visualization
3. Client calls `/visualize_path` → Server returns path image

### Device Mode

1. Device calls server `/init_map` → Downloads and caches map/embeddings locally
2. Device processes images locally using cached embeddings
3. GPS results logged locally, path visualization updated incrementally

## Performance Notes

- **Embedding Generation**: ~100 patches take 30-60 seconds on GPU
- **GPS Matching**: ~50ms per image with cached embeddings
- **Map Coverage**: 1000m uses 1x1 grid (minimal tiles), 2000m+ uses larger grids
- **TCP vs UDP**: Device uses TCP for reliable large data transfer

## Visualization Features

- **Red dots**: GPS coordinates (5x5 pixels minimum)
- **Red lines**: Path connections between consecutive points (thin, pure red)
- **Incremental updates**: Each `fetch_gps` adds new point and line
- **Storage**: Server saves to `data/server_paths/`, clients save to `data/client_paths/`

## Configuration

- **Default ports**: Server 5000, Device TCP 18001-18003
- **Session persistence**: `data/sessions.pkl` stores metadata and path image file paths
- **Coverage optimization**: Minimal grid sizes for efficient tile usage
- **Non-overlapping patches**: 100 patches exactly for 1000m coverage

## Troubleshooting

### Import Errors

- Server must run from project root: `python server/app.py`
- Device requires copied modules in `device/src/`

### Port Conflicts

- Use `tmux kill-server` to clean up processes
- Device uses high ports (18001-18003) to avoid conflicts

### Memory Issues

- Sessions store file paths, not binary data
- Large embeddings cached as JSON files
- Use `mode=server` for lightweight responses
