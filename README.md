# Satellite Embedding Server

FastAPI server for processing satellite imagery with DINOv2 embeddings.

## Overview

Server for processing satellite imagery with DINOv2 embeddings. Fetches high-resolution satellite imagery, extracts patches, generates embeddings using Facebook's DINOv2 model, and provides GPS-tagged data.

## Architecture

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

## Quick Start

### 1. Setup Models and Dependencies

```bash
# Install dependencies
pip install -r requirements.txt

# Install PyTorch CPU-only (for production)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. Setup Google Earth Engine Credentials

Place your service account key at:

```
secrets/earth-engine-key.json
```

Or use the setup script:

```bash
./secret.sh
```

### 3. Start Server

```bash
# Start the server
python server.py --port 5000
```

### 4. Test Server

```bash
# Test device mode with data storage
python test/test_fetch_gps_device.py

# Test complete simulation with path visualization
python test/test_simulation.py
```

## API Endpoints

### FastAPI Features

- **Interactive API Documentation**: `GET /docs` (Swagger UI)
- **Alternative API Docs**: `GET /redoc` (ReDoc)
- **OpenAPI Schema**: `GET /openapi.json`

### Core Endpoints

#### `GET /health`

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "sessions_count": 2,
  "server": "Satellite Embedding Server"
}
```

#### `POST /init_map`

Initialize a new map session with satellite imagery and DINOv2 embeddings.

**Request Body:**

```json
{
  "lat": 50.4162,
  "lng": 30.8906,
  "meters": 1000,
  "mode": "device"
}
```

**Parameters:**

- `lat` (float): Latitude of center point
- `lng` (float): Longitude of center point
- `meters` (int): Coverage area in meters (default: 2000m = 2km)
- `mode` (str): "server" or "device"

**Response (Server Mode):**

```json
{
  "session_id": "uuid-string",
  "success": true,
  "message": "Map session created with N patches",
  "coverage": "1000m x 1000m",
  "patch_count": 4
}
```

**Response (Device Mode):**

```json
{
  "session_id": "uuid-string",
  "success": true,
  "map_data": {
    "full_map": [...],  // Complete image as nested arrays
    "map_bounds": {
      "min_lat": 50.4117,
      "max_lat": 50.4207,
      "min_lng": 30.8836,
      "max_lng": 30.8976
    },
    "patches": [
      {
        "embedding": [...],  // 384-dimensional DINOv2 embedding
        "lat": 50.4162,
        "lng": 30.8906,
        "coords": [0, 0, 100, 100]  // Image coordinates [x1,y1,x2,y2]
      }
    ],
    "meters_coverage": 1000,
    "patch_count": 4
  }
}
```

#### `GET /sessions`

List all active sessions.

**Response:**

```json
{
  "success": true,
  "sessions": [
    {
      "session_id": "uuid-string",
      "created_at": 1654123456.789,
      "meters_coverage": 1000,
      "patch_count": 4,
      "map_bounds": {...}
    }
  ],
  "count": 1
}
```

#### `POST /fetch_gps`

Find GPS coordinates from an input image by matching its DINOv2 embedding against session embeddings.

**Request Body:** Form data with:

- `session_id` (string): Session ID to search within
- `image` (file): Image file to process

**Response:**

```json
{
  "success": true,
  "gps": {
    "lat": 50.4162,
    "lng": 30.8906
  },
  "similarity": 0.85,
  "confidence": "high"
}
```

#### `POST /visualize_path`

Returns the stored path visualization image with incremental red dots and connecting lines.

**Request Body:**

```json
{
  "session_id": "uuid-string"
}
```

**Response:** JPEG image with path visualization showing red dots connected by thin red lines (updated incrementally by fetch_gps calls)

## Processing Pipeline

1. **Map Initialization** (`init_map`)

   - Calculate minimal grid size for desired coverage
   - Fetch satellite imagery using GEE API (parallel downloads)
   - Crop to exact meter coverage and extract 100 patches (10x10 grid)
   - Generate 384-dimensional DINOv2 embeddings for each patch
   - Store in `data/sessions.pkl` with GPS coordinates

2. **GPS Matching** (`fetch_gps`)

   - Generate embedding for input image using DINOv2
   - Find closest patch by cosine similarity
   - Add new red dot to path visualization image (5x5 pixels minimum)
   - Draw thin red connecting line to previous point (if exists)
   - Store path image file in `data/server_paths/`
   - Update session data incrementally

3. **Path Visualization** (`visualize_path`)
   - Return stored path image from session data
   - Image updated incrementally by each `fetch_gps` call
   - Large red dots with thin red connecting lines (no borders)

## Data Storage Structure

Clean separation between server and client data:

```
data/
├── maps/                           # Full satellite map images
├── embeddings/                     # Patch embedding data (JSON)
├── server_paths/                   # Server-side path visualization images
├── client_paths/                   # Client-side received path images
└── sessions.pkl                    # Persistent session storage (file paths only)
```

### Maps Directory

- **Images**: High-quality JPEG satellite maps
- **Metadata**: JSON with GPS bounds, coverage, patch counts

### Embeddings Directory

- **JSON**: Structured data with GPS coordinates, embeddings, and statistics
- **NPY**: NumPy arrays for efficient loading in Python

### Path Visualization Directories

**Server Paths (`data/server_paths/`)**:
- Path visualization images maintained by server
- Updated incrementally with each `fetch_gps` call
- Red dots connected by thin red lines

**Client Paths (`data/client_paths/`)**:  
- Path images received by test clients
- Stored when calling `visualize_path` endpoint
- Clean separation from server-side storage

Example embedding JSON structure:

```json
{
  "session_id": "uuid",
  "location": {"lat": 50.4162, "lng": 30.8906},
  "embedding_info": {
    "total_patches": 4,
    "embedding_dimension": 384,
    "map_bounds": {...}
  },
  "patches": [
    {
      "patch_id": 0,
      "gps_coordinates": {"lat": 50.4162, "lng": 30.8906},
      "image_coordinates": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
      "embedding": [...],  // 384 float values
      "embedding_stats": {
        "min": -2.1,
        "max": 3.4,
        "mean": 0.12,
        "std": 0.67,
        "norm": 15.8
      }
    }
  ]
}
```

## DINOv2 Model Details

- **Model**: Facebook Research DINOv2-ViT-Small (`dinov2_vits14`)
- **Input Size**: 224×224 RGB images
- **Output**: 384-dimensional embeddings
- **Preprocessing**: ImageNet normalization + resize
- **Fallback**: Random embeddings if model fails to load
- **Device**: Automatic CUDA/CPU detection

## Directory Structure

```
server/
├── server.py              # Main FastAPI server
├── README.md              # Documentation (this file)
├── requirements.txt       # Dependencies
├── secret.sh              # GEE credentials setup script
├── data/                  # Data storage
│   ├── sessions.pkl       # Persistent session storage (file paths only)
│   ├── maps/             # Full satellite map images
│   ├── embeddings/       # Embedding data and metadata
│   ├── server_paths/     # Server-side path visualization images
│   └── client_paths/     # Client-side received path images
├── src/                  # Source code modules
│   ├── server_core.py    # Main server class
│   ├── models.py         # Data structures and Pydantic models
│   ├── embedder.py       # DINOv2 model wrapper
│   ├── init_map.py       # Map initialization logic
│   ├── fetch_gps.py      # GPS matching logic
│   ├── visualize_map.py  # Path visualization
│   ├── gee_sampler.py    # Google Earth Engine sampler
│   └── image_metadata.py # Image metadata extraction
├── test/                 # Test scripts
│   ├── test_fetch_gps_device.py  # Test device mode with storage
│   └── test_simulation.py        # Complete simulation test
└── secrets/              # Credentials (create this)
    └── earth-engine-key.json
```

## Usage Examples

### Initialize a Map Session

```bash
curl -X POST http://localhost:5000/init_map \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 50.4162,
    "lng": 30.8906,
    "meters": 1000,
    "mode": "device"
  }'
```

### Find GPS from Image

```bash
curl -X POST http://localhost:5000/fetch_gps \
  -F "session_id=your-session-id" \
  -F "image=@/path/to/your/image.jpg"
```

### Generate Path Visualization

```bash
curl -X POST http://localhost:5000/visualize_path \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your-session-id"}' \
  --output path_visualization.jpg
```

### Check Server Health

```bash
curl http://localhost:5000/health
```

### List Active Sessions

```bash
curl http://localhost:5000/sessions
```

## Deployment

This server folder is self-contained and can be deployed to any server:

1. **Copy the server folder** to your deployment server
2. **Setup models**: `./setup_models.sh`
3. **Add GEE credentials** to `secrets/earth-engine-key.json`
4. **Start server**: `python server.py --host 0.0.0.0 --port 5000`

### Production Deployment

For production, use uvicorn directly:

```bash
# Install production dependencies
pip install uvicorn[standard]

# Run with uvicorn (better performance)
uvicorn server:app --host 0.0.0.0 --port 5000 --workers 4

# Or use the server script
python server.py --host 0.0.0.0 --port 80
```

## Configuration

The server can be configured via command line arguments:

- `--port`: Port number (default: 5000)
- `--host`: Host address (default: 0.0.0.0)
- `--debug`: Enable debug mode

## Error Handling

The server includes comprehensive error handling:

- **Model Loading**: Falls back to random embeddings if DINOv2 fails
- **GEE API**: Retry logic for satellite image downloads
- **HTTP Errors**: Proper status codes and error messages
- **Session Storage**: Graceful handling of corrupted session files

## Performance Notes

- **DINOv2 Loading**: ~2-3 seconds on first run (downloads model)
- **Embedding Generation**: ~100ms per patch on CPU, ~10ms on GPU
- **Satellite Downloads**: 20-30 seconds for 1km area (parallel)
- **Memory Usage**: ~500MB base + ~2GB for DINOv2 model

## Troubleshooting

### Common Issues

1. **DINOv2 fails to load**

   - Check internet connection for model download
   - Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`
   - Restart server and check DINOv2 loading

2. **GEE authentication fails**

   - Verify `secrets/earth-engine-key.json` exists and is valid
   - Check Google Cloud project has Earth Engine API enabled

3. **Out of memory errors**
   - Reduce coverage area (use smaller `meters` value)
   - Use CPU instead of GPU for DINOv2

### Support

For additional support:

- Check `/docs` endpoint for interactive API documentation
- Review server logs for detailed error messages
