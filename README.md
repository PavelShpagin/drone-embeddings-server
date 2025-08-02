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
# Install dependencies and setup DINOv2 model
./setup_models.sh

# Or manually install dependencies
pip install -r requirements.txt
```

### 2. Setup Google Earth Engine Credentials

Place your service account key at:

```
secrets/earth-engine-key.json
```

### 3. Start Server

```bash
# Using the main server file
python server.py --port 5000

# Or using the launcher script
python run_server.py --port 5000
```

### 4. Test Server

```bash
# Test the API endpoints
python test_server.py --server http://localhost:5000

# Or run test client directly
python test/test_client_local.py --server http://localhost:5000
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

## Processing Pipeline

1. **Image Acquisition**

   - Calculate optimal grid size for desired coverage
   - Fetch satellite imagery using GEE API (parallel downloads)
   - Stitch tiles into seamless map

2. **Image Processing**

   - Crop to exact meter coverage
   - Extract 100x100 pixel patches (50m ground resolution)
   - Handle small areas by resizing when needed

3. **Embedding Generation**

   - Run DINOv2 model on each patch (224x224 input)
   - Generate 384-dimensional embeddings using Facebook's `dinov2_vits14`
   - Calculate GPS coordinates for each patch center

4. **Session Storage**
   - Store in hash table: `session_id → {map, patches, embeddings, gps}`
   - Persistent storage in binary format (`data/sessions.pkl`)
   - Each patch includes: embedding vector, lat/lng, image coordinates

## Data Storage Structure

The test client saves data in organized directories:

```
data/
├── maps/                           # Map images and metadata
│   ├── map_lat_lng_device_timestamp.jpg
│   └── map_lat_lng_device_timestamp_metadata.json
├── embeddings/                     # Embedding data
│   ├── map_lat_lng_device_timestamp_embeddings.json
│   └── map_lat_lng_device_timestamp_embeddings.npy
├── gee_api/                       # Downloaded satellite images
└── server_api/                    # Legacy format (deprecated)
```

### Maps Directory

- **Images**: High-quality JPEG satellite maps
- **Metadata**: JSON with GPS bounds, coverage, patch counts

### Embeddings Directory

- **JSON**: Structured data with GPS coordinates, embeddings, and statistics
- **NPY**: NumPy arrays for efficient loading in Python

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
├── run_server.py          # Server launcher
├── test_server.py         # Test runner
├── setup_models.sh        # Model setup script
├── requirements.txt       # Dependencies
├── Documentation.md       # This file
├── data/                  # Data storage
│   ├── sessions.pkl       # Persistent session storage
│   ├── maps/             # Map images and metadata
│   ├── embeddings/       # Embedding data
│   └── gee_api/          # Downloaded satellite images
├── src/                  # Source code
│   └── gee_sampler.py    # Google Earth Engine sampler
├── test/                 # Test scripts
│   └── test_client_local.py  # HTTP test client
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
   - Clear cache: `./setup_models.sh clean`

2. **GEE authentication fails**

   - Verify `secrets/earth-engine-key.json` exists and is valid
   - Check Google Cloud project has Earth Engine API enabled

3. **Out of memory errors**
   - Reduce coverage area (use smaller `meters` value)
   - Use CPU instead of GPU for DINOv2

### Support

For additional support:

- Check `/docs` endpoint for interactive API documentation
- Review logs for detailed error messages
- Test with `./setup_models.sh test` to verify model loading
